import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from utils.nn_toolkit import auto_device


@dataclass
class ModelArgs:
    dim: int = 4096  # 隐层dimension
    n_layers: int = 32  # decoder个数
    n_heads: int = 32  # 多头注意力个数
    n_kv_heads: Optional[int] = None  # kv注意力个数(GQA)
    vocab_size: int = -1  # 词表大小
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0  # 相对位置编码参数(底数)

    max_batch_size: int = 32
    max_seq_len: int = 2048  # 输入最大序列长度

    intermediate_size: int = 14336  # FFN(或MLP)层的隐层节点数
    padding_idx: Optional[int] = None  # Embedding层的可选配置参数, 对于的向量不会被更新


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):  # 即最后一维的向量, 求其平方和的均值再开方得RMS(X), 再乘以可训练的W矩阵：W·X/RMS(X)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000., device=None, scaling_factor=1.0):
        super().__init__()
        if device is not None:
            self.device = device = auto_device()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size], position_ids: (bs, seq_len)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1,
                                                                        1)  # ->(bs, dim/2, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # (bs, seq_len)->(bs, 1, seq_len)
        # Force float32 since bfloat16 loses precision on long contexts
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,
                                                                                          2)  # ->(bs, seq_len, dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)  # ->(bs, seq_len, dim)
            cos = emb.cos()  # (bs, seq_len, dim)
            sin = emb.sin()  # (bs, seq_len, dim)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """旋转辅助函数：把x最后维度(词向量)从中间一分为二:x1,x2, 再移位拼接成[-x2,x1], 输出的shape等于输入的shape"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """对q,k tensors应用Rotary Position Embedding
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1)
    Returns:
        `tuple(torch.Tensor)` 旋转后的q,k
    """
    # q=(bs,n_h,seq_len,dim)
    cos = cos.unsqueeze(unsqueeze_dim)  # 在dim=1处插入新维度(bs,seq_len,dim)->(bs,1,seq_len,dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 应用旋转到q,k向量
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    等效于torch.repeat_interleave(x, dim=1, repeats=n_rep)
    (batch, num_key_value_heads, seq_len, head_dim)->(batch, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None) -> torch.Tensor:
    # 此为官方示例代码：Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ModelArgs, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.dim
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads  # q是多头
        self.num_key_value_heads = config.n_kv_heads  # k,v缩减策略：k,v头比q少(减少开销的同时不会明显影响模型效果)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_seq_len
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    # Adapted from LlamaAttention.forward
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # b,n_h,seq,dim
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # repeat_kv用于复制k,v张量, 使之与q对齐
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # method1:
        # self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill_(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # method2:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaDecoderLayer(nn.Module):
    def __init__(self, layer_id: int, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.dim

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_id)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs


def _update_causal_mask(attention_mask: torch.Tensor, input_tensor: torch.Tensor):
    """
    融合causal_mask + attention_mask
    :param attention_mask: 有效token标记1, 无效token(例如 padding)标记0
    :param input_tensor: token embedding, (batch,seq_len,dim)
    :return:
    """
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    assert input_tensor.shape[1] == attention_mask.shape[-1]
    sequence_length = input_tensor.shape[1]  # 获取seq_len

    causal_mask = torch.full((sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        if attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask.eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)
        else:
            raise ValueError("attention_mask must be 2D tensor")

    return causal_mask


class LlamaModel(nn.Module):

    def __init__(self, params: ModelArgs):
        super().__init__()
        # padding_idx对应的pad_emb向量为全零向量, 且不会更新梯度, 但可以手动换成其它取值的向量
        self.padding_idx = params.padding_idx
        self.embed_tokens = nn.Embedding(params.vocab_size, params.dim, padding_idx=self.padding_idx)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(LlamaDecoderLayer(layer_id, params))

        self.norm = LlamaRMSNorm(params.dim, eps=params.norm_eps)

    def forward(self, tokens: torch.Tensor, position_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        h = self.embed_tokens(tokens)

        causal_mask = _update_causal_mask(attention_mask, h) if attention_mask is not None else None

        for layer in self.layers:
            h = layer(h, position_ids, causal_mask)
        h = self.norm(h)
        return h


def _init_weights(module):
    if isinstance(module, nn.Linear):
        std = 0.02
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class LlamaTransformer(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 注意：Llama3的token embedding与output embedding "不"共享权重
        # [X] (self.model.embed_tokens.weight = self.lm_head.weight)

        # init params
        self.apply(_init_weights)

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None, shift: bool = False, fast_inference: bool = True):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask
        )

        if labels is not None:  # 与训练有关的部分
            logits = self.lm_head(hidden_states)  # (B, T, vocab_size)
            # logits = logits.float()  # 这里是为了在计算loss用于训练时保持精度
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))  # ignore_index=-1, 默认-100
        else:  # 推理：只保留最后一个时间步可以优化推理耗时
            if fast_inference:
                hidden_states = hidden_states[:, [-1], :]
            logits = self.lm_head(hidden_states)  # [-1]可以保留时间维度, 如果是-1则不会保留
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, eos_token_id=None, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = input_ids
        for _ in range(max_new_tokens):
            # 截断：crop it > max_seq_len
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            # 前向传播：logits for the index in the sequence
            logits, _ = self(idx_cond)  # logits.shape=(B,T,vocal_size)
            # T值：T越大, logits之间差距越小, token预测越随机
            logits = logits[:, -1, :] / temperature  # logits.shape: (B,1,vocal_size)->(B,vocal_size)
            # top_k：optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')  # 注意 v[:, [-1]]]返回(B,1), 然而v[:, -1]会返回(B,)
            # softmax: apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # do sampling: sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == eos_token_id:
                break

            # autoregression：append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def from_pretrained(cls, model_name='meta-llama/Meta-Llama-3-8B', local_path=None, torch_type=None):
        from transformers import AutoModelForCausalLM

        model_name = {
            'llama-3-8B': 'meta-llama/Meta-Llama-3-8B'
        }[model_name]

        # n_layers, n_head and n_embd are determined from model_name
        config_args = {
            'meta-llama/Meta-Llama-3-8B': dict(n_layers=32, n_heads=32, dim=4096, n_kv_heads=8)
        }[model_name]
        config_args['vocab_size'] = 128256
        config_args['max_seq_len'] = 2048
        print(f"loading weights from pretrained model: {model_name} => {config_args}")

        config = ModelArgs(**config_args)
        model = LlamaTransformer(config).to(torch_type)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.bias')]  # discard bias
        print(f"parameters num: {len(sd_keys)}")

        model_path = local_path if local_path else model_name
        model_hf = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in tqdm(sd_keys_hf):
            if k not in sd_keys:
                print(f"[warning] - {k} not in your structure")
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        # 释放显存
        del model_hf
        torch.cuda.empty_cache()

        return model
