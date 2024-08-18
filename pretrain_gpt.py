import math
import time
from dataclasses import dataclass

import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
import tiktoken
import os
import tiktoken

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not a bias but a mask
        self.register_buffer('bias',
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                                               config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # ATTENTION的两种实现, flash attention将加速x2 (论文中x7)
        # 1.origin attention impl
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill_(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 2.flash attention impl(用于替代origin的4行实现)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：输入token_embedding与输出token_embedding
        # 节省大量参数
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 与训练有关的部分
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available:
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # 早期torch版本的AdamW没有参数fused(cuda-only), 它可以略微提高训练速度
        fused_available = True  # 当前使用的torch2.2的AdamW有这个参数
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


def set_seed(seed: int = None):
    """设置随机种子方便重现"""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def auto_device():
    """自动选择计算设备"""
    device = "cpu"  # 默认cpu
    if torch.cuda.is_available():
        device = "cuda"  # nvidia的gpu
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"  # 苹果mac的gpu
    return device


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('dataset/shakspere.txt', 'r', encoding='utf') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]  # 这个+1是必须的,Y需要后移1位
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) >= len(self.tokens):  # 如果下一个batch凑不满, 即已遍历完数据集则重置
            self.current_position = 0
        return x, y


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderEdu:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "dataset/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        # 移位操作
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # 如果当前shard不足以支撑下一个batch, 则移动到新shard, 旧shard剩余token也直接丢掉了
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


def get_lr(iterate, max_lr=6e-4, warmup_steps=10, max_steps=50):
    # 大致模拟了一个先经过warmup_steps线性上升到max_lr, 再max_steps余弦平滑下降到min_lr的学习率变化过程, 接下来保持min_lr
    min_lr = max_lr * 0.1
    if iterate < warmup_steps:
        return max_lr * (iterate + 1) / warmup_steps
    if iterate > max_steps:
        return min_lr
    decay_ratio = (iterate - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr + (max_lr - min_lr) * coeff


def get_grad_accum_steps(total_tokens_per_batch, B=8, T=1024):
    """
    计算累计梯度更新步数：通过多次小batch反向传播累积梯度并在最后一次更新,来模拟一次大batch反向传播
    :param total_tokens_per_batch: 待模拟的大batch, 注意这里指的是tokens per batch, 例如50万token/batch
    :param B: 小batch_size, 指小batch中的sequence数量
    :param T: 时间步长(小batch中的token sequence序列长度)
    :return:
    """
    assert total_tokens_per_batch % (B * T) == 0  # 确保倍数关系
    grad_accum_steps = total_tokens_per_batch // (B * T)  # B*T即1个小batch中的tokens总数
    print(f"total desired token per batch : {total_tokens_per_batch}")
    print(f"calculated gradient accum steps : {grad_accum_steps}")
    return grad_accum_steps


def train_shakspere(max_steps=50):
    """
    小数据集的预训练：采用莎士比亚的文章作为预训练预料
    :param max_steps:
    :return:
    """
    # 训练的同时可以运行指令监控GPU(在Linux/Unix下)：watch -n 0.1 nvidia-smi
    set_seed(1337)
    device = auto_device()

    # 设置矩阵乘法精度(不设置的话, 默认是'highest'), 在4090提速x1.25, 但需要注意数据吞吐效率是否跟得上
    torch.set_float32_matmul_precision('high')

    # A100建议B=16, RTX4090建议B=8
    # 在GPT3论文中batch_size的含义是number of tokens per batch, 即batch_size=B*T, 不同规模模型batch_size取值0.5M~3.2M
    # 为了模拟这种超大的batch_size, 可以采用gradient_accumulation
    B, T = 8, 1024
    train_loader = DataLoaderLite(B=B, T=T)

    # Creates model and optimizer in default precision
    model = GPT(GPTConfig(vocab_size=50304))  # 随机初始化的模型, 50304为2的幂可以利用硬件并行略微提速
    model.to(device)

    # Linux环境可以使用compile编译, 提速x2以上
    # model = torch.compile(model)  #  Windows not yet supported for torch.compile

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    # 使用梯度累积来模拟大batch_size：当显存不足却又需要大batch时非常有用
    grad_accum_steps = get_grad_accum_steps(32768, B=B, T=T)  # 2**19=524288≈0.5M number of tokens per batch

    for step in range(max_steps):  # 每个step会使用1个batch的数据更新模型
        t_start = time.time()

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # Enables autocasting for the forward pass (model + loss), 提速接近x2
            # 为什么建议使用bfloat16做训练？
            # 1.bfloat16(分辨率0.01)比float16表示的精度低(0.001), 但数值跨度与float32一样大, 此外bfloat16可以表示的最小非零值远小于float16的最小非零值, 可以兼容训练过程中的溢出和下溢, 更好缓解梯度爆炸/消失
            # 2.虽然单个单个参数的梯度值不大, 但大量参数的累积效应大
            # 3.很可能训练过程中数值分会发生较大变化, 并不是一个较稳定的范围
            # 4.bfloat16有部分GPU做了加速优化在里面
            with torch.autocast(device_type=device, dtype=torch.bfloat16):  # 混合精度运算:提示性能的同时保持精度
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach().item()
            # Exits the context manager before backward()
            loss.backward()

        # 有效防止梯度爆炸：Total norm of the parameter gradients (viewed as a single vector)
        # 计算所有参数梯度的L2范数并使之小于max_norm, 如未超出不管, 如果超出则将所有梯度按比缩小到它们的L2范数为max_norm
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)  # 计算当前step的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 动态调整优化器中各参数组的学习率
        optimizer.step()  # 梯度下降, 更新参数

        torch.cuda.synchronize()  # 这里将等待GPU完成当前batch运算

        t_end = time.time()
        dt = t_end - t_start  # 单位：秒
        tokens_per_sec = (train_loader.B * train_loader.T) / dt
        print(
            f"step {step}, loss: {loss_accum}, lr: {lr}, norm: {norm}, dt: {dt * 1000:.2f}ms, tok/sec: {tokens_per_sec:}")


def train_fineweb10B(max_lr=6e-4, warmup_steps=140, max_steps=4750):
    """
    较大数据集的预训练, 训练的同时可以运行指令监控GPU(在Linux/Unix下)：watch -n 0.1 nvidia-smi
    :param max_lr: 最大学习率
    :param warmup_steps: 预热迭代步数(学习率从最小逐步上升,于此时达到最大,之后开始下降)
    :param max_steps: 训练的最大迭代步数(最多只训练这么多步)
    :return:
    """
    # 设置随机数种子：用于实验复现
    set_seed(1337)
    # 选择硬件：CPU/GPU/MPS
    device = auto_device()

    # 设置矩阵乘法精度(不设置的话, 默认是'highest'), 在4090提速x1.25, 但需要注意数据吞吐效率是否跟得上
    torch.set_float32_matmul_precision('high')

    # B为batch_size(即序列的数量, A100建议B=16, RTX4090建议B=8), T为序列长度(pretrain时一个batch中所有序列是一样长)
    # 在GPT3论文中batch_size的含义是number of tokens per batch, 即batch_size=B*T, 不同规模模型batch_size取值0.5M~3.2M, 为了模拟这种超大的batch_size, 可以采用gradient_accumulation
    B, T = 8, 1024  # tokens_per_batch = 8192

    # 训练数据集, 验证数据集(监控泛化能力和过拟合问题), 两数据集理论上要同分布
    train_loader = DataLoaderEdu(B=B, T=T, split='train')
    val_loader = DataLoaderEdu(B=B, T=T, split='val')

    # 分词器tokenizer初始化：效果预览时token->text需要用到
    enc = tiktoken.get_encoding('gpt2')

    # 若使用混合精度训练, 此处使用默认精度加载model即可：Creates model and optimizer in default precision
    model = GPT(GPTConfig(vocab_size=50304))  # 使用2的幂作为网络节点数可以略微提速：50304为2的幂
    model.to(device)  # 模型ship to GPU

    # Linux/Unix环境可以使用compile编译, 但Windows目前不支持, 可提速x2以上
    # model = torch.compile(model)

    # 初始化优化器：参数分组, 并对二维矩阵参数设置了weight_decay
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    # 梯度累积策略：使用梯度累积能来模拟大batch训练, 当显存不足却又需要大batch时非常有用
    # 注意：我设置的待模拟的大batch的tokens_per_batch=4*8*1024=32768, 但论文GPT-small的tokens_per_batch=2**19=524288≈0.5M
    grad_accum_steps = get_grad_accum_steps(32768, B=B, T=T)

    # 训练循环 training loop: 每个step会使用1个batch的数据更新模型
    for step in range(max_steps):
        # 阶段性验证：validate process
        if step % 100 == 0 or step == max_steps - 1:
            model.eval()
            val_loader.reset()  # 重置数据集loader
            with torch.no_grad():  # no_grad：推理阶段不计算梯度(节省大量显卡内存)
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    # 注意：除以val_loss_steps后, 加起来的val_loss_accum才表示平均每个batch(或平均每个step)的loss
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach().item()
            print(f"validation loss: {val_loss_accum:.4f}")

            # 阶段性保存模型 &配置 &状态：save checkpoints
            if step > 0 and (step % 2000 == 0 or step == max_steps - 1):
                ckpt_dor = 'checkpoints'
                os.makedirs(ckpt_dor, exist_ok=True)
                checkpoint_path = os.path.join(ckpt_dor, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum
                }
                # 如果想更精准的恢复训练, 还可以将optimizer.state_dict() 和 rng seeds 等等都加到checkpoint
                torch.save(checkpoint, checkpoint_path)
        # 阶段性生成文本预览效果：generate preview
        if (step > 0 and step % 100 == 0) or step == max_steps - 1:
            model.eval()  # 切换成推理模式
            num_return_sequences = 3
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)

            sample_rng = torch.Generator(device=device).manual_seed(42)  # 独立的随机数种子

            while xgen.size(1) < max_length:
                # 前向传播
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(xgen)  # (B, T, vocab_size)
                    # 仅取最后一个时间步(t=-1)的logits
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    # 计算5万多个tokens各自的softmax概率
                    probs = F.softmax(logits, dim=-1)
                    # 返回概率最大的top-k=50各token索引及其概率 (huggingface pipeline default), topk_probs.shape=(5, 50), topk_indices.shape=(5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # 采样下一个token, 基于它们的top-k概率(multinomial does not demand the input to sum to 1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                    # 返回batch中各sequence下一个token的词表索引indices
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    # 自回归：生成的token添加到输入序列尾部,作为新的输入,继续预测下一个token
                    xgen = torch.cat((xgen, xcol), dim=1)
            # 解码token->text, 并打印出来
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"sample {i}: {decoded}")

        # 训练部分:training process
        t_start = time.time()

        model.train()  # 切换成训练模式：影响normalization, dropout等机制

        optimizer.zero_grad()  # 梯度清零

        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # 混合精度运算:提示性能的同时保持精度, 提速接近x2, When entering an autocast-enabled region, Tensors may be any type. You should not call half() or bfloat16() on your model(s) or inputs when using autocasting.
            with torch.autocast(device_type=device, dtype=torch.bfloat16):  # 前向传播：forward pass(model + loss)开启autocasting
                logits, loss = model(x, y)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach().item()

            loss.backward()  # 反向传播：loss backward之前退出autocasting context

        # 梯度截断：计算所有参数梯度的L2范数并使之小于max_norm, 如未超出不管, 如果超出则将所有梯度按比缩小到它们的L2范数为max_norm, 有效防止梯度爆炸：Total norm of the parameter gradients (viewed as a single vector)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 学习率动态调整：每个step都动态计算当前step的学习率
        lr = get_lr(step, max_lr=max_lr, warmup_steps=warmup_steps, max_steps=max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 动态调整优化器中各参数组的学习率

        # 梯度下降: 更新参数
        optimizer.step()

        torch.cuda.synchronize()  # 这里将等待GPU完成当前batch运算, 用于计时器校准

        t_end = time.time()

        # 打印日志
        dt = t_end - t_start  # 单位：秒
        tokens_per_sec = (train_loader.B * train_loader.T) / dt
        print(
            f"step {step}, loss: {loss_accum}, lr: {lr}, norm: {norm}, dt: {dt * 1000:.2f}ms, tok/sec: {tokens_per_sec:}")


def simple_train_test():
    """开发过程中的验证：从莎士比亚数据集中取了一个batch来验证前向和反向传播是否成功, 误差是否在下降"""
    device = auto_device()

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    with open('dataset/shakspere.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    text = text[:1000]
    tokens = enc.encode(text)
    B, T = 4, 32
    buf = torch.tensor(tokens[:B * T + 1])
    x = buf[:-1].view(B, T).to(device)
    y = buf[1:].view(B, T).to(device)

    model = GPT(GPTConfig(vocab_size=50304))  # 随机初始化的模型, 50304为2的幂可以利用硬件并行略微提速
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for i in range(50):
        optimizer.zero_grad()  # 清空梯度
        logits, loss = model(x, y)  # 前向传播
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        print(f"step {i}, loss: {loss.item()}")


def simple_inference_test():
    """开发过程中的验证：用huggingface官方的gpt2模型weights注入自定义的backbone, 并测试是否能推理成功"""
    num_return_sequences = 5
    max_length = 30

    model = GPT.from_pretrained('gpt2')
    model.eval()

    device = auto_device()
    model.to(device)

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5,8)
    x = tokens.to(device)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)  是在指定维度上 从 topk_indices 张量中 根据索引 ix 选择元素
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)
    # 解码
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == '__main__':
    # simple_inference_test()  # 验证backbone是否能正确加载官方weights
    # simple_train_test()  # 验证模型的前向和反向传播是否正确, 可正常降低loss
    # train_shakspere()  # 小数据集pretrain: 莎士比亚文章
    train_fineweb10B()  # 大数据集pretrain: fineweb 10 billion 数据集


