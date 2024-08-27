import math
import random

import numpy as np
import torch


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


def check_hardware():
    from torch.backends import cudnn
    # 判断硬件是否支持cudnn
    print('cudnn is supported?', cudnn.is_available())

    import transformers
    # 判断硬件是否支持bfloat16
    print('bfloat16 is supported?', transformers.utils.import_utils.is_torch_bf16_gpu_available())


# def clear(model):
#     del model
#     torch.cuda.empty_cache()


def print_trainable_parameters(model: torch.nn.Module):
    """
    打印可训练参数，类似PeftModel 的 print_trainable_parameters 方法
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = 100 * trainable_params / total_params

    # 返回可训练参数量、所有参数量、可训练参数量占比（百分比）
    print(
        f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.4f}")


def print_model_all_parameters(model):
    """
    查看模型参数的 requires_grad 情况
    """
    print("Layer Name & Parameters")
    print("----------------------------")
    for name, parameter in model.named_parameters():
        print(f"{name:50} | dtype:{parameter.dtype} | Requires_grad: {parameter.requires_grad}")


def get_optimizer(model: torch.nn.Module, weight_decay, learning_rate, device):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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

    fused_available = True  # 当前使用的torch2.2的AdamW有这个参数, 早期torch版本的AdamW没有参数fused(cuda-only), 它可以略微提高训练速度
    use_fused = fused_available and device == "cuda"
    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer


def get_lr(iterate, max_lr=6e-4, warmup_steps=10, max_steps=50):
    # 模拟先经过warmup_steps线性上升到max_lr, 再max_steps余弦平滑下降到min_lr的学习率变化过程, 接下来保持min_lr
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


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    """
    构造一个自定义的batch sampler：它将dataset里所有样本按length升序排序, 然后把相邻的样本按batch_size组成一个batch返回
    注意：返回的batch(当batch_size>1)依然长度参差不齐, 还需要后续collate_fn完成padding操作
    """

    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')

        # np.random.shuffle(ids)  # 随机打乱：测试padding时使用

        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)
