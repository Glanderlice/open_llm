import os
import time

import numpy as np
import tiktoken
import torch
from torch.nn import functional as F

from model.gpt import GPT, GPTConfig
from utils.nn_toolkit import set_seed, auto_device, get_grad_accum_steps, get_lr, get_optimizer


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


def pretrain_shakspere(max_steps=50):
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
    optimizer = get_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

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


def pretrain_fineweb10B(max_lr=6e-4, warmup_steps=140, max_steps=4750):
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
            with torch.autocast(device_type=device,
                                dtype=torch.bfloat16):  # 前向传播：forward pass(model + loss)开启autocasting
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
    # pretrain_shakspere()  # 小数据集pretrain: 莎士比亚文章
    pretrain_fineweb10B()  # 大数据集pretrain: fineweb 10 billion 数据集
