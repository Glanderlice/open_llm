import time

import datasets
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from model.llama import LlamaTransformer
from model.lora import add_lora
from utils.nn_toolkit import auto_device, LengthBasedBatchSampler, get_lr, set_seed


def get_processor(tokenizer, shift=False):
    def process_function(sample):
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['instruction']}{sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt = tokenizer.encode(prompt, add_special_tokens=False)  # optional: +tokenizer.eos_token
        output = tokenizer.encode(f"{sample['output']}<|eot_id|>",
                                  add_special_tokens=False)  # optional: +tokenizer.eos_token
        input_ids = prompt + output
        labels = [-100] * len(prompt) + output
        if shift:
            input_ids = input_ids[:-1]
            labels = labels[1:]
        sample = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }
        return sample

    return process_function


def train_lora():
    set_seed(12345)
    device = auto_device()
    # model_path = 'D:/PycharmProjects/llama3_proj/models/Meta-Llama-3-8B-Instruct'
    model_path = "D:/PycharmProjects/LLM_Project/huanhuan_chat/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 这个从modelscope下载的llama3与HF不太一样
    tokenizer.pad_token = tokenizer.eos_token

    preprocess = get_processor(tokenizer, shift=True)

    ds = datasets.load_dataset('json', data_files={'train': 'D:/PycharmProjects/open_llm/dataset/huanhuan.json'})
    train_dataset = ds["train"]
    train_dataset = ds["train"].map(preprocess, remove_columns=train_dataset.column_names)

    batch_sampler = LengthBasedBatchSampler(train_dataset, batch_size=8, drop_last=True, shuffle=True)
    collate_fn = DataCollatorForSeq2Seq(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn
    )

    # iterator = iter(train_dataloader)
    # item = next(iterator)
    # print(item)

    model = LlamaTransformer.from_pretrained('llama-3-8B', local_path=model_path, torch_type=torch.bfloat16).to('cuda')
    add_lora(model, alpha=32, target=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
             dropout_p=0.05)

    # optimizer = get_optimizer(model, weight_decay=0.1, learning_rate=1e-4, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0
    )
    iterator = iter(train_dataloader)
    for step in range(400):
        # 训练部分:training process
        t_start = time.time()

        model.train()  # 切换成训练模式：影响normalization, dropout等机制

        optimizer.zero_grad()  # 梯度清零

        loss_accum = 0.0

        batch = next(iterator)
        x = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y, attention_mask)
        loss_accum = loss.detach().item()

        loss.backward()  # 反向传播：loss backward之前退出autocasting context

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 学习率动态调整：每个step都动态计算当前step的学习率
        lr = get_lr(step, max_lr=1e-4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 动态调整优化器中各参数组的学习率

        # 梯度下降: 更新参数
        optimizer.step()

        torch.cuda.synchronize()  # 这里将等待GPU完成当前batch运算, 用于计时器校准

        t_end = time.time()

        # 打印日志
        dt = t_end - t_start  # 单位：秒
        print(
            f"step {step}, loss: {loss_accum}, lr: {lr}, norm: {norm}, dt: {dt * 1000:.2f}ms")

        if step % 20 == 0:
            model.eval()
            dialogs = [[
                # {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
                {"role": "user", "content": "你是谁？"}
            ]]
            chats = tokenizer.apply_chat_template(dialogs, add_generation_prompt=True, tokenize=True)

            with torch.no_grad():
                for idx, chat in enumerate(chats):
                    tokens = torch.tensor(chat).long()
                    tokens = tokens.unsqueeze(0)
                    tokens = tokens.to("cuda:0")
                    outputs = model.generate(
                        input_ids=tokens,
                        max_new_tokens=50,
                        temperature=1,
                        top_k=50,
                        eos_token_id=tokenizer.encode('<|eot_id|>', add_special_tokens=False)[0]
                    )
                    print(outputs)
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Model output:\n{output_text}")

def train_batch():
    device = auto_device()
    set_seed(123)

    model_path = "D:/PycharmProjects/LLM_Project/huanhuan_chat/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 这个从modelscope下载的llama3与HF不太一样
    tokenizer.pad_token = tokenizer.eos_token

    preprocess = get_processor(tokenizer, shift=True)

    ds = datasets.load_dataset('json', data_files={'train': 'D:/PycharmProjects/open_llm/dataset/huanhuan.json'})
    train_dataset = ds["train"]
    train_dataset = ds["train"].map(preprocess, remove_columns=train_dataset.column_names)

    batch_sampler = LengthBasedBatchSampler(train_dataset, batch_size=8, drop_last=True, shuffle=True)
    collate_fn = DataCollatorForSeq2Seq(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn
    )

    iterator = iter(train_dataloader)
    batch = next(iterator)
    print(batch)

    model = LlamaTransformer.from_pretrained('llama-3-8B', local_path=model_path, torch_type=torch.bfloat16).to('cuda')
    add_lora(model, r=16, alpha=16, target=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
             dropout_p=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0
    )

    for step in range(100):
        # 训练部分:training process
        t_start = time.time()

        model.train()  # 切换成训练模式：影响normalization, dropout等机制

        optimizer.zero_grad()  # 梯度清零

        x = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y, attention_mask)
        loss_accum = loss.detach().item()

        loss.backward()  # 反向传播：loss backward之前退出autocasting context

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 学习率动态调整：每个step都动态计算当前step的学习率
        lr = get_lr(step, max_lr=1e-4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 动态调整优化器中各参数组的学习率

        # 梯度下降: 更新参数
        optimizer.step()

        torch.cuda.synchronize()  # 这里将等待GPU完成当前batch运算, 用于计时器校准

        t_end = time.time()

        # 打印日志
        dt = t_end - t_start  # 单位：秒
        print(
            f"step {step}, loss: {loss_accum}, lr: {lr}, norm: {norm}, dt: {dt * 1000:.2f}ms")

        if step % 20 == 0:
            model.eval()
            dialogs = [[
                # {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
                {"role": "user", "content": "你是谁？"}
            ]]
            chats = tokenizer.apply_chat_template(dialogs, add_generation_prompt=True, tokenize=True)

            with torch.no_grad():
                for idx, chat in enumerate(chats):
                    tokens = torch.tensor(chat).long()
                    tokens = tokens.unsqueeze(0)
                    tokens = tokens.to("cuda:0")
                    outputs = model.generate(
                        input_ids=tokens,
                        max_new_tokens=50,
                        temperature=1,
                        top_k=50,
                        eos_token_id=tokenizer.encode('<|eot_id|>', add_special_tokens=False)[0]
                    )
                    print(outputs)
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Model output:\n{output_text}")


if __name__ == '__main__':
    # local_model = "D:/PycharmProjects/llama3_proj/models/Meta-Llama-3-8B"
    # model = LlamaTransformer.from_pretrained("llama-3-8B", local_model).to('cuda')

    train_batch()
