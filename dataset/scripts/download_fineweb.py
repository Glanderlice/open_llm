"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os

import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# 初始化tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token


def tokenize(doc):
    # 对单个文档进行分词(tokenize), 并转为numpy.uint16
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    text_tokens = enc.encode_ordinary(doc["text"])
    tokens.extend(text_tokens)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 ** 16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


shard_size = int(1e7)  # 1e7 is used due to limit of MEM. default is 1e8, 即100M tokens per shard, total of 100 shards


def download_dataset():
    remote_name = "sample-10BT"
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    return fw


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def doc_to_numpy(dataset, data_cache_dir):
    shard_index = 0
    token_count = 0
    progress_bar = None

    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # 临时的token缓存

    for row in dataset:
        tokens = tokenize(row)

        if token_count + len(tokens) < shard_size:
            # 如果没超出缓存长度则将tokens加入当前缓存
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)

            if progress_bar is None:  # update progress bar
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # 先保存缓存的tokens到当前的shard
            split = "val" if shard_index == 0 else "train"  # shard_0作为validate数据集
            filename = os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count  # all_tokens_np缓存 当前剩余的空间
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)

            # 开辟一个新的shard
            shard_index += 1
            progress_bar = None
            # 将之前多出来的remainder个tokens塞进新缓存
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # 将最终剩余的tokens写入最后一个shard(不一定被填满)
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])


if __name__ == '__main__':
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "dataset/edu_fineweb10B_1")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    # 从huggingface下载数据集(仅使用train分支)
    ds = download_dataset()
    # 将数据集以numpy格式写入本地文件
    doc_to_numpy(ds, DATA_CACHE_DIR)
