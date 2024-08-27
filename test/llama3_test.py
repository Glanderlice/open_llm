from transformers import AutoModelForCausalLM
from torch.nn import functional as F
from model.llama import LlamaTransformer
from utils.nn_toolkit import set_seed, auto_device
import torch


def consistency_test():
    set_seed(12345)
    device = auto_device()

    model_path = 'D:/PycharmProjects/llama3_proj/models/Meta-Llama-3-8B'

    # 构造伪输入
    test_input = torch.randint(0, 10000, (1, 4)).to(device)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    with torch.no_grad():
        res = model(test_input, output_hidden_states=True)
    print(res.logits)

    # model = LlamaTransformer.from_pretrained("llama-3-8B", model_path, torch.bfloat16).to(device)
    # model.eval()
    # with torch.no_grad():
    #     logits, _ = model(test_input)
    # print(logits)


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def hellaswag_test():
    from utils.hellaswag import render_example, iterate_examples
    val = list(enumerate(iterate_examples("val")))
    len(val)
    from transformers import AutoModelForCausalLM

    local_model = "D:/PycharmProjects/llama3_proj/models/Meta-Llama-3-8B"
    # llama3 = AutoModelForCausalLM.from_pretrained(local_model, torch_dtype=torch.bfloat16).to('cuda')
    llama3 = LlamaTransformer.from_pretrained("llama-3-8B", local_model, torch_type=torch.bfloat16).to('cuda')

    from tqdm import tqdm

    num_total = 0
    num_correct_norm = 0

    llama3.eval()

    # 创建一个进度条
    for i, example in tqdm(val, desc="Processing", unit="example"):
        # only process examples where i % ddp_world_size == ddp_rank
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to("cuda")
        mask = mask.to("cuda")
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # logits = llama3(tokens).logits  # HF-llama3 => HellaSwag accuracy: 7037/10042=0.7008
                logits, _ = llama3(tokens, fast_inference=False)  # 自定义llama3 => HellaSwag accuracy:
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    acc_norm = num_correct_norm / num_total
    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")


if __name__ == '__main__':
    hellaswag_test()
