from transformers import AutoModelForCausalLM

from model.lora import add_lora, save_lora_weights, load_lora_weights
from utils.nn_toolkit import set_seed, auto_device, print_trainable_parameters, print_model_all_parameters
import torch


def unit_test():
    set_seed(12345)
    device = auto_device()

    model_path = 'D:/PycharmProjects/llama3_proj/models/Meta-Llama-3-8B-Instruct'

    # 构造伪输入
    test_input = torch.randint(0, 10000, (2, 8)).to(device)

    # 原始model
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    print(model)

    # forward并获取最后一个隐层的值, 即im_head的前一层
    model.eval()
    with torch.no_grad():
        res1 = model(test_input, output_hidden_states=True)
    res1 = res1.hidden_states[-1]

    # model加LoRA
    add_lora(model, demo=True)
    print(model)
    print_trainable_parameters(model)
    print_model_all_parameters(model)

    model.eval()
    with torch.no_grad():
        res2 = model(test_input, output_hidden_states=True)
    res2 = res2.hidden_states[-1]
    # 在lora的影响下理论上应该不一样
    print("llama3 vs llama3_add_lora last hidden state (expect False)：", torch.allclose(res1, res2, atol=1e-6))

    # 保存lora
    save_lora_weights(model, "lora.pt", recover=True)  # recover=True将使model恢复成加lora前的状态
    with torch.no_grad():
        res3 = model(test_input, output_hidden_states=True)
    res3 = res3.hidden_states[-1]
    # 恢复后的推理结果跟原始model理论上应该一样
    print("llama3 vs llama3_remove_lora last hidden state  (expect True)：", torch.allclose(res1, res3, atol=1e-6))

    # 释放显存
    del model
    torch.cuda.empty_cache()

    # 重新初始化llama3, 并从文件加载lora
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    load_lora_weights(model, "lora.pt")

    with torch.no_grad():
        res4 = model(test_input, output_hidden_states=True)
    res4 = res4.hidden_states[-1]
    # 加载lora后的推理结果跟原始model理论上应该不同
    print("llama3 vs llama3_load_lora last hidden state (expect False)：", torch.allclose(res1, res4, atol=1e-6))
    # 加载lora的推理结果跟第一次添加lora的结果理论上应该一样
    print("llama3_add_lora vs llama3_load_lora last hidden state (expect True)：", torch.allclose(res2, res4, atol=1e-6))


if __name__ == '__main__':
    unit_test()
