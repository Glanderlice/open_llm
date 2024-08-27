import copy
from typing import List

import torch
from torch import nn


class LoraLinear(nn.Module):
    """
    LoRA仅对线性层有效, 可以将线性层表示成两个小矩阵的矩阵乘积
    """

    def __init__(
            self,
            base_layer: nn.Linear,  # 原来的线性层
            r: int = 8,  # lora rank
            alpha: int = 16,  # lora alpha
            dropout_p: float = 0.0,  # lora dropout
            **kwargs
    ):
        super(LoraLinear, self).__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = float(alpha) / float(self.r)  # lora缩放系数用于微调强度控制：较大系数会增加LORA的影响, 较小反之
        self.dropout = nn.Dropout(dropout_p)
        self.base_layer = copy.deepcopy(base_layer)
        # 冻结原来的层的参数
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # 定义 lora_A 和 lora_B 为 Parameter
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False, dtype=base_layer.weight.dtype)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False, dtype=base_layer.weight.dtype)

        # 初始化矩阵A和B, 其中B初始为0矩阵, 这样能使刚开始时LoRA不会对结果产生影响, 直到反向传播生效
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

        if kwargs.get("demo", False):  # B矩阵随机初始化主要是单元测试时使用, 传参demo=True激活单元测试
            # DEMO mode is activated: weight B is randomly initialized
            nn.init.normal_(self.lora_B.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xl = self.lora_A(self.dropout(x))
        xl = self.lora_B(xl)
        return self.base_layer(x) + xl * self.scaling


def add_lora(
        module: nn.Module,
        r: int = 8,
        alpha: int = 16,
        dropout_p: float = 0.0,
        target: str | List[str] = 'proj',
        **kwargs
):
    """
    在原来model的基础上加lora, in-place操作, 会直接替换原model的子模块。只要target出现在Linear module名称中即算命中
    注意：LoRA机制只对线性层有效, target因此必须标记线性层的名称
    :param device:
    :param module: 原module, 一般指大模型
    :param r: 秩
    :param alpha: 缩放因子
    :param dropout_p: dropout probability
    :param target: 标记tag, 默认'proj'指全部线性层, 可传str/list, 例如：'k_proj' 或 ['k_proj', 'q_proj', 'v_proj']
    :return: None
    """
    if isinstance(target, str):
        target = [target]

    assert isinstance(target, list), 'parameter target should be a list(str) or str'

    # 注意：.named_modules()递归返回所有的子模块, 而.named_children()仅返回直接子模块
    for name, child in module.named_children():
        if len(list(child.named_children())) > 0:  # 1.模块非叶子节点->继续递归
            add_lora(child, r, alpha, dropout_p, target, **kwargs)
        elif isinstance(child, nn.Linear) and any(s in name for s in target):  # 2.线性层 && 模块名符合target, 例如：k_proj->加LoRA
            device = child.weight.device
            lora_linear = LoraLinear(child, r=r, alpha=alpha, dropout_p=dropout_p, **kwargs).to(device)
            setattr(module, name, lora_linear)  # 植入当前module, 替换原线性层
        else:  # 当模块是叶子节点 且 非LoRA目标->冻结参数
            for param in child.parameters():
                param.requires_grad = False

    return module


def save_lora_weights(module: nn.Module, adapter_path: str, recover=True):
    """
    卸载 lora 参数，并将原模型恢复至加载 lora 前的样子
    """
    lora_parameters = {}  # 所有LoRA层全部放进去, key包含了它的层级关系

    def search_lora_linear(module: nn.Module, prefix: List[str]):
        for name, child in module.named_children():
            new_prefix = prefix + [name]
            if isinstance(child, LoraLinear):
                # 保存 lora 参数
                lora_name = '.'.join(new_prefix)
                lora_parameters[lora_name] = {
                    "lora_A_weight": child.lora_A.weight.data.cpu(),
                    "lora_B_weight": child.lora_B.weight.data.cpu(),
                    "r": child.r,
                    "alpha": child.alpha,
                    "dropout_p": child.dropout.p,
                }
                if recover:  # 将model恢复成LoRA之前的状态
                    setattr(module, name, child.base_layer)
            else:
                search_lora_linear(child, new_prefix)  # DFS递归

    search_lora_linear(module, [])

    if recover:
        for name, param in module.named_parameters():  # 解冻原模型
            param.requires_grad = True

    torch.save(lora_parameters, f"{adapter_path}")


def load_lora_weights(module: nn.Module, adapter_path: str):
    """
    加载 lora 参数
    """
    lora_parameters = torch.load(f"{adapter_path}")

    # 先冻结模型全部参数
    for param in module.parameters():
        param.requires_grad = False

    for name, lora_params in lora_parameters.items():
        child = dict(module.named_modules())[name]  # name示例：layers.0.self_attn.q_proj
        if isinstance(child, nn.Linear):
            device = child.weight.device  # 与线性层放在同一device
            lora_linear = LoraLinear(child, lora_params['r'], lora_params['alpha'], lora_params['dropout_p'])
            lora_linear.lora_A.weight.data = lora_params["lora_A_weight"]
            lora_linear.lora_B.weight.data = lora_params["lora_B_weight"]
            lora_linear.to(device)

            # 名称示例：layers.0.self_attn.q_proj
            parts = name.split(".")
            obj = module
            for part in parts[:-1]:  # 根据名称找到倒数第二个module(因此这里没包含最后1个元素), 比如：self_attn
                obj = getattr(obj, part)
            para_name = parts[-1]
            setattr(obj, para_name, lora_linear)  # 最后一级parts[-1]即属性名, 例如：k_proj
