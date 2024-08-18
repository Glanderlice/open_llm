import torch
from torch.backends import cudnn
# 判断硬件是否支持cudnn
print(cudnn.is_available())

import transformers
# 判断硬件是否支持bfloat16
print(transformers.utils.import_utils.is_torch_bf16_gpu_available())