import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch

print(torch.cuda.is_available())
