import random
import torch
import numpy as np

from agent.classifiers.utils.device_mgmt import is_a_cuda_device


def set_random_seed(seed_val, use_gpu):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    # para que "torch.use_deterministic_algorithms(True)" funcione:
    # nano ~/.bash_profile
    # export CUBLAS_WORKSPACE_CONFIG=":4096:8"
    # source ~/.bash_profile

    torch.use_deterministic_algorithms(True)
    if use_gpu and is_a_cuda_device():
        torch.cuda.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
