import torch
import platform
import logging

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from agent.data.entities.config import ROOT_LOGGER_ID

logger = logging.getLogger(ROOT_LOGGER_ID)

MACOS_SYSTEM = "Darwin"

CUDA_GPU = "cuda:0"
APPLE_GPU = "mps"
CPU = "cpu"


def get_device(use_gpu):    
    platform_system = platform.system()   

    if use_gpu:
        if platform_system == MACOS_SYSTEM:
            logger.info("System %s, using device %s", platform_system, APPLE_GPU)
            return torch.device(APPLE_GPU)
        else:
            logger.info("System %s, using device %s", platform_system, CUDA_GPU)
            return torch.device(CUDA_GPU)
    
    logger.info("System %s, using device %s", platform_system, CPU)
    return torch.device(CPU)


def is_a_cuda_device():
    return platform.system() != MACOS_SYSTEM


def get_gpu_total_memory():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    available_memory = info.total // 1024 ** 2
    logging.info("GPU memory: %s", available_memory)
    return available_memory


def get_gradient_accumulation_steps(batch_size):
    gradient_accumulation_steps = 1
    batch_size2 = batch_size
    if get_gpu_total_memory() < 24000:
        gradient_accumulation_steps = 2
        batch_size2 = batch_size // gradient_accumulation_steps
        logging.info("GPU memory too small, using gradient accumulation steps %s and batch size %s",
                     gradient_accumulation_steps, batch_size2)
    return gradient_accumulation_steps, batch_size2
