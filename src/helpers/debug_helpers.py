import torch
from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex, nvmlShutdown
import os


def print_gpu_util_info(device_index):
    if torch.cuda.device_count() >= (device_index + 1):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device_index)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"\nGPU {device_index} UTILIZATION:")
        print(f'total    : {info.total // 1024 ** 2} MB.')
        print(f'free     : {info.free // 1024 ** 2} MB.')
        print(f'used     : {info.used // 1024 ** 2} MB.')
        print(f"--> {round((info.used / info.total) * 100, 2)} %")
        print("\n")
        nvmlShutdown()


def get_ram_util_info():
    if not torch.cuda.is_available():
        total_memory, used_memory, free_memory = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
        print("\nMEMORY UTILIZATION:")
        print(f"RAM memory used: {round((used_memory / total_memory) * 100, 2)} %")


def debug(message=None, obj=None):
    if message is not None:
        print(message)
    if obj is not None:
        print(obj)
