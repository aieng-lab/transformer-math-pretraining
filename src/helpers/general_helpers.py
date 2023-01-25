import pathlib
from pathlib import Path
from collections.abc import MutableMapping
import torch
from pynvml import *
import logging
import psutil
import threading

logging.basicConfig()

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def create_path_if_not_exists(path, is_file=False):
    if is_file:
        parent = pathlib.Path(path).parent.absolute()
        pathlib.Path(parent).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            pass
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def path_exists(path_to_file):
    file = Path(path_to_file)
    return file.exists()


def flatten(d: dict, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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


def print_all_gpu_util_info():
    for i in range(torch.cuda.device_count()):
        print_gpu_util_info(i)


def print_ram_util_info():
    if not torch.cuda.is_available():
        total_memory, used_memory, free_memory = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
        print("\nMEMORY UTILIZATION:")
        print(f"RAM memory used: {round((used_memory / total_memory) * 100, 2)} %")


def print_memory_info(used=False, available=False):
    memory = psutil.virtual_memory()
    percentage = memory.percent
    available_mem = memory.available
    used_mem = memory.used
    print(f"MEMORY UTILIZATION:")
    print(f"{percentage}%")
    if used:
        print(f"Used: {round(used_mem / (1000 ** 3), 2)} GB")
    if available:
        print(f"Available: {round(available_mem / (1000 ** 3), 2)} GB")


def print_process_memory_info():
    pid = os.getpid()
    python_process = psutil.Process(pid)
    #memory_use = python_process.memory_info()[0]/2.**30
    #print(f"\nMemory use of process {pid}: {memory_use} GB")
    print(f"Memory of process {pid}:")
    print(f"{round(psutil.Process(pid).memory_info().rss / 1024 ** 3, 2)} GB")





def print_thread_num():
    print(f"Active threads: {threading.active_count()}")


def find_file_by_name(dir_path, file_name):
    if not file_name.__contains__("."):
        file_name = f"{file_name}.*"
    found = []
    for result in Path(dir_path).glob(file_name):
        found.append(result)
    if len(found) == 1:
        return found[0]
    elif len(found) == 0:
        return None
    else:
        _logger.warning(f"Found several files matching '{file_name}' in directory '{dir_path}'")
        return found


def get_max_num_of_workers():
    return len(os.sched_getaffinity(0))


def get_size_in_bytes(path):
    if os.path.isfile(path):
        size = os.stat(path).st_size
    else:
        size = 0
        for sub_path, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(sub_path, f)
                size += os.stat(fp).st_size
    return size


def print_size(path):
    if os.path.isfile(path):
        size = os.stat(path).st_size
    else:
        size = 0
        for sub_path, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(sub_path, f)
                size += os.stat(fp).st_size

    byte_size = size
    mb_size = size / (1000 ** 2)
    gb_size = mb_size / 1000
    print(f"\nSize of {path}:")
    print(f"{byte_size} Bytes")
    print(f"{round(mb_size, 2)} MB")
    print(f"{round(gb_size, 2)} GB")


if __name__ == "__main__":
    d = {"bert_input": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": {"mlm": [1, 0, 3, 4], "is_next": 0}}
    flat_d = flatten(d)
    print(d)
    print(flat_d)

    found = find_file_by_name("/home/katja/singularity/python-images/transformer_pretraining/python/src/books",
                              "config")
    print(found)

    cpu_count = get_max_num_of_workers()
    print(cpu_count)

    print_size("/home/katja/singularity/python-images/transformer_pretraining/python/train_data/wiki")
