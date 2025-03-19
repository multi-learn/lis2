import os

from torch import distributed as dist


def get_rank():
    """
    Get the rank of the current process.
    Returns:
        str: 'cpu' if CUDA is not available, 'cuda' if there is a single GPU, or 'cuda:{rank}' for distributed GPUs.
    """
    if not torch.cuda.is_available():
        return "cpu"
    if torch.cuda.device_count() < 2:
        return "cuda"
    return "cuda:" + str(dist.get_rank())


def get_rank_num():
    """
    Get the rank number of the current process.
    Returns:
        int: 0 if CUDA is not available or there is a single GPU, or the rank number for distributed GPUs.
    """
    if not torch.cuda.is_available():
        return 0
    if torch.cuda.device_count() < 2:
        return 0
    return dist.get_rank()


def is_main_gpu():
    """
    Check if the current process is running on the main GPU.
    Returns:
        bool: True if CUDA is not available or there is a single GPU, or if the current process is the main process in distributed training.
    """
    if not torch.cuda.is_available():
        return True
    if torch.cuda.device_count() < 2:
        return True
    return dist.get_rank() == 0


def synchronize():
    """
    Synchronize all processes in distributed training.
    This function is a barrier that ensures all processes reach this point before proceeding.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size():
    """
    Get the total number of processes in distributed training.
    Returns:
        int: 1 if CUDA is not available or not in distributed mode, or the total number of processes in distributed training.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_sum(tensor):
    """
    Perform distributed sum reduction on the input tensor.
    Args:
        tensor (torch.Tensor): Input tensor to be summed across all processes.

    Returns:
        torch.Tensor: Resulting tensor after the sum reduction.
    """
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0)) 
        return s.getsockname()[1]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # os.environ['MASTER_PORT'] = str(find_free_port())
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    synchronize()
    dist.destroy_process_group()

import torch

def get_device_ids(gpus):
    """
    Returns the appropriate list of GPU IDs or 'cpu' based on the input argument.

    Args:
        gpus (int, list of int, str):
            - An integer representing a single GPU ID.
            - A list of integers representing multiple GPU IDs.
            - The string 'auto' to use all available GPUs.

    Returns:
        list of int or str: List of GPU IDs or 'cpu' if no GPUs are available.

    Raises:
        ValueError: If the `gpus` argument is not valid (not an int, list of ints, or 'auto').
        IndexError: If the GPU ID in the list exceeds the available GPU count.
    """
    if gpus == 'auto':
        if torch.cuda.is_available():
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                return [int(dev) for dev in visible_devices if dev.isdigit()]
            else:
                return list(range(torch.cuda.device_count()))
        else:
            return ['cpu']

    elif isinstance(gpus, int):
        if gpus < 0:
            raise ValueError("GPU ID cannot be negative.")
        if torch.cuda.is_available() and gpus <= torch.cuda.device_count():
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            return [gpus]
        else:
            return ['cpu']

    elif isinstance(gpus, list):
        if not all(isinstance(g, int) for g in gpus):
            raise ValueError("All elements in the GPU list must be integers.")
        if any(g < 0 for g in gpus):
            raise ValueError("GPU IDs cannot be negative.")
        if torch.cuda.is_available() and all(g < torch.cuda.device_count() for g in gpus):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            return gpus
        else:
            return ['cpu']

    else:
        raise ValueError("Invalid argument for `gpus`. Must be an integer, a list of integers, or 'auto'.")
