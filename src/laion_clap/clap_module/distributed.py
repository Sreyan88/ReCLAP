import os

import torch
import socket
from typing import Callable

import torch
from loguru import logger
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.distributed.nn import all_gather as nn_all_gather


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all([var in os.environ for var in pmi_vars]):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    # print("//////////////////////////////////")
    # print(os.environ)
    local_rank = 0
    for v in ('SLURM_LOCALID', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            break
    global_rank = 0
    for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK'):
        if v in os.environ:
            global_rank = int(os.environ['RANK'])
            break
    world_size = 1
    for v in ('SLURM_NTASKS', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 8
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.horovod:
        assert hvd is not None, "Horovod is not installed"
        hvd.init()
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.local_rank = local_rank
        args.rank = world_rank
        args.world_size = world_size
        # args.local_rank = int(hvd.local_rank())
        # args.rank = hvd.rank()
        # args.world_size = hvd.size()
        args.distributed = True
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        print(f"Distributed training: local_rank={args.local_rank}, "
              f"rank={args.rank}, world_size={args.world_size}, "
              f"hostname={socket.gethostname()}, pid={os.getpid()}")
    elif is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        elif 'OMPI_COMM_WORLD_SIZE' in os.environ: # using Summit cluster
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            args.local_rank = local_rank
            args.rank = world_rank
            args.world_size = world_size
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True
        print(f"Distributed training: local_rank={args.local_rank}, "
              f"rank={args.rank}, world_size={args.world_size}, "
              f"hostname={socket.gethostname()}, pid={os.getpid()}")

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)
    return device


def launch(
    job_fn: Callable,
    num_machines: int = 1,
    num_gpus_per_machine: int = 1,
    machine_rank: int = 0,
    dist_url: str = "tcp://127.0.0.1:23457",
    args=(),
):
    """
    Launch a job in a distributed fashion: given `num_machines` machines, each
    with `num_gpus_per_machine` GPUs, this function will launch one process per
    GPU. This wrapper uses :func:`torch.multiprocessing.spawn`.

    The user has to launch one job on each machine, manually specifying a machine
    rank (incrementing integers from 0). This function will offset process ranks
    per machine. One process on `machine_rank = 0` will be the *main process*,
    and a free port on that machine will be used for process communication.

    Default arguments imply one machine with one GPU, and communication URL
    as `localhost`.

    .. note::

        We assume all machines have same number of GPUs per machine, with IDs as
        `(0, 1, 2 ...)`. If you do not wish to use all GPUs on a machine,
        set `CUDA_VISIBLE_DEVICES` environment variable appropriately.

    Args:
        job_fn: Function to launch -- this could be your model training function.
        num_machines: Number of machines, each with `num_gpus_per_machine` GPUs.
        num_gpus_per_machine: GPUs per machine, with IDs as `(0, 1, 2 ...)`.
        machine_rank: A manually specified rank of the machine, serves as a
            unique identifier and useful for assigning global ranks to processes.
        dist_url: Disributed process communication URL as `tcp://x.x.x.x:port`.
            Set this as the IP (and a free port) of machine with rank 0.
        args: Arguments to be passed to `job_fn`.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not found! Cannot launch distributed processes.")

    world_size = num_machines * num_gpus_per_machine

    # Spawn `num_gpus_per_machine` processes per machine, and provide
    # "local process rank" (GPU ID) as the first arg to `_dist_worker`.
    # fmt: off
    if world_size > 1:
        mp.spawn(
            _job_worker,
            nprocs=num_gpus_per_machine,
            args=(
                job_fn, world_size, num_gpus_per_machine, machine_rank, dist_url, args
            ),
            daemon=False,
        )
    else:
        # Default to single machine, single GPU, with ID 0.
        _job_worker(0, job_fn, 1, 1, 0, dist_url, args)
    # fmt: on


def _job_worker(
    local_rank: int,
    job_fn: Callable,
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int,
    dist_url: str,
    args: tuple,
):
    """
    Single distibuted process worker. This function should never be used directly,
    only used by :func:`launch`.
    """

    # Adjust global rank of process based on its machine rank.
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception as e:
        logger.error(f"Error launching processes, dist URL: {dist_url}")
        raise e

    synchronize()
    # Set GPU ID for each process according to its rank.
    torch.cuda.set_device(local_rank)
    job_fn(*args)


def synchronize() -> None:
    """Synchronize (barrier) all processes in a process group."""
    if dist.is_initialized():
        dist.barrier()


def get_world_size() -> int:
    """Return number of processes in the process group, each uses 1 GPU."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Return rank of current process in the process group."""
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process() -> bool:
    """
    Check whether current process is the main process. This check is useful
    to restrict logging and checkpointing to main process. It will always
    return `True` for single machine, single GPU execution.
    """
    return get_rank() == 0


def gather_across_processes(t: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather tensors from multiple GPU processes in a list. The order of elements
    is preserved by GPU process IDs. This operation is differentiable; gradients
    will be scattered back to devices in the backward pass.

    Args:
        t: Tensor to gather across processes.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [t]

    output = list(nn_all_gather(t))
    return output


def gpu_mem_usage() -> int:
    """
    Return gpu memory usage (in megabytes). If not using GPU, return 0 without
    raising any exceptions.
    """
    if torch.cuda.is_available():
        # This will be in bytes, so we divide by (1024 * 1024).
        return torch.cuda.max_memory_allocated() // 1048576
    else:
        return 0
