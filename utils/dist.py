import logging
import os

import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_deepspeed_config(
    train_local_batch_size: int,
    gradient_accumulation_steps: int,
    gradient_clipping: float = -1,
):
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": train_local_batch_size,
        "zero_optimization": {
            "stage": 2,
            # "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,  # default
            "reduce_bucket_size": 5e8,  # default
            "allgather_bucket_size": 5e8,  # default
            "round_robin_gradients": True,
        },
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16": {"enable": True},
        "zero_allow_untested_optimizer": True,
        "flops_profiler": {
            "enabled": False,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 3,
            "detailed": True,
        },
    }
    if gradient_clipping > 0:
        deepspeed_config["gradient_clipping"] = gradient_clipping
    return deepspeed_config


def dist_init():
    try:
        logger.info(
            f"local rank: {os.environ.get('LOCAL_RANK', '')}, global rank: {os.environ.get('RANK', '')}, world size: {os.environ.get('WORLD_SIZE', '')}, master addr: {os.environ.get('MASTER_ADDR', '')}, master port: {os.environ.get('MASTER_PORT', '')}, mpi rank: {os.environ.get('OMPI_COMM_WORLD_RANK', '')}, mpi local rank: {os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '')}, mpi world size: {os.environ.get('OMPI_COMM_WORLD_SIZE', '')}"
        )
    except Exception as e:
        logger.warning(f"Failed to print environment variables: {e}")

    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        if os.environ["LOCAL_RANK"] != os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"):
            os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ["LOCAL_RANK"]
            logger.info(f"Set OMPI_COMM_WORLD_LOCAL_RANK to {os.environ['LOCAL_RANK']}")

    #     os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "")
    #     os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE")


def dist_init_wo_accelerate():
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    master_uri = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method=master_uri
    )
