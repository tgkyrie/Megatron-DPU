from typing import Dict, Optional, List
import torch

import byteps.torch as bps
from byteps.torch import ops as bps_ops

from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_data_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_data_parallel_world_size,
    get_pipeline_model_parallel_world_size,
)


def _get_rank_prefix():
    try:
        if torch.distributed.is_initialized():
            return f"[Rank {torch.distributed.get_rank()}]"
    except:
        pass
    return "[Rank ?]"


_DECLARED_BPS_GROUPS: Dict[str, int] = {}


def build_byteps_group_name(scope: str, logical_name: str) -> str:
    if not logical_name:
        raise ValueError("logical_name must be a stable non-empty string")

    pp_rank = get_pipeline_model_parallel_rank()

    if scope == "dp":
        # DP gradients are reduced across the DP group.
        # All ranks in the same DP group should have the same BytePS name.
        # DP group is formed by ranks with the same tp_rank (within each TP group).
        # So we use tp_rank to distinguish different DP groups.
        # For example, with TP_SIZE=2, DP_SIZE=2:
        #   - DP group 0: ranks with tp_rank=0 (e.g., Rank 0 and Rank 2)
        #   - DP group 1: ranks with tp_rank=1 (e.g., Rank 1 and Rank 3)
        tp_rank = get_tensor_model_parallel_rank()
        # CP rank varies inside the collective and must not be part of the BytePS name.
        return f"dp.tp{tp_rank}.pp{pp_rank}.{logical_name}"
    if scope == "tp":
        # TP gradients are reduced across the TP group.
        # All ranks in the same TP group should have the same BytePS name.
        # TP group is formed by ranks with the same dp_rank (across DP groups).
        # So we use dp_rank to distinguish different TP groups.
        # For example, with TP_SIZE=2, DP_SIZE=2:
        #   - TP group 0: ranks with dp_rank=0 (e.g., Rank 0 and Rank 1)
        #   - TP group 1: ranks with dp_rank=1 (e.g., Rank 2 and Rank 3)
        dp_rank = get_data_parallel_rank()
        cp_rank = get_context_parallel_rank()
        return f"tp.dp{dp_rank}.pp{pp_rank}.cp{cp_rank}.{logical_name}"

    raise ValueError(f"Unsupported BytePS scope: {scope}")


def declare_and_cache_byteps_group(name: str, expected_workers: int) -> None:
    cached_workers = _DECLARED_BPS_GROUPS.get(name)
    if cached_workers is None:
        bps.declare(name, expected_workers=expected_workers)
        _DECLARED_BPS_GROUPS[name] = expected_workers
        return
    if cached_workers != expected_workers:
        raise RuntimeError(
            f"BytePS tensor name {name} was declared with inconsistent group size: "
            f"{cached_workers} vs {expected_workers}"
        )


def _declare_group_and_get_name(group, scope: str, logical_name: str) -> str:
    if group is None:
        raise ValueError("group must not be None")
    expected_workers = group.size()
    name = build_byteps_group_name(scope, logical_name)
    declare_and_cache_byteps_group(name, expected_workers)
    return name


def byteps_allreduce(
    tensor,
    group,
    scope: str,
    logical_name: str,
    average: bool = False,
    version: int = 0,
    priority: int = 0,
):
    name = _declare_group_and_get_name(group, scope, logical_name)
    return bps.push_pull(
        tensor,
        average=average,
        name=name,
        version=version,
        priority=priority,
    )


def byteps_allreduce_async_inplace(
    tensor,
    group,
    scope: str,
    logical_name: str,
    average: bool = False,
    version: int = 0,
    priority: int = 0,
):
    name = _declare_group_and_get_name(group, scope, logical_name)
    result = bps_ops.push_pull_async_inplace(
        tensor,
        average=average,
        name=name,
        version=version,
        priority=priority,
    )
    return result


def byteps_allreduce_inplace(
    tensor,
    group,
    scope: str,
    logical_name: str,
    average: bool = False,
    version: int = 0,
    priority: int = 0,
):
    name = _declare_group_and_get_name(group, scope, logical_name)
    # Use async inplace + synchronize for synchronous operation
    handle = bps_ops.push_pull_async_inplace(
        tensor,
        average=average,
        name=name,
        version=version,
        priority=priority,
    )
    bps_ops.synchronize(handle)
    return tensor


def pre_declare_all_byteps_groups(
    num_tp_layers: int,
    num_dp_buckets: int,
    use_dpu_tp: bool = True,
    use_dpu_dp: bool = True,
):
    """
    Pre-declare all BytePS groups in a consistent order across all workers.
    This ensures that all workers assign the same declared_key to each tensor.
    
    Args:
        num_tp_layers: Number of TP layers (for TP tensors)
        num_dp_buckets: Number of DP buckets (for DP tensors)
        use_dpu_tp: Whether TP BytePS is enabled
        use_dpu_dp: Whether DP BytePS is enabled
    """
    import torch.distributed as dist
    
    # Get parallel sizes
    tp_size = get_tensor_model_parallel_world_size()
    dp_size = get_data_parallel_world_size()
    pp_size = get_pipeline_model_parallel_world_size()
    
    # Get current rank's position in each parallel dimension
    tp_rank = get_tensor_model_parallel_rank()
    dp_rank = get_data_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()
    cp_rank = get_context_parallel_rank()
    
    # Pre-declare TP tensors (if enabled)
    # TP tensors are reduced across TP group, so they have the same name for all ranks in the same TP group
    # TP group is formed by ranks with the same dp_rank
    if use_dpu_tp:
        # Declare TP tensors in a deterministic order
        # Use layer index and tensor type to generate names
        tp_tensor_names = []
        for layer_idx in range(num_tp_layers):
            # Common TP tensor names from mappings.py
            for tensor_type in ['col_parallel_linear_fc1', 'row_parallel_linear_proj',
                               'col_parallel_linear_qkv', 'row_parallel_linear_fc2']:
                name = f"tp.dp{dp_rank}.pp{pp_rank}.cp{cp_rank}.tp_{tensor_type}_{layer_idx}"
                tp_tensor_names.append(name)
        
        # Declare all TP tensors
        for name in tp_tensor_names:
            if name not in _DECLARED_BPS_GROUPS:
                bps.declare(name, expected_workers=tp_size)
                _DECLARED_BPS_GROUPS[name] = tp_size
    
    # Pre-declare DP tensors (if enabled)
    # DP tensors are reduced across DP group, so they have the same name for all ranks in the same DP group
    # DP group is formed by ranks with the same tp_rank
    if use_dpu_dp:
        # Declare DP tensors in a deterministic order
        dp_tensor_names = []
        for bucket_idx in range(num_dp_buckets):
            name = f"dp.tp{tp_rank}.pp{pp_rank}.bucket_{bucket_idx}"
            dp_tensor_names.append(name)
        
        # Declare all DP tensors
        for name in dp_tensor_names:
            if name not in _DECLARED_BPS_GROUPS:
                bps.declare(name, expected_workers=dp_size)
                _DECLARED_BPS_GROUPS[name] = dp_size
    
    # Global barrier to ensure all workers have finished pre-declaration
    if dist.is_initialized():
        dist.barrier()
