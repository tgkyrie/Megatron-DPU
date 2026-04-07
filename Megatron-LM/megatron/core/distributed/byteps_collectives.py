from typing import Dict, Optional

import byteps.torch as bps
from byteps.torch import ops as bps_ops

from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_data_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)


_DECLARED_BPS_GROUPS: Dict[str, int] = {}


def build_byteps_group_name(scope: str, logical_name: str) -> str:
    if not logical_name:
        raise ValueError("logical_name must be a stable non-empty string")

    pp_rank = get_pipeline_model_parallel_rank()

    if scope == "dp":
        tp_rank = get_tensor_model_parallel_rank()
        # DDP gradients are reduced across the DP+CP group, so CP rank varies inside
        # the collective and must not be part of the BytePS name.
        return f"dp.tp{tp_rank}.pp{pp_rank}.{logical_name}"
    if scope == "tp":
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
    return bps_ops.push_pull_async_inplace(
        tensor,
        average=average,
        name=name,
        version=version,
        priority=priority,
    )


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
    return bps.push_pull_inplace(
        tensor,
        average=average,
        name=name,
        version=version,
        priority=priority,
    )
