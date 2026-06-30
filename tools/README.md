# Experiment Tools

Current Megatron entrypoint:

- `run_megatron_ucx_semantic.py`: current Megatron NCCL / host-server BytePS /
  DPU-server BytePS runner. This is the script referenced by `AGENTS.md`.

Policy for current runners:

- Let `Megatron-LM/examples/qwen/*.sh` provide worker-side defaults whenever
  possible.
- Runner code should pass topology/runtime values such as ports, node rank,
  root URI, worker/server counts, and per-host node IP.
- Do not duplicate worker-side defaults for `USE_DPU*`, `DMLC_ENABLE_UCX`,
  `DMLC_ENABLE_RDMA`, `DMLC_USE_GDR`, `BYTEPS_*`, `UCX_*`, `PSLITE_*`,
  `CUDA_VISIBLE_DEVICES`, `TRAIN_ITERS`, `EVAL_INTERVAL`, `EVAL_ITERS`, or
  `SEED` unless that variable is the explicit experimental knob.
- In Megatron runners, scheduler/server roles do not run the qwen training
  scripts, so their BytePS/UCX environment may be set by the runner, but it
  should match the worker script defaults or the explicit sweep variable.
- In BytePS-only runners, keep UCX/BytePS/GPU/NIC defaults in `byteps/sh/*.sh`
  and let the runner pass only topology values plus benchmark arguments.

Other experiment runners:

- `run_paper_revision_megatron.sh`: scripted paper experiment matrix built on
  Megatron training scripts.
- `run_paper_extra_benchmarks.sh`: smaller Megatron ablation runner.
- `run_tp_only_ucx_sweep.sh`: TP-only UCX parameter sweep.
- `run_byteps_payload_sweep.sh`: BytePS payload-size sweep.
- `run_byteps_payload_scaling_matrix.sh`: optional 2/4/8 worker payload-size
  scaling matrix for BytePS-style systems.
- `run_vgg16_byteps_benchmark.sh`: VGG16 BytePS/MegaScale-PS worker/server
  path benchmark.
- `run_vgg16_ddp_benchmark.sh`: VGG16 DDP baseline.
- `run_vgg16_scaling_matrix.sh`: optional 2/4/8 worker VGG16 scaling matrix.

Shell runners expect persistent containers to already be running. By default
they use `BYTEPS_CONTAINER=byteps-latest` and
`WORKER_CONTAINER=megatron-dpu-latest`; override these variables if a test uses
different container names.

When changing shell runners, compare every exported variable with the defaults
in the invoked training script. Keep exports only when they are topology/runtime
inputs or explicit experimental knobs.

The BytePS push-pull and VGG16 microbenchmarks do not invoke
`Megatron-LM/examples/qwen/*.sh`. For VGG16, keep runtime defaults in the real
BytePS role scripts under `byteps/sh/` and let the runners pass only topology
values such as worker/server counts, rank, and port. For direct push-pull
payload sweeps, explicit benchmark-size arguments are expected because there is
no Megatron training wrapper involved.

Paper experiment coverage:

For a directory-level audit of the existing `docs/benchmark_runs/` outputs, see
`tools/RESULT_PROVENANCE.md`.

| Script | Paper role | Keep status |
| --- | --- | --- |
| `run_megatron_ucx_semantic.py` | Current operational runner for NCCL / Host-PS / DPU-PS Megatron checks, including eval-loss parsing. | Keep as the current runbook entrypoint. |
| `run_paper_revision_megatron.sh` | Covers the paper Megatron matrix: DP / TP / TP+DP, NCCL vs Host-PS, server-count, partition, overlap, and long loss runs. | Keep as the main paper batch runner. |
| `run_paper_extra_benchmarks.sh` | Older smaller TP+DP ablation runner for server-count, partition, and fused/overlap checks. | Redundant with `run_paper_revision_megatron.sh`; keep only as historical evidence unless its exact old run needs to be reproduced. |
| `run_tp_only_ucx_sweep.sh` | TP-only Qwen-3B UCX sweep used for the tuned TP-only Host-PS row. | Keep while the paper uses the tuned TP-only row. |
| `run_byteps_payload_sweep.sh` | Generates the 256KB to 16MB push-pull payload sweep used by the bandwidth figure. | Keep. |
| `run_byteps_payload_scaling_matrix.sh` | Optional scaling matrix for BytePS-style payload microbenchmarks across 2/4/8 workers and servers. | Keep as supplemental experiment tooling. |
| `run_vgg16_byteps_benchmark.sh` | Generates the BytePS/MegaScale-PS side of the VGG16 images/s figure through `byteps/sh/scheduler.sh`, `server.sh`, and `worker.sh`. | Keep. |
| `run_vgg16_ddp_benchmark.sh` | Generates the DDP baseline for the VGG16 images/s figure through `byteps/sh/worker_ddp.sh`. | Keep. |
| `run_vgg16_scaling_matrix.sh` | Optional scaling matrix for VGG16 DDP and BytePS-style systems across 2/4/8 workers. | Keep as supplemental experiment tooling. |

The VGG16 runners intentionally do not duplicate GPU/NIC/NCCL/UCX defaults.
Those defaults live in `byteps/sh/worker.sh`, `server.sh`, `scheduler.sh`, and
`worker_ddp.sh`; the runners only provide the multi-node topology.
