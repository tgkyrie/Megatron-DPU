# BytePS 机间 RDMA/PS 优化变更文档

本文档用于确认下一步从 upstream `bccl-github` 对照移植的范围、正确性依据和验证方法。结论先行：本项目只需要强化机间 RDMA/PS 通信路径，不需要移植会把本机多 GPU 先聚合到 root 的 `BYTEPS_REDUCE_ROOTS` 路径。

## 1. 目标范围

目标是让 Megatron 的 DP/TP 子组 allreduce 继续通过 BytePS 的 PS 架构完成，同时把 worker 与 server 之间的数据通路优化成更完整的 GDR/RDMA 路径：

- 每个参与 GPU 按 DP/TP 子组语义独立参与 BytePS push/pull。
- 跨机规约仍发生在 PS/server 侧，server 收齐该 key 的 `expected_workers` 后规约并返回结果。
- 尽量减少 GPU->CPU->RDMA 或 RDMA->CPU->GPU 的 bounce copy。
- 保留当前本地已有的 DP/TP 子组声明、RDMA control/data 双通道、`BYTEPS_PUSH_THREAD`、默认 `raw` hash。
- 不把本机多 GPU 先通过 NCCL/local reduce 聚成一个 root 再发给 PS，因为这会改变我们想观察的网络通信形态。

非目标：

- 不移植 `BYTEPS_REDUCE_ROOTS` 作为默认路径。
- 不引入 alltoall/P2P、AllGatherV、TensorFlow/XLA、SyncBN、DCAdam、PipeSGD、server-side averaging 等与当前机间 push/pull 目标无直接关系的功能。
- 不替换当前 Megatron DP/TP 子组命名和 `expected_workers` 公开接口。

## 2. 当前代码理解

### 2.1 Megatron 调用边界

Megatron 已经在 DP 和 TP 两条路径接入 BytePS：

- `Megatron-LM/megatron/core/distributed/byteps_collectives.py`
  - `build_byteps_group_name()` 负责生成 DP/TP 子组稳定名称。
  - `declare_and_cache_byteps_group()` 调用 `bps.declare(name, expected_workers=group.size())`，并检查同名 key 的 group size 一致性。
  - `byteps_allreduce_async_inplace()` 与 `byteps_allreduce_inplace()` 封装 BytePS in-place push/pull。
- `Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py`
  - `use_dpu_reduce` 且未启用 distributed optimizer 时，DP bucket 走 BytePS。
- `Megatron-LM/megatron/core/tensor_parallel/mappings.py`
  - `_bps_reduce()` 用 BytePS 完成 TP group 内 allreduce，并在该调用点同步。

这里的关键正确性条件是：同一个 DP/TP 子组内所有 rank 必须使用相同 BytePS name，且 `expected_workers` 必须等于该子组大小；不同子组必须使用不同 name，避免 server 把不属于同一个 collective 的 worker 混在同一 key 上。

### 2.2 BytePS torch API 与 expected_workers

当前本地实现已有公开声明路径：

- `byteps/byteps/torch/ops.py` 的 `declare(name, expected_workers=None)`。
- `byteps/byteps/torch/ops.cc` 的 `RegisterTensorGroup()`。
- `byteps/byteps/common/global.cc` 的 `RegisterTensorExpectedWorkers()`。
- `byteps/byteps/server/server.cc` 的 `kGroupRegister` 处理和 `GetExpectedWorkers()`。

这套实现是 DP/TP 子组 allreduce 必需能力：server 不再默认等待全局 `DMLC_NUM_WORKER`，而是按每个 tensor key 的 `expected_workers` 收齐请求。上游 `bccl-github` 里的内部 expected participant 计数不能替代这套公开接口，只能在 GDR 路径中继续尊重它。

### 2.3 当前 GDR/RDMA 路径

当前本地 BytePS 已有：

- `DMLC_USE_GDR` 分支。
- `PushLoopGDR` / `PullLoopGDR`。
- RDMA control/data endpoint 分离。
- `BYTEPS_PUSH_THREAD` 多 push/pull 线程。
- 默认 `BYTEPS_KEY_HASH_FN=raw`。

但当前 GDR path 仍比较直接：`RunGDRPushLoopOnce()` 和 `RunGDRPullLoopOnce()` 在任务没有 `cpubuff` 时直接用 tensor data 指针构造 `ps::SArray<char>` 发起 `ZPush` / `ZPull`。它缺少 upstream `bccl-github` 后续引入的 GDR allreduce v1/v2 阶段管理、ready table、GDR buffer 管理、GPU reducer/CUDA kernel、CUDA event 查询和部分 server 线程安全修复。

### 2.4 bfloat16 支持缺口

Megatron/Qwen 训练常用 bf16，但当前 BytePS torch adapter 侧没有把 `torch::kBFloat16` 映射到 BytePS dtype，`byteps/byteps/common/common.h` 也没有 `BYTEPS_BFLOAT16`。因此 bf16 tensor 进入 BytePS push/pull 时会被判为不支持，或者被迫转成 fp16/fp32，都会偏离 Megatron bf16 训练语义。

## 3. 需要加入的变更表

| 优先级 | 功能 | 当前状态 | 需要做的变更 | 上游参考 | 正确性依据 |
|---|---|---|---|---|---|
| 直接需要 | GDR allreduce v1/v2 中的机间 push/pull 数据路径 | 本地有 `PushLoopGDR` / `PullLoopGDR`，但路径较粗糙 | 对照引入 GDR queue、phase/ready table、buffer 管理和 server GDR 状态；只保留 worker-server push/pull 相关部分 | `73fe922`, `9d32bf2`, `6ae9c5a`, `dde5350`, `32d3f84` | PS 架构不变：worker push 到 server，server 收齐 `expected_workers` 后规约，worker pull 结果；GDR 只改变数据放置和传输方式，不改变 collective 参与者集合 |
| 直接需要 | GDR buffer 管理 | 当前直接使用 tensor pointer 或 `cpubuff`，缺少完整生命周期控制 | 增加 GDR buffer 分配、复用、释放和按 key/phase 管理，避免同一 tensor 多阶段读写覆盖 | `32d3f84` | buffer 生命周期必须覆盖 async `ZPush`/`ZPull` 完成回调；回调前不能释放或复用正在被 RDMA 使用的内存 |
| 直接需要 | server 侧 GDR 状态与线程安全修复 | server 已有 per-key state 和 engine queue，但缺少 upstream 的部分 GDR ready 状态和线程安全修复 | 合入 server `KeyState`/ready table/handler lock 相关修复；保留本地 `kGroupRegister` 与 `expected_workers` | `3e19415`, `97907a7`, `97c7322`, `500ea58` | 多 worker、多 server、反复启动时，key state、shm/resource name 和请求队列不能互相污染；`expected_workers` 仍是收齐条件 |
| 直接需要 | GPU reducer 与 CUDA copy/reduce kernel | server 当前主要依赖 CPU reducer，GPU 数据路径能力不足 | 移植 GPU copy/sum/local op kernel；只取 push/pull/GDR 需要的 reducer，不引入 DCAdam/batched optimizer 逻辑 | `43f84f3`, `059ce44`, `2573371` | server 规约结果必须逐 element 等价于 CPU reducer；dtype、len、alignment 必须与 BytePS `DataType` 一致 |
| 直接需要 | bf16 dtype 与 CUDA reducer | BytePS common/torch adapter 不支持 `torch::kBFloat16` | 增加 `BYTEPS_BFLOAT16`、torch dtype 映射、server/GPU reducer bf16 sum/copy | `f213219`, `a02c62e` | bf16 allreduce 必须保持输入输出 dtype 为 bf16；结果允许按 bf16 累加语义与 NCCL/torch 分布式结果做容差对比 |
| 直接需要 | GDR 初始化和 GPU 选择 | 当前 `byteps_init()` 直接 `std::stoi(getenv("DMLC_USE_GDR"))`，env 缺失会出错；CUDA device 默认用 local rank | 改为安全 env parse；引入 visible device/default GPU selection/local init barrier 修复 | `2181807`, `1f11ac2`, `f7f33d6`, `8a037d0` | 不设置 GDR env 时应能走默认路径；设置 GDR 时每个 local rank 必须绑定正确 GPU，避免 RDMA 注册错误设备内存 |
| 需要 | fine-grained loop management | 当前 GDR 和非 GDR 初始化启动的 loop 粒度较粗 | 支持按 env/模式只启动需要的 GDR push/pull/server loop，避免无关 NCCL/local-copy loop 干扰 | `ab798d6` | 我们的目标是所有 GPU 直接走 RDMA；不应在 GDR 模式隐式启动 local reduce、H2D/D2H 或 NCCL 聚合路径 |
| 需要 | condition variable 替代 busy waiting | common loop 和 torch wait 中存在 sleep/poll | 对照 upstream 把核心队列等待改成条件变量，至少覆盖 GDR push/pull、handle wait 等热点 | `c55e2f4` | 不改变完成顺序和回调语义，只降低空转 CPU 占用；poll API 对外语义保持不变 |
| 需要 | CUDA event query in NCCL/GDR loop | 当前有 CUDA ready event guard，但 GDR path 仍需核对同步点 | 对照引入非阻塞 CUDA event query，避免过度 synchronize | `4943c49` | RDMA 发起前必须保证 GPU 写完成；但不应全局同步 CUDA stream，避免放大尾延迟 |
| 保留并适配 | expected_workers 子组声明 | 本地已有且符合 Megatron DP/TP 需求 | 保留 Python API、C++ register、server `kGroupRegister`；确保新 GDR key/phase 仍按该值收齐 | 本地实现为主，上游只作参考 | 这是 DP/TP 子组 allreduce 的正确性基础，不能被上游内部 participant 计数替换 |
| 保留并适配 | RDMA control/data 双通道 | 本地已有 | 合并 upstream 时不得退回单 endpoint；GDR push/pull 数据必须走 data-plane endpoint | 本地实现为主 | control 与 data 分离是当前机间带宽优化基础，能减少控制消息和大 tensor 数据互相阻塞 |
| 保留并适配 | `BYTEPS_PUSH_THREAD` 与默认 raw hash | 本地已有 | 新 GDR queue/phase 需要继续兼容多 push/pull 线程和 raw key 分布 | 本地实现为主 | 多线程和 raw hash 是当前跨 server 分片吞吐优化点，不能因移植 GDR 路径丢失 |

## 4. 不需要加入或默认关闭的功能

| 功能 | 处理 | 原因 |
|---|---|---|
| `BYTEPS_REDUCE_ROOTS` 与 local reduce root 选择 | 不移植为默认路径；如代码冲突，保持关闭 | 它的作用是本机多 GPU 先选 root/local reduce，再由 root 发给 PS；这会减少每机发往 PS 的参与 GPU 数，不符合“所有 GPU 都强制走 RDMA、主要观察网络通信”的目标 |
| GDR allgather / alltoall / P2P | 不移植 | 当前 Megatron 接入点是 DP/TP allreduce 语义，PS push/pull 已覆盖需求 |
| server-side averaging | 暂不移植 | 当前平均语义由 torch API 的 `average` 与现有流程控制，先保持 SUM/AVG 行为不变，避免和 Megatron gradient scaling 叠加 |
| DCAdam / batched optimizer ops | 不移植 | 不是通信路径优化，会扩大代码面和验证范围 |
| TensorFlow/XLA/SyncBN/PipeSGD | 不移植 | 与当前 PyTorch Megatron DP/TP 机间通信目标无关 |

## 5. BYTEPS_REDUCE_ROOTS 的明确判断

`BYTEPS_REDUCE_ROOTS` 不是跨机 PS 规约能力，它解决的是节点内多 GPU 到 PS 前的代表 root 选择问题。典型流程是：

1. 本机多个 GPU 先通过 local reduce/NCCL 或 copy 路径汇聚。
2. 只有选出的 root 代表本机向 PS 发起 push。
3. PS 在跨机维度规约这些 root 的数据。
4. root 再把结果在本机广播/拷贝给其他 GPU。

如果目标是减少跨机请求数、降低每机网络压力，这个功能有价值；但如果目标是让所有 GPU 都直接走 RDMA、观察和优化 worker-server 网络通信，它会隐藏一部分 GPU 的网络流量，并引入本机聚合瓶颈。因此本项目当前不需要把它作为核心优化点。

## 6. 正确性约束

后续代码变更必须同时满足以下条件：

- DP 子组 name 必须由 TP rank/PP rank/逻辑 bucket 稳定决定；TP 子组 name 必须由 DP rank/PP rank/CP rank/逻辑 op 稳定决定。
- 同一 BytePS key 的所有参与 rank 必须声明相同 `expected_workers`；不同子组不得共用 key。
- server 处理 push 时必须按 `expected_workers` 收齐，而不是按全局 worker 数收齐。
- GDR 模式不能绕过 `kGroupRegister`；如果没有声明，才允许回退到 `ps::NumWorkers()`。
- GDR push 发起前必须保证 GPU tensor 数据已经 ready；GDR pull 完成后才能让上层 handle 标记完成。
- RDMA data message 必须继续使用 data-plane endpoint；控制消息继续使用 control endpoint。
- bf16 tensor 进入 BytePS 后，dtype 不能被错误映射为 fp16/fp32；输出 dtype 必须仍是 bf16。
- `average=True` 和 `average=False` 的语义必须与当前 BytePS torch API 保持一致，不能额外叠加 Megatron 的 gradient scaling。
- 失败路径要明确：inconsistent `expected_workers`、不支持 dtype、GDR device 绑定失败都应直接报错，而不是静默 fallback 到错误结果。

## 7. 验证计划

### 7.1 静态检查

- 检查 `byteps/byteps/common/common.h` 是否新增 `BYTEPS_BFLOAT16`，并在 reducer、adapter、command type 中完整覆盖。
- 检查 `byteps/byteps/torch/adapter.cc` 是否映射 `torch::kBFloat16`。
- 检查 GDR 初始化不再对缺失 `DMLC_USE_GDR` 调用 `std::stoi(nullptr)`。
- 检查 GDR loop 没有隐式依赖 `BYTEPS_REDUCE_ROOTS`。
- 检查 `expected_workers` 的 register path 在 GDR key/phase 下仍被使用。

### 7.2 功能验证

- 单机单进程：`bps.init()`、`bps.declare()`、fp16/fp32/bf16 push_pull smoke test。
- 多进程子组：构造 DP/TP 子组，验证同名 key 只等待子组大小，非全局 worker 数。
- 多机 RDMA：保持 `DMLC_USE_GDR=0`，对比 BytePS push/pull 结果与 `torch.distributed.all_reduce`。
- bf16：用 bf16 tensor 跑 DP bucket 和 TP reduce，验证 dtype 不变且结果在容差内。
- 异步：`push_pull_async_inplace` 的 `poll()`/`synchronize()` 不提前完成，完成后数据正确。

### 7.3 性能验证

- 使用 `byteps/example/pytorch/pushpull_bench.py` 或等价脚本测 fp16/bf16 大 tensor 带宽。
- 监控 RDMA data-plane 流量、CPU memcpy、CPU 利用率、server engine queue 延迟。
- 对比三组配置：
  - 当前本地 RDMA 双通道 + 非完整 GDR path。
  - 完整 GDR path。
  - GDR 关闭 fallback。
- 确认完整 GDR path 提升来自机间数据通路，而不是本机 `BYTEPS_REDUCE_ROOTS` 聚合减少流量。

## 8. 移植顺序建议

1. 先修初始化安全性、GPU visible device/local init barrier、server 资源名和线程安全修复。
2. 再合入 GDR queue/phase/ready table/buffer 管理，保持现有 push/pull API 不变。
3. 接入 GPU reducer/CUDA kernel，只覆盖 push/pull server 规约和 GDR copy/reduce 必需路径。
4. 加 bf16 dtype 与 bf16 CUDA reducer。
5. 最后做 fine-grained loop management、condition variable 和 CUDA event query 优化。

这个顺序的原则是先保证不会启动失败或收错 worker，再替换数据路径，最后降低 CPU 空转和同步开销。每一步都可以用现有 Megatron DP/TP 子组接口做回归验证。

## 9. 最新复核结论

复核来源是 GitHub `bytedance/byteps` 的 `bccl-github` 分支，远端 HEAD 为 `a02c62e5b471a1c7d3c2fa1ae600bf8574f1b6a5`。该分支的 GDR allreduce 不是当前 `DMLC_USE_GDR`/`BYTEPS_GDR_DIRECT` 旧路径的局部修补，而是一套 joint-mode 设计：

- 使用 `BYTEPS_USE_GDR_ALLREDUCE` 开关，默认关闭。
- 要求 `DMLC_ROLE=joint`，worker 进程内同时启动本地 server 线程。
- launcher 会把 `DMLC_NUM_WORKER` 和 `DMLC_NUM_SERVER` 都设为 `BYTEPS_NUM_NODES * BYTEPS_LOCAL_SIZE`。
- common/server 使用新的 `GDR_V1_PUSH_PULL`、`GDR_V2_PUSH_PULL`、`GDR_WAIT_PUSH_PULL` queue，以及 server 侧 GDR ready table、GPU buffer registration、CUDA reducer。
- ps-lite 的 UCX 路径需要 `USE_UCX=1`，GPU direct 场景还需要 `USE_CUDA=1 CUDA_HOME=...`。

因此，不能把本地 master 的 `DMLC_USE_GDR` 分支简单替换成 `bccl-github` 的几个函数。临时 worktree 做过全量迁移试验：按本仓库路径映射套 upstream `common/server/setup/launcher` 补丁不能直接应用；机械复制核心文件后，server-only 最小构建也会因为 upstream server 编译路径新增 `global.cc`、`numa.h`、joint-mode/server API 等依赖而失败。这个结果说明完整替换必须作为一次 BytePS core 迁移来做，并且要重新合并本地 DP/TP `expected_workers` 公开接口。

本分支当前只保留已验证的安全变更：

- 普通 `byteps/sh/worker.sh` 默认 `DMLC_USE_GDR=0`，避免未显式开启 GDR 时误入旧 legacy GDR。
- `byteps/sh/worker_direct_gpu.sh` 默认 `DMLC_USE_GDR=0`，用于“每 GPU 一个 ps-lite worker”的 direct no-GDR 实验；只有显式设置 `DMLC_USE_GDR=1` 时才进入旧 direct-GDR 路径。
- `byteps/setup.py` 支持在显式 `BYTEPS_WITH_UCX=1 BYTEPS_WITH_GPU=1` 时给 ps-lite 传 `USE_UCX=1 USE_CUDA=1 CUDA_HOME=...`，默认不改变现有构建行为。
- `byteps/setup.py` 和 `Dockerfile.byteps` 增加可选 TensorPipe Van 构建入口：`BYTEPS_WITH_TP=1` / `BYTEPS_TP_HOME=...`；运行时通过 `BYTEPS_USE_TP=1` 或 `DMLC_PS_VAN_TYPE=tp` 选择。
- legacy `DMLC_USE_GDR` 代码仍在源码中，但不应视为 `bccl-github` GDR 替代实现。

已验证：

- `bash -n byteps/sh/worker.sh`
- `python3 -m py_compile byteps/setup.py`
- `BYTEPS_SERVER_ONLY=1 BYTEPS_FORCE_NO_CUDA=1 BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 BYTEPS_WITHOUT_MXNET=1 BYTEPS_WITHOUT_PRESETUP=1 python3 setup.py build_ext --force`

当前环境没有 RDMA header、UCX header 和 PyTorch，因此不能在本机验证真实 RDMA/UCX/GDR 数据面。

## 10. 当前可运行配置

非 GDR RDMA：

```bash
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_ENABLE_UCX=0
export DMLC_USE_GDR=0
export DMLC_PS_ROOT_URI=<scheduler_ip>
export DMLC_PS_ROOT_PORT=<scheduler_port>
export DMLC_NUM_WORKER=<num_worker_nodes>
export DMLC_NUM_SERVER=<num_servers>
export DMLC_INTERFACE=<rdma_control_interface>
```

UCX：

```bash
# build
export BYTEPS_WITH_UCX=1
export BYTEPS_WITH_GPU=1
export BYTEPS_CUDA_HOME=/usr/local/cuda

# runtime
export DMLC_ENABLE_UCX=1
export DMLC_USE_GDR=0
export DMLC_PS_ROOT_URI=<scheduler_ip>
export DMLC_PS_ROOT_PORT=<scheduler_port>
export DMLC_NUM_WORKER=<num_worker_nodes>
export DMLC_NUM_SERVER=<num_servers>
export DMLC_INTERFACE=<control_interface>
```

环境变量含义：

- `DMLC_ENABLE_RDMA=ibverbs`：使用 ps-lite RDMAVan。
- `DMLC_ENABLE_UCX=1`：强制 ps-lite 使用 UCXVan；优先级高于 `DMLC_ENABLE_RDMA`。
- `BYTEPS_WITH_UCX=1`：构建时编译 UCX 支持。
- `BYTEPS_WITH_GPU=1`：构建 UCX 时给 ps-lite 打开 CUDA-aware 路径。
- `BYTEPS_CUDA_HOME`：指定 CUDA 目录，用于 ps-lite `USE_CUDA=1`。
- `DMLC_USE_GDR=0`：保持 legacy GDR 关闭；当前分支不把它作为 bccl GDR 使用。
- `DMLC_NUM_WORKER`/`DMLC_NUM_SERVER`：ps-lite worker/server 数量。
- `DMLC_PS_ROOT_URI`/`DMLC_PS_ROOT_PORT`：scheduler 地址。
- `DMLC_INTERFACE`：控制面使用的网卡接口。
- `BYTEPS_UUID`/`BYTEPS_JOB_ID`：共享内存和本机 socket 名称前缀；direct no-GDR 多进程同机运行时必须让每个进程不同，`worker_direct_gpu.sh` 会自动生成。
- `BYTEPS_VISIBLE_DEVICE`：BytePS 内部 `cudaSetDevice()` 使用的可见 GPU ordinal；direct 脚本把单卡 mask 后固定为 `0`。

TensorPipe：

```bash
# build
export BYTEPS_WITH_TP=1
export BYTEPS_TP_HOME=/usr/local/tensorpipe
export BYTEPS_TP_LIBS="tensorpipe tensorpipe_uv"

# runtime
export BYTEPS_USE_TP=1
export DMLC_PS_VAN_TYPE=tp
unset DMLC_ENABLE_RDMA
unset DMLC_ENABLE_UCX
export DMLC_USE_GDR=0
```

`Dockerfile.byteps` 默认 `ARG ENABLE_TP=1`，会从 PyTorch 镜像中的 torch 安装目录复制 TensorPipe 头文件和 `libtensorpipe*` 到 `/usr/local/tensorpipe`。如需回到非 TP 构建，可使用 `--build-arg ENABLE_TP=0`。

运行脚本中，`scheduler.sh`、`server.sh`、`worker.sh`、direct 三个脚本、`pushpull.sh`、Megatron Qwen/GPT 示例、以及 ps-lite `tests/test.sh` 都会识别 `BYTEPS_USE_TP=1` 或 `DMLC_PS_VAN_TYPE=tp`，并在这种情况下清掉 `DMLC_ENABLE_RDMA`/`DMLC_ENABLE_UCX`，避免 ps-lite 的 `Postoffice` 优先选择 RDMA/UCX 而不是 TP。`tests/test_stress.sh` 是 UCX/RDMA stress 专用脚本，仍保持 UCX 默认。

direct no-GDR 每 GPU 一个 ps-lite worker：

```bash
export NUM_NODES=<num_nodes>
export GPU_WORKERS_PER_NODE=<gpus_per_node>
export NUM_SERVERS=<num_servers>
export DMLC_USE_GDR=0

bash byteps/sh/scheduler_direct.sh
bash byteps/sh/server_direct.sh
NODE_RANK=<node_rank> GPU_ID=<gpu_id> bash byteps/sh/worker_direct_gpu.sh
```
