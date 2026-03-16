# Megatron + BytePS 主要改动整理（tgkyrie 相关）

本文件整理 tgkyrie 的关键提交、Megatron 后端通信替换为 BytePS 的实现点，以及 BytePS push/pull 相关优化，并给出核心代码位置与行号。

## 一、关键提交时间线（tgkyrie）

| 日期 | 提交 | 主题 | 说明 |
|---|---|---|---|
| 2025-11-24 | `a39bfe7` | [Init] | 仓库初始化结构（Megatron-LM/BytePS 引入）。 |
| 2025-11-24 | `e76c4d3` | [INIT] | 大量导入 Megatron-LM 与 BytePS 代码。 |
| 2025-11-25 | `d72d072` | [INIT] | 首次把 Megatron 的 DP 梯度同步改成 BytePS。 |
| 2025-11-26 | `dd15a04` | [OPT] Add Optional Args for dpu reduce | 新增 `--use-dpu-reduce`，在初始化流程中按需 `bps.init()`。 |
| 2025-11-29 | `a90a8ca` | [OPT] allreduce in-place instead of copy | DP 梯度同步由 copy 改为 in-place（节省拷贝）。 |
| 2026-01-27 | `e23c5f4` | [OPT] Optimize BytePS Bandwidth | RDMA 双通道 + push/pull 多线程等带宽优化。 |
| 2026-01-27 | `67ff54e` | [FIX] fix a MegatronLM bug | 非主线功能修复。 |

## 二、工作目标与落地点

### 1) 将 Megatron 后端通信改为 BytePS

**配置与入口**
- 新增配置开关：`Megatron-LM/megatron/core/distributed/distributed_data_parallel_config.py:8-12`
- 新增 CLI 参数：`Megatron-LM/megatron/training/arguments.py:2565-2568`
- 初始化阶段按需 `bps.init()`：`Megatron-LM/megatron/training/initialize.py:107-110`

**DP 梯度同步替换（核心路径）**
- `Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:381-417`
  - 在不使用分布式优化器时，改用 BytePS `push_pull` 完成 DP 梯度同步。

**in-place 版本（提交 a90a8ca）**
- 同文件中 `a90a8ca` 将：
  - `push_pull + copy` 替换为 `bps.push_pull_inplace(...)`
  - 当前 HEAD 仍保留 `push_pull + copy`（如需恢复 in-place，可按该提交逻辑改回）。

---

### 2) 优化 BytePS push/pull 逻辑

#### 2.1 RDMA 双通道（控制/数据分离）

**新增 data-plane endpoint**
- 连接与 endpoint 管理：`byteps/3rdparty/ps-lite/src/rdma_van.h:177-260`
- 发送逻辑（push/pull 走 data endpoint）：`byteps/3rdparty/ps-lite/src/rdma_van.h:384-461`
- data_endpoints 容器：`byteps/3rdparty/ps-lite/src/rdma_van.h:928-934`

**协议字段与结构支持**
- `RequestContext.isDataPlane`：`byteps/3rdparty/ps-lite/src/rdma_utils.h:171-175`
- `Endpoint.isDataPlane`：`byteps/3rdparty/ps-lite/src/rdma_transport.h:28-68`

**对比原始 BytePS**
- 原始 BytePS 只有单一 RDMA endpoint，控制与数据复用。

#### 2.2 Push/Pull 多线程

- 新增 `BYTEPS_PUSH_THREAD`：`byteps/byteps/common/global.cc:120-162`
- 根据线程数启动多个 PushLoop/PullLoop：`byteps/byteps/common/operations.cc:47-71`

**对比原始 BytePS**
- 原始 BytePS 默认只启动单线程 push/pull。

#### 2.3 Key Hash 策略调整（负载分布）

- 默认 hash 改为 `raw`：`byteps/byteps/common/global.cc:154-163`
- `raw` 分配逻辑：`byteps/byteps/common/global.cc:630-676`

**对比原始 BytePS**
- 原始默认 `djb2`。

#### 2.4 可观测性与接口补充

- PUSH 队列 debug：`byteps/byteps/common/scheduled_queue.cc:101-105`
- 追加 `get_pushpull_speed` 接口：`byteps/byteps/torch/ops.py:37-46`
- 新增 push/pull benchmark：`byteps/example/pytorch/pushpull_bench.py`

---

## 三、对比“原有 BytePS”的核心差异总结

1. **RDMA 双通道**：新增 data-plane endpoint，push/pull 数据走 data endpoint（提升带宽）。
2. **多线程 Push/Pull**：通过 `BYTEPS_PUSH_THREAD` 并发 push/pull。
3. **哈希策略变更**：默认 `raw`，更直接按 key 分配 server。
4. **接口与观测增强**：新增 `get_pushpull_speed` 与 push/pull benchmark。

---

## 四、与“我们的工作目标”的对应关系

- **目标 1：Megatron 后端通信换为 BytePS**
  - 配置开关 + CLI 参数 + 初始化 `bps.init()` + DP 梯度同步替换（见第 2 节）。

- **目标 2：优化 BytePS push/pull 逻辑**
  - RDMA 双通道 + Push/Pull 多线程 + hash 策略调整（见第 2 节）。

---

## 五、补充说明（建议）

- 如果要对外汇报，可将“替换点”和“优化点”拆成两张表，附上 commit 与行号。
- 如果需要恢复 in-place allreduce（`a90a8ca`），可把 `param_and_grad_buffer.py` 的 `push_pull + copy` 改回 `push_pull_inplace`。

---

## 六、整体思路与收益

### 6.1 设计思路

1. **Megatron 的 DP 同步替换为 BytePS**  
   - 目的：在多机、多卡场景下，用 BytePS 的 PS/调度机制替代传统 NCCL allreduce，提升跨机带宽利用与可调度性。  
   - 做法：在 DP 梯度同步入口处切换到 `bps.push_pull`，通过 `--use-dpu-reduce` 控制开关，统一在初始化阶段 `bps.init()`，避免每个训练脚本各自初始化。  

2. **BytePS push/pull 侧的带宽优化**  
   - 目的：提升 RDMA 通信吞吐，降低跨机通信瓶颈。  
   - 做法：  
     - RDMA 双通道（控制/数据分离）以降低控制流与数据流互相干扰；  
     - push/pull 多线程提升并发度；  
     - hash 策略改为 `raw`，降低哈希开销并稳定分片分配；  
     - 增加可观测性接口，便于评估优化效果。  

### 6.2 收益（预期/可量化方向）

- **跨机带宽利用率提升**：  
  RDMA 数据面单独通道 + 多线程 push/pull，可显著提高有效吞吐。  

- **通信可控性增强**：  
  通过 `--use-dpu-reduce` 明确切换通信后端，便于 A/B 测试与回退。  

- **内存与拷贝开销下降**：  
  in-place 版本（`a90a8ca`）可减少额外 copy，降低显存与内存带宽压力。  

- **性能调参空间扩大**：  
  通过 `BYTEPS_PUSH_THREAD`、hash 策略等参数可针对不同网络/拓扑调优。  

> 注：实际收益需结合具体网络（RDMA/HCA/GID）、模型规模和 batch 规模进行基准测试。建议使用 `pushpull_bench.py` 与训练吞吐日志对比验证。  

---

## 七、修改前/修改后关键代码（含流程说明）

> 说明：以下片段为关键逻辑对比，已按“修改前 / 修改后”截取核心代码段。

### 7.1 d72d072：Megatron DP 同步由 NCCL allreduce -> BytePS push/pull

**代码作用**  
在 DP 梯度同步阶段，用 BytePS `push_pull` 替代 NCCL `all_reduce`。  

**为什么这样做**  
- BytePS 更适合跨机调度与带宽优化；  
- 可以通过开关做 A/B 测试与回退。  

**流程（摘要）**  
1. 进入梯度归约阶段；  
2. 若满足 `use_dpu_reduce`，走 BytePS 分支；  
3. 按 bucket 逐个 `push_pull`，回写梯度；  
4. 直接 `return`，跳过 NCCL allreduce。  

**修改前（NCCL allreduce）**
```python
# Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py
if self.ddp_config.use_distributed_optimizer:
    communication_group = self.intra_distributed_optimizer_instance_group
else:
    communication_group = self.data_parallel_group

with stream_context, _coalescing_manager(communication_group, async_ops=async_op) as cm:
    for idx, bucket in enumerate(self.buckets):
        ...
        torch.distributed.all_reduce(
            bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op
        )
```

**修改后（BytePS push/pull）**
```python
# Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py
# 注：早期字段名为 use_byteps_for_grad_sync，后续改为 use_dpu_reduce
if (not self.ddp_config.use_distributed_optimizer
        and getattr(self.ddp_config, "use_dpu_reduce", False)):
    byteps_average = self.ddp_config.average_in_collective
    for idx, bucket in enumerate(self.buckets):
        name = f"dp_bucket_{idx}"
        reduced = bps.push_pull(
            bucket.grad_data,
            average=byteps_average,
            name=name,
            version=0,
            priority=0
        )
        bucket.grad_data.copy_(reduced)
    self.grad_reduce_handle = None
    return
```

### 7.2 dd15a04：新增 `--use-dpu-reduce` 并迁移 `bps.init()` 位置

**代码作用**  
提供统一开关与初始化位置，控制是否启用 BytePS。  

**为什么这样做**  
- 避免训练脚本里重复初始化；  
- 统一入口更易维护与排查问题。  

**流程（摘要）**  
1. CLI 增加 `--use-dpu-reduce`；  
2. 初始化阶段按开关 `bps.init()`；  
3. 训练脚本移除硬编码 `bps.init()`。  

**修改前（arguments 无开关）**
```python
# Megatron-LM/megatron/training/arguments.py
group.add_argument('--overlap-grad-reduce', action='store_true',
                   default=False, help='If set, overlap DDP grad reduce.')
```

**修改后（新增开关）**
```python
# Megatron-LM/megatron/training/arguments.py
group.add_argument('--overlap-grad-reduce', action='store_true',
                   default=False, help='If set, overlap DDP grad reduce.')
group.add_argument('--use-dpu-reduce', action='store_true',
                   default=False, help='If set, use DPU for grad reduce.')
```

**修改前（init 阶段不触发 bps.init）**
```python
# Megatron-LM/megatron/training/initialize.py
args = get_args()
initialize_rerun_state_machine(...)
```

**修改后（按参数初始化 BytePS）**
```python
# Megatron-LM/megatron/training/initialize.py
args = get_args()
if args.use_dpu_reduce:
    bps.init()
initialize_rerun_state_machine(...)
```

**修改前（pretrain_gpt.py 内部直接 bps.init）**
```python
if __name__ == "__main__":
    bps.init()
    train_valid_test_datasets_provider.is_distributed = True
```

**修改后（移除 bps.init，统一到 initialize.py）**
```python
if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
```

### 7.3 a90a8ca：BytePS 梯度同步 in-place

**代码作用**  
把 `push_pull + copy` 改为 `push_pull_inplace`，减少中间拷贝。  

**为什么这样做**  
- 降低显存占用与内存带宽压力；  
- 减少额外 tensor 分配开销。  

**修改前（push_pull + copy）**
```python
reduced = bps.push_pull(
    bucket.grad_data,
    average=byteps_average,
    name=name,
    version=0,
    priority=0
)
bucket.grad_data.copy_(reduced)
```

**修改后（in-place）**
```python
bps.push_pull_inplace(bucket.grad_data, average=byteps_average, name=name)
```

### 7.4 e23c5f4：RDMA 双通道（data-plane）

**代码作用**  
拆分 RDMA 控制面与数据面：控制消息与数据传输走不同 endpoint。  

**为什么这样做**  
- 避免控制流和大数据包相互阻塞；  
- 改善 RDMA 实际吞吐与延迟稳定性。  

**流程（摘要）**  
1. `Connect()` 同时建立 control/data 两个连接；  
2. push/pull 数据走 data endpoint；  
3. 控制类消息仍走 control endpoint。  

**修改前（单 endpoint）**
```cpp
// byteps/3rdparty/ps-lite/src/rdma_van.h
void Connect(const Node &node) override { ... endpoints_[node.id] ... }
...
trans->SendPushRequest(msg, msg_buf, addr_tuple);
...
trans->SendPullResponse(msg, msg_buf, addr_tuple, temp_mr->second->lkey);
```

**修改后（双 endpoint，push/pull 走 data endpoint）**
```cpp
// byteps/3rdparty/ps-lite/src/rdma_van.h
void Connect2Node(const Node &node, bool dataPlane=false) { ... data_endpoints_ ... }
void Connect(const Node &node) override {
  Connect2Node(node,false);
  Connect2Node(node,true);
}
...
dataTrans->SendPushRequest(msg, msg_buf, addr_tuple);
...
dataTrans->SendPullResponse(msg, msg_buf, addr_tuple, temp_mr->second->lkey);
```

### 7.5 e23c5f4：Push/Pull 多线程 + hash 默认策略

**代码作用**  
引入 push/pull 多线程并调整默认 hash 策略。  

**为什么这样做**  
- 多线程提高并发度与带宽利用率；  
- `raw` 策略减少 hash 开销并提升可预测性。  

**流程（摘要）**  
1. 读取 `BYTEPS_PUSH_THREAD`；  
2. 启动多个 PushLoop/PullLoop；  
3. 默认 hash 改为 `raw`。  

**修改前（单线程 + djb2）**
```cpp
// byteps/byteps/common/global.cc
_start_step = getenv("BYTEPS_TRACE_START_STEP") ? ... : _start_step;
_end_step = getenv("BYTEPS_TRACE_END_STEP") ? ... : _end_step;
...
_is_distributed_job = (_num_worker > 1) ? true : _is_distributed_job;
_hash_knob = std::string(getenv("BYTEPS_KEY_HASH_FN") ? ... : "djb2");
```

**修改后（多线程 + raw）**
```cpp
// byteps/byteps/common/global.cc
_start_step = getenv("BYTEPS_TRACE_START_STEP") ? ... : _start_step;
_push_thread = getenv("BYTEPS_PUSH_THREAD") ? atoi(...) : 1;
_end_step = getenv("BYTEPS_TRACE_END_STEP") ? ... : _end_step;
...
_is_distributed_job = (_num_worker > 0) ? true : _is_distributed_job;
_hash_knob = std::string(getenv("BYTEPS_KEY_HASH_FN") ? ... : "raw");
```

**修改前（单线程 Push/Pull）**
```cpp
// byteps/byteps/common/operations.cc
if (BytePSGlobal::IsRootDevice()) {
  func.push_back(PullLoop);
  func.push_back(DecompressLoop);
}
...
if (BytePSGlobal::IsRootDevice()) {
  func.push_back(PushLoop);
  func.push_back(CompressLoop);
  func.push_back(RootCopyHost2DeviceLoop);
}
```

**修改后（多线程 Push/Pull）**
```cpp
// byteps/byteps/common/operations.cc
if (BytePSGlobal::IsRootDevice()) {
  for (int i = 0; i < BytePSGlobal::GetPushThread(); i++) {
    func.push_back(PullLoop);
  }
  func.push_back(DecompressLoop);
}
...
if (BytePSGlobal::IsRootDevice()) {
  for (int i = 0; i < BytePSGlobal::GetPushThread(); i++) {
    func.push_back(PushLoop);
  }
  func.push_back(CompressLoop);
  func.push_back(RootCopyHost2DeviceLoop);
}
```

---

## 八、分支差异（byteps 分支 vs 其他分支）

> 说明：本仓库当前没有名为 `byteps` 的分支；常用的是 `wpb_byteps` 与 `master`。若你指的是其他远端分支（如 `origin/hc_bps`），请明确分支名。

### 7.1 `master` vs `wpb_byteps`

- 当前 **`master` 与 `wpb_byteps` 指向同一提交**（无差异）。

### 7.2 `master` vs `origin/hc_bps`（示例差异概览）

差异较大的文件（节选，`git diff --stat master..origin/hc_bps`）：  
```
Dockerfile.arm64serverslim                         |  234 -
byteps/3rdparty/ps-lite/src/rdma_van.h             |  174 +-
byteps/byteps/common/global.cc                     |   58 +-
byteps/byteps/common/operations.cc                 |    8 +-
byteps/example/train_qwen_3b.sh                    |  236 -
byteps/sh/server.sh                                |   62 +-
byteps/sh/scheduler.sh                             |   58 +-
...
```

如果你想看**完整差异**或**具体文件的 diff**，告诉我分支对（例如 `origin/hc_bps` vs `master`），我可以追加到文档中。
