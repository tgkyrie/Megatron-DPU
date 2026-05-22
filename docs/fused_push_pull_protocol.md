# BytePS fused push_pull 协议设计草案

本文记录 `feat/fused-push-pull-protocol` 分支的协议融合方案。目标不是直接删除
`push response` 和 `pull request`，而是把它们承载的语义合并到一次
`fused push_pull` 请求/响应中，减少控制面往返，同时保持 BytePS 当前同步语义。

## 1. 当前四段流程

当前默认路径是：

```text
worker -> server: push request       携带梯度分片数据
server -> worker: push response      空响应，用于完成 ZPush timestamp
worker -> server: pull request       携带 worker 输出缓冲区地址/长度/rkey
server -> worker: pull response      携带聚合后的梯度分片数据
```

对应主要代码：

- worker 队列串联：`byteps/byteps/common/operations.cc`
  - `GetPushQueueList()` 加入 `PUSH`
  - `GetPullQueueList()` 加入 `PULL`
- worker 发送：
  - `RunPushLoopOnce()` 调 `KVWorker::ZPush(...)`
  - `RunPullLoopOnce()` 调 `KVWorker::ZPull(...)`
- ps-lite 完成语义：
  - `KVWorker::Process()` 收到 response 后推进 timestamp
  - `Customer::Receiving()` 负责 response 计数和唤醒
- server 聚合：
  - `BytePSHandler()` 收 push，server engine 完成 copy/sum/all-recv
  - `SendPushResponse()` 发空 push response
  - `SendPullResponse()` 发 pull response
- transport：
  - RDMA pull request 注册/携带远端写目标 `addr/rkey`
  - UCX pull request 缓存本地输出地址，后续 pull response 写入该地址

所以 `push response` 和 `pull request` 不能裸删：

- `push response` 负责让 `ZPush` 的 timestamp 完成，并触发队列进入 `PULL`；
- `pull request` 负责把 output buffer 地址、长度和 RDMA rkey / UCX 本地地址语义传给
  server/transport。

## 2. 目标两段流程

融合后目标路径：

```text
worker -> server: fused push request 携带梯度分片数据 + pull 输出缓冲区元信息
server -> worker: pull response      携带聚合后的梯度分片数据，完成 fused timestamp
```

关键约束：

- 保留旧路径作为默认路径。
- 通过环境变量开关启用，例如 `BYTEPS_ENABLE_FUSED_PUSH_PULL=1`。
- 第一阶段只支持默认 dense push_pull，不覆盖 group register、compressor 注册、async
  training 和 GDR 特化路径。
- fused request 的 timestamp 必须覆盖完整 push_pull，而不是只覆盖 push。

## 3. 建议代码改动

### 3.1 ps-lite 元信息

增加一个协议标记：

```cpp
Meta::fused_push_pull
RawMeta::fused_push_pull
```

用途：

- transport 不需要理解 BytePS 的 `RequestType` 编码；
- fused push request 仍然保持 `push=true, request=true`，同时标记这是完整 push_pull；
- server 可按 `req_meta.fused_push_pull` 生成 synthetic pull meta。

### 3.2 KVWorker API

新增：

```cpp
int ZFusedPushPull(const SArray<Key>& keys,
                   const SArray<Val>& push_vals,
                   SArray<Val>* pull_vals,
                   SArray<int>* lens,
                   int cmd,
                   const Callback& cb);
```

行为：

- 分片逻辑复用现有 slicer；
- request 发送 push 数据；
- `msg.meta.addr` 指向 pull/output buffer；
- `msg.meta.val_len` 仍是该分片长度；
- RDMA 路径把 output buffer 注册并把 rkey 写入 `msg.meta.option`；
- response 到达后复用 `Pull_()` 的 callback 校验和清理逻辑，最终执行用户 callback。

### 3.3 worker 队列

启用 fused 时：

- `GetPushQueueList()` 仍保留 `PUSH`；
- `GetPullQueueList()` 对 root worker 不再加入 `PULL`，但仍保留后续 `COPYH2D` /
  `BROADCAST` 等阶段；
- `RunPushLoopOnce()` 在满足条件时调用 `ZFusedPushPull(...)`；
- fused callback 完成后调用一次 `FinishOrProceed(task)`，此时队列从 `PUSH` 直接进入
  `COPYH2D` 或后续阶段。

不满足条件时走旧 `ZPush + ZPull` 路径。

### 3.4 server handler

`BytePSHandler()` 对 fused push request：

- 正常执行现有 push 聚合逻辑；
- 不调用 `SendPushResponse()`；
- 复制一份 `req_meta` 作为 synthetic pull meta：

```cpp
auto pull_meta = req_meta;
pull_meta.push = false;
```

- 如果聚合尚未完成，把 synthetic pull meta 放入 `state->q_pull_reqmeta`；
- server engine 的 `ALL_RECV` 完成后，复用现有 `SendPullResponse(...)` 发送 pull
  response；
- response 使用原 fused timestamp，让 worker 的一次 request 完成整个 push_pull。

### 3.5 RDMA transport

fused request 仍走 push request data path，但额外处理 output buffer：

- `RegisterMemory()` 对 fused request 同时注册：
  - `msg.data[1]`：push 数据源；
  - `msg.meta.addr`：pull response 目标；
- `AddMeta()` 对 fused request 写入 output buffer 的 rkey 到 `msg.meta.option`；
- `SendPushRequest()` 不变，仍发送 push 数据；
- server 之后发 `SendPullResponse()` 时复用 `msg.meta.addr` 和 `msg.meta.option`。

### 3.6 UCX transport

fused request 仍是 data message，但需要额外缓存 output buffer 地址：

- `SendMsg()` 在识别 fused request 时调用 `rx_pool_->CacheLocalAddress(key, addr)`；
- `IsDataMsg()` 仍返回 true，保证 push 数据正常发送；
- pull response 仍复用现有接收逻辑。

## 4. 分阶段实现

第一阶段：协议骨架

- 增加 `Meta/RawMeta` 字段和 pack/unpack；
- 增加 `RequestType::kFusedPushPull`；
- 增加 env knob；
- 编译通过，默认关闭，不改变旧行为。

第二阶段：worker + server 语义

- 增加 `ZFusedPushPull()`；
- worker 队列启用 fused 路径；
- server 用 synthetic pull meta 替代 push response + pull request；
- 先支持非 GDR、非 compression、sync dense push_pull。

第三阶段：transport

- RDMA 注册 output buffer 并传 rkey；
- UCX 缓存 output buffer 地址；
- 确认 `put_zcopy` baseline 下数据生命周期安全。

第四阶段：验证

最小验证顺序：

1. fused 关闭：确认旧路径行为和性能不变；
2. `1w1s` BytePS-only `vgg16`：确认能看到 `Model:`、`Running warmup`、
   `Total img/sec`；
3. `2w2s`：确认多 worker 聚合语义正确；
4. `8w8s`：对比 baseline 吞吐和 server profile 控制消息数量。

成功标准：

- worker 全部 `rc=0`；
- 无 `Endpoint is not connected` / `Connection reset by remote peer` / segfault；
- fused 开启时 server profile 中不再出现独立 `push resp` 和 `pull req` 往返；
- 大 tensor 下吞吐不下降，小 tensor/多 partition 场景有控制面收益。

## 5. 风险点

- `msg.meta.addr` 当前在 push request 中默认指向 push 数据，fused 后要改为 output
  buffer 地址；transport 不能再隐含依赖该字段表示 push source。
- RDMA `meta.option` 已用于 rkey，UCX 已用于 `UCX_OPTION_*`，不要把它当通用协议标记。
- compression 改变 push 数据长度，第一阶段不要融合 compression 路径。
- GDR 路径有独立 buffer 选择逻辑，第一阶段先排除。
- server 的 `state->seen_sender` / `pull_cnt` 是一轮聚合的完成门控，synthetic pull
  meta 必须复用这套去重逻辑。

