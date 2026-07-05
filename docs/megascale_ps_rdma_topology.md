# MegaScalePS RDMA 拓扑

本文档记录 MegaScalePS 验证时默认使用的机器、角色和 RDMA 网口。

## 基本原则

- 默认只使用 `roce1`，也就是 `192.168.1.x` 网段。
- `roce1` 对应的 HCA 统一是 `mlx5_1`。
- `roce0` / `192.168.2.x` / `mlx5_0` 只作为备用或双 rail 测试时使用。
- worker 默认放在有 GPU 的机器上。
- server 默认和 worker 分开；优先使用不跑 worker 的机器作为 server。
- scheduler 默认放在 `gpu01`，不计入 `DMLC_NUM_SERVER`。

## 默认规模

### 最小 RDMA 测试

最小测试使用 `2 server + 2 worker`：

```bash
export DMLC_NUM_WORKER=2
export DMLC_NUM_SERVER=2
export DMLC_PS_ROOT_URI=192.168.1.10
export DMLC_PS_ROOT_PORT=9010
export DMLC_ENABLE_RDMA=ibverbs
export NCCL_IB_HCA=mlx5_1
```

| Role | Host | Container | RDMA IP | Interface | HCA |
| --- | --- | --- | --- | --- | --- |
| scheduler | gpu01 | megascale_ps | 192.168.1.10 | ens39f1np1 | mlx5_1 |
| worker 0 | gpu01 | megascale_ps | 192.168.1.10 | ens39f1np1 | mlx5_1 |
| worker 1 | gpu02 | megascale_ps | 192.168.1.11 | ens39f1np1 | mlx5_1 |
| server 0 | R750-1 | megascale_ps | 192.168.1.40 | enp202s0f1np1 | mlx5_1 |
| server 1 | R750-2 | megascale_ps | 192.168.1.41 | enp202s0f1np1 | mlx5_1 |

### 正常全量验证

正常全量验证使用 `8 server + 8 worker`，刚好覆盖 16 台机器。`gpu01` 同时作为 scheduler 和 worker 0。

```bash
export DMLC_NUM_WORKER=8
export DMLC_NUM_SERVER=8
export DMLC_PS_ROOT_URI=192.168.1.10
export DMLC_PS_ROOT_PORT=9010
export DMLC_ENABLE_RDMA=ibverbs
export NCCL_IB_HCA=mlx5_1
```

默认 worker 机器：

| Worker rank | Host | RDMA IP | Interface | HCA |
| --- | --- | --- | --- | --- |
| 0 | gpu01 | 192.168.1.10 | ens39f1np1 | mlx5_1 |
| 1 | gpu02 | 192.168.1.11 | ens39f1np1 | mlx5_1 |
| 2 | gpu03 | 192.168.1.12 | ens39f1np1 | mlx5_1 |
| 3 | gpu04 | 192.168.1.13 | ens39f1np1 | mlx5_1 |
| 4 | asus01 | 192.168.1.20 | ens93f1np1 | mlx5_1 |
| 5 | asus02 | 192.168.1.21 | ens93f1np1 | mlx5_1 |
| 6 | asus03 | 192.168.1.22 | ens93f1np1 | mlx5_1 |
| 7 | asus04 | 192.168.1.23 | ens93f1np1 | mlx5_1 |

默认 server 机器：

| Server rank | Host | RDMA IP | Interface | HCA |
| --- | --- | --- | --- | --- |
| 0 | R750-1 | 192.168.1.40 | enp202s0f1np1 | mlx5_1 |
| 1 | R750-2 | 192.168.1.41 | enp202s0f1np1 | mlx5_1 |
| 2 | R750-3 | 192.168.1.42 | enp202s0f1np1 | mlx5_1 |
| 3 | R750-4 | 192.168.1.43 | enp202s0f1np1 | mlx5_1 |
| 4 | server12 | 192.168.1.30 | ens3f1np1 | mlx5_1 |
| 5 | server13 | 192.168.1.31 | ens3f1np1 | mlx5_1 |
| 6 | server14 | 192.168.1.32 | ens3f1np1 | mlx5_1 |
| 7 | server15 | 192.168.1.33 | ens3f1np1 | mlx5_1 |

## 全部机器清单

| Host | Default full role | Small test role | GPU use | roce1 IP | roce1 interface | roce1 HCA | roce0 IP | roce0 interface | roce0 HCA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu01 | scheduler + worker 0 | scheduler + worker 0 | worker | 192.168.1.10 | ens39f1np1 | mlx5_1 | 192.168.2.10 | ens39f0np0 | mlx5_0 |
| gpu02 | worker 1 | worker 1 | worker | 192.168.1.11 | ens39f1np1 | mlx5_1 | 192.168.2.11 | ens39f0np0 | mlx5_0 |
| gpu03 | worker 2 | unused | worker | 192.168.1.12 | ens39f1np1 | mlx5_1 | 192.168.2.12 | ens39f0np0 | mlx5_0 |
| gpu04 | worker 3 | unused | worker | 192.168.1.13 | ens39f1np1 | mlx5_1 | 192.168.2.13 | ens39f0np0 | mlx5_0 |
| asus01 | worker 4 | unused | worker | 192.168.1.20 | ens93f1np1 | mlx5_1 | 192.168.2.20 | ens93f0np0 | mlx5_0 |
| asus02 | worker 5 | unused | worker | 192.168.1.21 | ens93f1np1 | mlx5_1 | 192.168.2.21 | ens93f0np0 | mlx5_0 |
| asus03 | worker 6 | unused | worker | 192.168.1.22 | ens93f1np1 | mlx5_1 | 192.168.2.22 | ens93f0np0 | mlx5_0 |
| asus04 | worker 7 | unused | worker | 192.168.1.23 | ens93f1np1 | mlx5_1 | 192.168.2.23 | ens93f0np0 | mlx5_0 |
| R750-1 | server 0 | server 0 | server | 192.168.1.40 | enp202s0f1np1 | mlx5_1 | 192.168.2.40 | enp202s0f0np0 | mlx5_0 |
| R750-2 | server 1 | server 1 | server | 192.168.1.41 | enp202s0f1np1 | mlx5_1 | 192.168.2.41 | enp202s0f0np0 | mlx5_0 |
| R750-3 | server 2 | unused | server | 192.168.1.42 | enp202s0f1np1 | mlx5_1 | 192.168.2.42 | enp202s0f0np0 | mlx5_0 |
| R750-4 | server 3 | unused | server | 192.168.1.43 | enp202s0f1np1 | mlx5_1 | 192.168.2.43 | enp202s0f0np0 | mlx5_0 |
| server12 | server 4 | unused | server | 192.168.1.30 | ens3f1np1 | mlx5_1 | 192.168.2.30 | ens3f0np0 | mlx5_0 |
| server13 | server 5 | unused | server | 192.168.1.31 | ens3f1np1 | mlx5_1 | 192.168.2.31 | ens3f0np0 | mlx5_0 |
| server14 | server 6 | unused | server | 192.168.1.32 | ens3f1np1 | mlx5_1 | 192.168.2.32 | ens3f0np0 | mlx5_0 |
| server15 | server 7 | unused | server | 192.168.1.33 | ens3f1np1 | mlx5_1 | 192.168.2.33 | ens3f0np0 | mlx5_0 |

## Per-host Environment

Each role process must set the host-specific `DMLC_INTERFACE` and `DMLC_NODE_HOST`.

Examples:

```bash
# gpu01 worker or scheduler
export DMLC_INTERFACE=ens39f1np1
export DMLC_NODE_HOST=192.168.1.10

# gpu02 worker
export DMLC_INTERFACE=ens39f1np1
export DMLC_NODE_HOST=192.168.1.11

# R750-1 server
export DMLC_INTERFACE=enp202s0f1np1
export DMLC_NODE_HOST=192.168.1.40

# server12 server
export DMLC_INTERFACE=ens3f1np1
export DMLC_NODE_HOST=192.168.1.30
```

## Notes

- `server12-15` are assigned to the server pool in the default 8+8 topology so that the full run covers all 16 machines while keeping server and worker pools separate.
- If `server12-15` need to be used as GPU workers later, move them explicitly from the server pool to the worker pool and update `DMLC_NUM_WORKER` / `DMLC_NUM_SERVER`.
- At the time this file was updated, `asus03` had `megascale_ps` container status `Exited (255)`, but the host RDMA/GPU inventory was readable and is recorded above.
