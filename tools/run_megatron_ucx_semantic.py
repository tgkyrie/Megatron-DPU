#!/usr/bin/env python3
"""Current Megatron host/DPU/NCCL experiment runner.

Worker-side training parameters should come from the
Megatron-LM/examples/qwen/*.sh defaults whenever possible. This runner only
passes topology/runtime values to workers, plus explicit NCCL-disable flags for
baseline cases. Scheduler/server roles do not run those training scripts, so
their BytePS/UCX environment is set here and kept aligned with the script
defaults.
"""
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pexpect


ROOT = Path("/home/cmcc/CMCC/Megatron-DPU")
STAMP = time.strftime("%Y%m%d-%H%M%S")
OUT = ROOT / "docs" / "benchmark_runs" / f"{STAMP}-megatron-host-vs-dpu-dataaddr"
LOGDIR = OUT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

X86_IMAGE = os.environ.get("X86_IMAGE", "192.168.1.10:5000/byteps:latest")
WORKER_IMAGE = os.environ.get("WORKER_IMAGE", "192.168.1.10:5000/megatron-dpu:latest")
DPU_IMAGE = os.environ.get("DPU_IMAGE", "192.168.1.10:5000/byteps-server:latest")

SCHEDULER = "gpu01"
HOST_SERVERS = ["R750-1", "R750-2", "R750-3", "R750-4", "server12", "server13", "server14", "server15"]
WORKERS = ["gpu01", "gpu02", "gpu03", "gpu04", "asus01", "asus02", "asus03", "asus04"]
WORKER_IPS = ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13",
              "192.168.1.20", "192.168.1.21", "192.168.1.22", "192.168.1.23"]
HOST_SERVER_IPS = ["192.168.1.40", "192.168.1.41", "192.168.1.42", "192.168.1.43",
                   "192.168.1.30", "192.168.1.31", "192.168.1.32", "192.168.1.33"]
DPU_SERVER_HOSTS = HOST_SERVERS
DPU_SERVER_IPS = ["192.168.1.97", "192.168.1.96", "192.168.1.98", "192.168.1.99",
                  "192.168.1.95", "192.168.1.93", "192.168.1.92", "192.168.1.94"]

DPU_PASS = os.environ.get("DPU_PASS", "ubuntu")
FUSED_PUSH_PULL = os.environ.get("BYTEPS_ENABLE_FUSED_PUSH_PULL", "1")
SERVER_ENABLE_SCHEDULE = os.environ.get("BYTEPS_SERVER_ENABLE_SCHEDULE", "0")
NETWORK_MODE = os.environ.get("NETWORK_MODE", "ucx").lower()
if NETWORK_MODE not in ("rdma", "ucx"):
    raise SystemExit("NETWORK_MODE must be rdma or ucx")
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", "10"))
EVAL_INTERVAL = os.environ.get("EVAL_INTERVAL", "100")
EVAL_ITERS = os.environ.get("EVAL_ITERS", "10")
MONITOR_TIMEOUT = int(os.environ.get("MONITOR_TIMEOUT", "480"))
NO_ITER_TIMEOUT = int(os.environ.get("NO_ITER_TIMEOUT", "300"))
ROLE_TIMEOUT = int(os.environ.get("ROLE_TIMEOUT", "600"))
WORKER_TIMEOUT = int(os.environ.get("WORKER_TIMEOUT", "600"))
PS_PORT_BASE = int(os.environ.get("PS_PORT_BASE", "9080"))
MASTER_PORT_BASE = int(os.environ.get("MASTER_PORT_BASE", "19340"))
RUN_CASES = [c.strip() for c in os.environ.get("RUN_CASES", "hostservers,dpuservers").split(",") if c.strip()]
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
SYNC_QWEN_SCRIPTS = os.environ.get("SYNC_QWEN_SCRIPTS", "1")
QWEN_SCRIPT_SOURCE_DIR = os.environ.get("QWEN_SCRIPT_SOURCE_DIR", "")
UCX_NET_DEVICES = os.environ.get("UCX_NET_DEVICES_OVERRIDE", "mlx5_0:1,mlx5_1:1")
DPU_UCX_NET_DEVICES = os.environ.get("DPU_UCX_NET_DEVICES_OVERRIDE", "mlx5_3:1")
UCX_MAX_EAGER_RAILS = os.environ.get("UCX_MAX_EAGER_RAILS_OVERRIDE", "2")
UCX_MAX_RNDV_RAILS = os.environ.get("UCX_MAX_RNDV_RAILS_OVERRIDE", "2")
CUDA_VISIBLE_DEVICES_OVERRIDE = os.environ.get("CUDA_VISIBLE_DEVICES_OVERRIDE", "")
UCX_RAIL_VARIANTS = [v.strip() for v in os.environ.get("UCX_RAIL_VARIANTS", "default,single,dual").split(",") if v.strip()]
DMLC_NUM_PORTS = os.environ.get("DMLC_NUM_PORTS", "").strip()

WORKLOAD = os.environ.get("WORKLOAD", "dp_qwen3b")
WORKLOADS = {
    "dp_qwen3b": {
        "script": "examples/qwen/train_qwen_3b.sh",
        "parallel": "DP",
        "model": "qwen_3b",
        "tp_size": "1",
        "byteps_flags": {"USE_DPU": "1"},
        "nccl_flags": {"USE_DPU": "0"},
    },
    "tp_llama2_7b": {
        "script": "examples/qwen/train_llama_7b_tp_byteps.sh",
        "parallel": "TP",
        "model": "llama2_7b",
        "tp_size": "8",
        "byteps_flags": {"USE_DPU": "1"},
        "nccl_flags": {"USE_DPU": "0"},
    },
    "tp_qwen3_4b": {
        "script": "examples/qwen/train_qwen3_4b_tp_byteps.sh",
        "parallel": "TP",
        "model": "qwen3_4b",
        "tp_size": "8",
        "byteps_flags": {},
        "nccl_flags": {"USE_DPU": "0", "USE_DPU_DP": "0", "USE_DPU_TP": "0"},
    },
    "dptp_qwen3_4b": {
        "script": "examples/qwen/train_qwen3_4b_tp_dp_byteps.sh",
        "parallel": "DPTP",
        "model": "qwen3_4b",
        "tp_size": "2",
        "byteps_flags": {"USE_DPU_DP": "1", "USE_DPU_TP": "1"},
        "nccl_flags": {"USE_DPU_DP": "0", "USE_DPU_TP": "0"},
    },
}
if WORKLOAD not in WORKLOADS:
    raise SystemExit(f"unknown WORKLOAD={WORKLOAD}; choose one of {','.join(WORKLOADS)}")
WORKLOAD_CFG = WORKLOADS[WORKLOAD]

BYTEPS_DEFAULTS = {
    "rdma": {
        "dp_qwen3b": {
            "partition": "4194304",
            "address_pool": "10240",
            "rx_depth": "512",
            "start_depth": "32",
        },
        "tp_llama2_7b": {
            "partition": "2097152",
            "address_pool": "10240",
            "rx_depth": "256",
            "start_depth": "32",
        },
        "tp_qwen3_4b": {
            "partition": "2097152",
            "address_pool": "10240",
            "rx_depth": "256",
            "start_depth": "32",
        },
        "dptp_qwen3_4b": {
            "partition": "4194304",
            "address_pool": "10240",
            "rx_depth": "1024",
            "start_depth": "32",
        },
    },
    "ucx": {
        "dp_qwen3b": {
            "partition": "4194304",
            "address_pool": "10240",
            "rx_depth": "512",
            "start_depth": "32",
        },
        "tp_llama2_7b": {
            "partition": "1048576",
            "address_pool": "10240",
            "rx_depth": "512",
            "start_depth": "32",
        },
        "tp_qwen3_4b": {
            "partition": "4194304",
            "address_pool": "10240",
            "rx_depth": "512",
            "start_depth": "32",
        },
        "dptp_qwen3_4b": {
            "partition": "1048576",
            "address_pool": "10240",
            "rx_depth": "512",
            "start_depth": "32",
        },
    },
}
BYTEPS_DEFAULT = BYTEPS_DEFAULTS[NETWORK_MODE][WORKLOAD]
BYTEPS_PARTITION_BYTES = os.environ.get("BYTEPS_PARTITION_BYTES_DEFAULT", BYTEPS_DEFAULT["partition"])
BYTEPS_ADDRESS_POOL_SIZE = os.environ.get("BYTEPS_ADDRESS_POOL_SIZE_DEFAULT", BYTEPS_DEFAULT["address_pool"])
BYTEPS_RDMA_RX_DEPTH = os.environ.get("BYTEPS_RDMA_RX_DEPTH_DEFAULT", BYTEPS_DEFAULT["rx_depth"])
BYTEPS_RDMA_START_DEPTH = os.environ.get("BYTEPS_RDMA_START_DEPTH_DEFAULT", BYTEPS_DEFAULT["start_depth"])


def worker_network_overrides():
    """Only pass worker-side network overrides that are explicit experiment knobs."""
    exports = []
    if NETWORK_MODE == "rdma":
        exports.extend([
            "export DMLC_ENABLE_UCX=0",
            "export DMLC_ENABLE_RDMA=ibverbs",
        ])

    explicit_defaults = [
        ("BYTEPS_PARTITION_BYTES_DEFAULT", "BYTEPS_PARTITION_BYTES", BYTEPS_PARTITION_BYTES),
        ("BYTEPS_ADDRESS_POOL_SIZE_DEFAULT", "BYTEPS_ADDRESS_POOL_SIZE", BYTEPS_ADDRESS_POOL_SIZE),
        ("BYTEPS_RDMA_RX_DEPTH_DEFAULT", "BYTEPS_RDMA_RX_DEPTH", BYTEPS_RDMA_RX_DEPTH),
        ("BYTEPS_RDMA_START_DEPTH_DEFAULT", "BYTEPS_RDMA_START_DEPTH", BYTEPS_RDMA_START_DEPTH),
        ("BYTEPS_ENABLE_FUSED_PUSH_PULL", "BYTEPS_ENABLE_FUSED_PUSH_PULL", FUSED_PUSH_PULL),
        ("BYTEPS_SERVER_ENABLE_SCHEDULE", "BYTEPS_SERVER_ENABLE_SCHEDULE", SERVER_ENABLE_SCHEDULE),
    ]
    for env_name, script_name, value in explicit_defaults:
        if env_name in os.environ:
            exports.append(f"export {script_name}={shlex.quote(str(value))}")
    return "\n".join(exports)

MODEL_ENV_NAMES = [
    "NUM_LAYERS",
    "HIDDEN_SIZE",
    "FFN_HIDDEN_SIZE",
    "NUM_HEADS",
    "KV_CHANNELS",
    "SEQ_LENGTH",
    "MAX_POSITION_EMBEDDINGS",
    "MICRO_BATCH_SIZE",
    "GLOBAL_BATCH_SIZE",
    "VOCAB_SIZE",
    "MAKE_VOCAB_SIZE_DIVISIBLE_BY",
    "NUM_QUERY_GROUPS",
    "NORM_EPSILON",
    "ROTARY_BASE",
    "ROTARY_PERCENT",
    "UNTIE_EMBEDDINGS",
    "DISABLE_BIAS_LINEAR",
]
MODEL_OVERRIDES = {name: os.environ[name] for name in MODEL_ENV_NAMES if name in os.environ}
MODEL_EXPORTS = "\n".join(
    f"export {name}={shlex.quote(value)}" for name, value in MODEL_OVERRIDES.items()
)
WORKER_SCRIPT_OVERRIDE_NAMES = [
    "NUM_NODES",
    "GPUS_PER_NODE",
    "TP_SIZE",
    "TRAIN_ITERS",
    "EVAL_INTERVAL",
    "EVAL_ITERS",
    "SEED",
]
WORKER_SCRIPT_OVERRIDES = {
    name: os.environ[name] for name in WORKER_SCRIPT_OVERRIDE_NAMES if name in os.environ
}
WORKER_SCRIPT_EXPORTS = "\n".join(
    f"export {name}={shlex.quote(value)}" for name, value in WORKER_SCRIPT_OVERRIDES.items()
)
WORKER_EXTRA_EXPORT_NAMES = [
    name.strip()
    for name in os.environ.get("WORKER_EXTRA_EXPORT_NAMES", "").split(",")
    if name.strip()
]
WORKER_EXTRA_EXPORTS = "\n".join(
    f"export {name}={shlex.quote(os.environ[name])}"
    for name in WORKER_EXTRA_EXPORT_NAMES
    if name in os.environ
)
QWEN_SCRIPT_NAMES = [
    "train_qwen_3b.sh",
    "train_qwen_3b_tp_byteps.sh",
    "train_qwen_3b_tp_dp_byteps.sh",
    "train_qwen3_4b_tp_byteps.sh",
    "train_qwen3_4b_tp_dp_byteps.sh",
    "train_llama_7b_tp_byteps.sh",
]


def run_local(cmd, timeout=120, check=True):
    print(f"+ {cmd}", flush=True)
    cp = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout)
    if cp.stdout:
        print(cp.stdout, end="")
    if cp.stderr:
        print(cp.stderr, end="", file=sys.stderr)
    if check and cp.returncode != 0:
        raise RuntimeError(f"command failed rc={cp.returncode}: {cmd}")
    return cp


def ssh(host, cmd, timeout=120, check=True):
    quoted = shlex.quote(cmd)
    return run_local(f"ssh {shlex.quote(host)} {quoted}", timeout=timeout, check=check)


def dpu(host, cmd, timeout=180, check=True):
    remote = (
        "ssh -tt -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"ubuntu@192.168.100.2 {shlex.quote(cmd)}"
    )
    print(f"+ ssh {host} [dpu] {cmd[:160]}", flush=True)
    child = pexpect.spawn("ssh", ["-tt", host, remote], encoding="utf-8", timeout=timeout)
    chunks = []
    rc = None
    while True:
        try:
            i = child.expect([r"[Pp]assword:", r"continue connecting.*\?", pexpect.EOF, pexpect.TIMEOUT])
        except pexpect.EOF:
            break
        except pexpect.TIMEOUT:
            child.close(force=True)
            if check:
                raise RuntimeError(f"DPU command timed out on {host}: {cmd}")
            return ""
        chunks.append(child.before)
        if i == 0:
            child.sendline(DPU_PASS)
        elif i == 1:
            child.sendline("yes")
        elif i == 2:
            break
        elif i == 3:
            child.close(force=True)
            if check:
                raise RuntimeError(f"DPU command timed out on {host}: {cmd}")
            return ""
    child.close()
    rc = child.exitstatus
    out = "".join(chunks) + (child.before or "")
    if out:
        print(out)
    if check and rc not in (0, None):
        raise RuntimeError(f"DPU command failed rc={rc} on {host}: {cmd}")
    return out


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def common_env(port, include_byteps=True, worker_use_dpu=True):
    num_ports_export = f"export DMLC_NUM_PORTS={shlex.quote(DMLC_NUM_PORTS)}\n" if DMLC_NUM_PORTS else ""
    base = f"""
export DMLC_PS_ROOT_URI=192.168.1.10
export DMLC_PS_ROOT_PORT={port}
export DMLC_NUM_WORKER=8
export DMLC_NUM_SERVER=8
{num_ports_export.rstrip()}
unset WORKER_ID
"""
    if not include_byteps:
        return base
    if NETWORK_MODE == "ucx":
        return base + f"""
export DMLC_ENABLE_UCX=1
export DMLC_ENABLE_RDMA=0
export DMLC_USE_GDR=0
export NCCL_IB_HCA=mlx5_1
export UCX_TLS=rc
export UCX_NET_DEVICES={UCX_NET_DEVICES}
export UCX_MAX_EAGER_RAILS={UCX_MAX_EAGER_RAILS}
export UCX_MAX_RNDV_RAILS={UCX_MAX_RNDV_RAILS}
export UCX_RNDV_THRESH=8k
export UCX_IB_TRAFFIC_CLASS=106
export PSLITE_UCX_IB_TRAFFIC_CLASS=106
export UCX_WARN_UNUSED_ENV_VARS=n
export PSLITE_UCX_USE_MT_MUTEX=y
export PSLITE_UCX_RNDV_SCHEME=put_zcopy
export BYTEPS_ENABLE_FUSED_PUSH_PULL={FUSED_PUSH_PULL}
export BYTEPS_SERVER_ENABLE_SCHEDULE={SERVER_ENABLE_SCHEDULE}
export BYTEPS_PARTITION_BYTES={BYTEPS_PARTITION_BYTES}
export BYTEPS_ADDRESS_POOL_SIZE={BYTEPS_ADDRESS_POOL_SIZE}
export BYTEPS_RDMA_RX_DEPTH={BYTEPS_RDMA_RX_DEPTH}
export BYTEPS_RDMA_START_DEPTH={BYTEPS_RDMA_START_DEPTH}
"""
    return base + f"""
export DMLC_ENABLE_UCX=0
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_USE_GDR=0
export NCCL_IB_HCA=mlx5_1
export BYTEPS_ENABLE_FUSED_PUSH_PULL={FUSED_PUSH_PULL}
export BYTEPS_SERVER_ENABLE_SCHEDULE={SERVER_ENABLE_SCHEDULE}
export BYTEPS_PARTITION_BYTES={BYTEPS_PARTITION_BYTES}
export BYTEPS_ADDRESS_POOL_SIZE={BYTEPS_ADDRESS_POOL_SIZE}
export BYTEPS_RDMA_RX_DEPTH={BYTEPS_RDMA_RX_DEPTH}
export BYTEPS_RDMA_START_DEPTH={BYTEPS_RDMA_START_DEPTH}
"""


def rail_settings(rail_variant, dpu_server=False):
    if NETWORK_MODE != "ucx":
        return {
            "mode": "none",
            "net_devices": "",
            "eager_rails": "",
            "rndv_rails": "",
            "hca": "mlx5_1",
            "worker_export": "",
        }
    if dpu_server:
        return {
            "mode": "dpu-single",
            "net_devices": DPU_UCX_NET_DEVICES,
            "eager_rails": "1",
            "rndv_rails": "1",
            "hca": "mlx5_3",
            "worker_export": "",
        }
    if rail_variant == "single":
        return {
            "mode": "single",
            "net_devices": "mlx5_1:1",
            "eager_rails": "1",
            "rndv_rails": "1",
            "hca": "mlx5_1",
            "worker_export": "export UCX_RAIL_MODE=single",
        }
    if rail_variant == "dual":
        return {
            "mode": "dual",
            "net_devices": UCX_NET_DEVICES,
            "eager_rails": UCX_MAX_EAGER_RAILS,
            "rndv_rails": UCX_MAX_RNDV_RAILS,
            "hca": "mlx5_1",
            "worker_export": "export UCX_RAIL_MODE=dual",
        }
    return {
        "mode": "default",
        "net_devices": UCX_NET_DEVICES,
        "eager_rails": UCX_MAX_EAGER_RAILS,
        "rndv_rails": UCX_MAX_RNDV_RAILS,
        "hca": "mlx5_1",
        "worker_export": "",
    }


def server_common_env(port, rail_variant="default", dpu_server=False):
    if NETWORK_MODE != "ucx":
        return common_env(port)
    rail = rail_settings(rail_variant, dpu_server=dpu_server)
    return common_env(port).replace(
        f"export NCCL_IB_HCA=mlx5_1\nexport UCX_TLS=rc\nexport UCX_NET_DEVICES={UCX_NET_DEVICES}\nexport UCX_MAX_EAGER_RAILS={UCX_MAX_EAGER_RAILS}\nexport UCX_MAX_RNDV_RAILS={UCX_MAX_RNDV_RAILS}",
        f"export NCCL_IB_HCA={rail['hca']}\nexport UCX_TLS=rc\nexport UCX_NET_DEVICES={rail['net_devices']}\nexport UCX_MAX_EAGER_RAILS={rail['eager_rails']}\nexport UCX_MAX_RNDV_RAILS={rail['rndv_rails']}",
    )


def docker_run_base(name, image, body, gpu=False, workdir=None):
    gpu_arg = "--gpus all" if gpu else ""
    wd = f"-w {shlex.quote(workdir)}" if workdir else ""
    return (
        f"sudo docker run --rm -d --name {shlex.quote(name)} "
        f"--privileged --ipc=host {gpu_arg} --net=host --shm-size=32g "
        f"--ulimit memlock=-1 --device=/dev/infiniband --cap-add IPC_LOCK "
        f"-v /tmp:/hosttmp {wd} {shlex.quote(image)} bash -lc {shlex.quote(body)}"
    )


def start_x86_scheduler(run, port, rail_variant="default"):
    body = f"""
echo running > /hosttmp/{run}-scheduler.status
{server_common_env(port, rail_variant=rail_variant)}
export DMLC_NODE_HOST=192.168.1.10
cd /usr/local
timeout {ROLE_TIMEOUT}s bash /usr/local/scheduler.sh > /hosttmp/{run}-scheduler.log 2>&1
echo rc=$? > /hosttmp/{run}-scheduler.status
"""
    ssh(SCHEDULER, docker_run_base(f"{run}-scheduler", X86_IMAGE, body), timeout=120)


def start_x86_server(run, host, host_ip, port, rail_variant="default"):
    body = f"""
echo running > /hosttmp/{run}-server.status
{server_common_env(port, rail_variant=rail_variant)}
export DMLC_NODE_HOST={host_ip}
cd /usr/local
timeout {ROLE_TIMEOUT}s bash /usr/local/server.sh > /hosttmp/{run}-server.log 2>&1
echo rc=$? > /hosttmp/{run}-server.status
"""
    ssh(host, docker_run_base(f"{run}-server", X86_IMAGE, body), timeout=120)


def ensure_dpu(host):
    ssh(host, "sudo ip addr add 192.168.100.1/30 dev tmfifo_net0 2>/dev/null || true; sudo ip link set tmfifo_net0 up", timeout=60)


def start_dpu_server(run, host, dpu_ip, port, rail_variant="default"):
    ensure_dpu(host)
    dpu_schedule = SERVER_ENABLE_SCHEDULE if SERVER_ENABLE_SCHEDULE else "0"
    body = f"""
echo running > /hosttmp/{run}-server.status
{server_common_env(port, rail_variant=rail_variant, dpu_server=True)}
export DMLC_NODE_HOST={dpu_ip}
export DMLC_INTERFACE=enp3s0f1s0
export NCCL_IB_HCA=mlx5_3
export UCX_NET_DEVICES={DPU_UCX_NET_DEVICES}
export UCX_MAX_EAGER_RAILS=1
export UCX_MAX_RNDV_RAILS=1
export BYTEPS_SERVER_ENABLE_SCHEDULE={dpu_schedule}
cd /usr/local
timeout {ROLE_TIMEOUT}s bash /usr/local/server.sh > /hosttmp/{run}-server.log 2>&1
echo rc=$? > /hosttmp/{run}-server.status
"""
    cmd = docker_run_base(f"{run}-server", DPU_IMAGE, body)
    dpu(host, cmd, timeout=180)


def workload_flag_exports(use_dpu):
    if use_dpu:
        return ""
    flags = WORKLOAD_CFG["byteps_flags"] if use_dpu else WORKLOAD_CFG["nccl_flags"]
    return "\n".join(f"export {k}={v}" for k, v in flags.items())


def start_worker(run, host, rank, host_ip, ps_port, master_port, use_dpu=True, rail_variant="default"):
    workload_flags = workload_flag_exports(use_dpu)
    script = WORKLOAD_CFG["script"]
    rail_export = rail_settings(rail_variant)["worker_export"] if use_dpu else ""
    network_overrides = worker_network_overrides() if use_dpu else ""
    body = f"""
echo running > /hosttmp/{run}-worker.status
{common_env(ps_port, include_byteps=False, worker_use_dpu=use_dpu)}
if [ -d /hosttmp/qwen_scripts ]; then
  cp /hosttmp/qwen_scripts/*.sh /usr/local/Megatron-LM/examples/qwen/
  chmod +x /usr/local/Megatron-LM/examples/qwen/*.sh
fi
{"export CUDA_VISIBLE_DEVICES=" + CUDA_VISIBLE_DEVICES_OVERRIDE if CUDA_VISIBLE_DEVICES_OVERRIDE else ""}
{workload_flags}
{rail_export}
{network_overrides}
export DMLC_NODE_HOST={host_ip}
export MASTER_ADDR=192.168.1.10
export MASTER_PORT={master_port}
export NODE_RANK={rank}
{WORKER_SCRIPT_EXPORTS}
{WORKER_EXTRA_EXPORTS}
{MODEL_EXPORTS}
cd /usr/local/Megatron-LM
timeout {WORKER_TIMEOUT}s bash {script} > /hosttmp/{run}-worker.log 2>&1
echo rc=$? > /hosttmp/{run}-worker.status
"""
    ssh(host, docker_run_base(f"{run}-worker", WORKER_IMAGE, body, gpu=True, workdir="/usr/local/Megatron-LM"), timeout=120)


def cat_host_file(host, path):
    cp = ssh(host, f"cat {shlex.quote(path)} 2>/dev/null || true", timeout=60, check=False)
    return cp.stdout


def cat_dpu_file(host, path):
    return dpu(host, f"cat {shlex.quote(path)} 2>/dev/null || true", timeout=60, check=False)


def stop_host_container(host, name):
    ssh(host, f"sudo docker stop -t 10 {shlex.quote(name)} >/dev/null 2>&1 || true", timeout=40, check=False)


def stop_dpu_container(host, name):
    dpu(host, f"sudo docker stop -t 10 {shlex.quote(name)} >/dev/null 2>&1 || true", timeout=60, check=False)


def cleanup_run_containers(run, use_dpu_servers):
    for host in WORKERS:
        stop_host_container(host, f"{run}-worker")
    stop_host_container(SCHEDULER, f"{run}-scheduler")
    if use_dpu_servers is True:
        for host in DPU_SERVER_HOSTS:
            stop_dpu_container(host, f"{run}-server")
    elif use_dpu_servers is False:
        for host in HOST_SERVERS:
            stop_host_container(host, f"{run}-server")


def get_host_status(host, run, role):
    text = cat_host_file(host, f"/tmp/{run}-{role}.status").strip()
    return text or "missing"


def parse_iterations(log):
    rows = []
    current_iter = None
    for line in log.splitlines():
        m = re.search(r"iteration\s+(\d+)/\s*\d+", line)
        if m:
            current_iter = int(m.group(1))
        t = re.search(r"elapsed time per iteration \(ms\):\s*([0-9.]+)", line)
        if t:
            rows.append({"iter": current_iter if current_iter is not None else len(rows) + 1,
                         "ms": float(t.group(1)), "loss": math.nan})
        loss = re.search(r"lm loss:\s*([0-9.eE+-]+)", line)
        if loss:
            if rows and math.isnan(rows[-1]["loss"]):
                rows[-1]["loss"] = float(loss.group(1))
            else:
                rows.append({"iter": current_iter if current_iter is not None else len(rows) + 1,
                             "ms": math.nan, "loss": float(loss.group(1))})
    return rows


def parse_eval_losses(log):
    losses = {}
    for line in log.splitlines():
        m = re.search(r"on (validation|test) set \| lm loss value:\s*([0-9.eE+-]+)", line)
        if m:
            losses[m.group(1)] = float(m.group(2))
    return losses


def summarize_rows(rows):
    timed = [r for r in rows if r["iter"] and r["iter"] >= 2 and not math.isnan(r["ms"])]
    vals = [r["ms"] for r in timed]
    if not vals:
        return math.nan, math.nan, 0
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def collect_case(run, case, use_dpu_servers):
    logs = {}
    statuses = {}
    scheduler_log = cat_host_file(SCHEDULER, f"/tmp/{run}-scheduler.log")
    scheduler_status = get_host_status(SCHEDULER, run, "scheduler")
    if scheduler_log or scheduler_status != "missing":
        logs[f"{SCHEDULER}-scheduler"] = scheduler_log
        statuses[f"{SCHEDULER}-scheduler"] = scheduler_status
    if use_dpu_servers:
        for host in DPU_SERVER_HOSTS:
            logs[f"{host}-dpu-server"] = cat_dpu_file(host, f"/tmp/{run}-server.log")
            statuses[f"{host}-dpu-server"] = cat_dpu_file(host, f"/tmp/{run}-server.status").strip() or "missing"
    elif use_dpu_servers is False:
        for host in HOST_SERVERS:
            server_log = cat_host_file(host, f"/tmp/{run}-server.log")
            server_status = get_host_status(host, run, "server")
            if server_log or server_status != "missing":
                logs[f"{host}-server"] = server_log
                statuses[f"{host}-server"] = server_status
    for host in WORKERS:
        logs[f"{host}-worker"] = cat_host_file(host, f"/tmp/{run}-worker.log")
        statuses[f"{host}-worker"] = get_host_status(host, run, "worker")

    for key, text in logs.items():
        write_text(LOGDIR / f"{run}-{key}.log", text)
    rank7 = logs["asus04-worker"]
    rows = parse_iterations(rank7)
    eval_losses = parse_eval_losses(rank7)
    iter_path = LOGDIR / f"{run}-rank7-iterations.tsv"
    iter_lines = ["iter\tms\tloss"]
    for r in rows:
        ms = "" if math.isnan(r["ms"]) else f"{r['ms']:.6f}"
        loss = "" if math.isnan(r["loss"]) else f"{r['loss']:.12g}"
        iter_lines.append(f"{r['iter']}\t{ms}\t{loss}")
    write_text(iter_path, "\n".join(iter_lines) + "\n")
    mean, std, n = summarize_rows(rows)
    status = "OK" if all(statuses[f"{h}-worker"].startswith("rc=0") for h in WORKERS) else "FAIL"
    return {
        "case": case,
        "run": run,
        "status": status,
        "mean": mean,
        "std": std,
        "n": n,
        "rows": rows,
        "eval_losses": eval_losses,
        "statuses": statuses,
        "rank7_log": str(LOGDIR / f"{run}-asus04-worker.log"),
        "iter_file": str(iter_path),
    }


def result_success(result):
    return result.get("status") == "OK" and result.get("n", 0) > 0


def monitor_run(run, case, eval_phase=False):
    deadline = time.time() + MONITOR_TIMEOUT
    start_time = time.time()
    last_count = -1
    last_eval = {}
    while time.time() < deadline:
        statuses = [get_host_status(h, run, "worker") for h in WORKERS]
        rank7_log = cat_host_file("asus04", f"/tmp/{run}-worker.log")
        rows = parse_iterations(rank7_log)
        count = len(rows)
        eval_losses = parse_eval_losses(rank7_log)
        changed = count != last_count or eval_losses != last_eval
        if changed:
            if eval_phase:
                print(f"[{case}:eval] iteration lines={count} eval={eval_losses} statuses={statuses}", flush=True)
            else:
                print(f"[{case}] rank7 iteration lines={count} statuses={statuses}", flush=True)
            last_count = count
            last_eval = eval_losses
        if all(s.startswith("rc=") for s in statuses):
            break
        if not eval_phase and count == 0 and time.time() - start_time > NO_ITER_TIMEOUT:
            print(f"[{case}] no iteration for {NO_ITER_TIMEOUT}s; stop this attempt", flush=True)
            break
        time.sleep(15)


def case_run_name(case, attempt):
    return f"mega-{case}-try{attempt}-{STAMP}"


def run_case(case, use_dpu_servers, ps_port, master_port, attempt, rail_variant="default"):
    run = case_run_name(case, attempt)
    print(f"=== CASE {case} run={run} rail={rail_variant} ps_port={ps_port} master_port={master_port} ===", flush=True)
    start_x86_scheduler(run, ps_port, rail_variant=rail_variant)
    time.sleep(2)
    if use_dpu_servers:
        for host, ip in zip(DPU_SERVER_HOSTS, DPU_SERVER_IPS):
            start_dpu_server(run, host, ip, ps_port, rail_variant=rail_variant)
    else:
        for host, ip in zip(HOST_SERVERS, HOST_SERVER_IPS):
            start_x86_server(run, host, ip, ps_port, rail_variant=rail_variant)
    time.sleep(10)
    for rank, (host, ip) in enumerate(zip(WORKERS, WORKER_IPS)):
        start_worker(run, host, rank, ip, ps_port, master_port, use_dpu=True, rail_variant=rail_variant)

    monitor_run(run, case, eval_phase=False)

    result = collect_case(run, case, use_dpu_servers)
    cleanup_run_containers(run, use_dpu_servers)
    result["ps_port"] = ps_port
    result["master_port"] = master_port
    result["attempt"] = attempt
    result["rail_variant"] = rail_variant
    return result


def run_nccl_case(ps_port, master_port, attempt):
    case = "nccl"
    run = case_run_name(case, attempt)
    print(f"=== CASE {case} run={run} master_port={master_port} ===", flush=True)
    for rank, (host, ip) in enumerate(zip(WORKERS, WORKER_IPS)):
        start_worker(run, host, rank, ip, ps_port, master_port, use_dpu=False)

    monitor_run(run, case, eval_phase=False)
    result = collect_case(run, case, None)
    cleanup_run_containers(run, None)
    result["ps_port"] = ps_port
    result["master_port"] = master_port
    result["attempt"] = attempt
    result["rail_variant"] = "none"
    return result


def failed_result(case, run, attempt, ps_port, master_port, exc, rail_variant="default"):
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    write_text(LOGDIR / f"{run}-exception.txt", text)
    return {
        "case": case,
        "run": run,
        "status": "EXCEPTION",
        "mean": math.nan,
        "std": math.nan,
        "n": 0,
        "rows": [],
        "eval_losses": {},
        "statuses": {"exception": repr(exc)},
        "rank7_log": str(LOGDIR / f"{run}-asus04-worker.log"),
        "iter_file": str(LOGDIR / f"{run}-rank7-iterations.tsv"),
        "ps_port": ps_port,
        "master_port": master_port,
        "attempt": attempt,
        "rail_variant": rail_variant,
    }


def run_with_retries(case, use_dpu_servers, start_offset):
    last = None
    attempts = max(1, MAX_RETRIES)
    for attempt in range(1, attempts + 1):
        offset = start_offset + attempt - 1
        ps_port = PS_PORT_BASE + offset
        master_port = MASTER_PORT_BASE + offset
        run = case_run_name(case, attempt)
        rail_variant = "none" if case == "nccl" else UCX_RAIL_VARIANTS[min(attempt - 1, len(UCX_RAIL_VARIANTS) - 1)]
        try:
            if case == "nccl":
                result = run_nccl_case(ps_port, master_port, attempt)
            else:
                result = run_case(case, use_dpu_servers, ps_port, master_port, attempt, rail_variant=rail_variant)
        except Exception as exc:
            print(f"[{case}] attempt {attempt}/{attempts} raised {exc!r}", flush=True)
            cleanup_run_containers(run, use_dpu_servers)
            result = failed_result(case, run, attempt, ps_port, master_port, exc, rail_variant=rail_variant)
        last = result
        if result_success(result):
            return result, start_offset + attempt
        print(f"[{case}] attempt {attempt}/{attempts} did not succeed: status={result.get('status')} n={result.get('n')}", flush=True)
        if attempt < attempts:
            time.sleep(20)
    return last, start_offset + attempts


def compare_losses(a_rows, b_rows):
    amap = {r["iter"]: r["loss"] for r in a_rows if not math.isnan(r["loss"])}
    bmap = {r["iter"]: r["loss"] for r in b_rows if not math.isnan(r["loss"])}
    common = sorted(set(amap) & set(bmap))
    diffs = [(i, amap[i], bmap[i], abs(amap[i] - bmap[i])) for i in common]
    max_abs = max((d[3] for d in diffs), default=math.nan)
    return common, diffs, max_abs


def sync_qwen_scripts():
    if SYNC_QWEN_SCRIPTS != "1":
        return
    tmp_script_dir = OUT / "qwen_scripts"
    tmp_script_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(QWEN_SCRIPT_SOURCE_DIR) if QWEN_SCRIPT_SOURCE_DIR else ROOT / "Megatron-LM" / "examples" / "qwen"
    for name in QWEN_SCRIPT_NAMES:
        src = source_dir / name
        text = src.read_text(encoding="utf-8", errors="replace")
        (tmp_script_dir / name).write_text(text, encoding="utf-8", errors="replace")
    files = " ".join(str(tmp_script_dir / name) for name in QWEN_SCRIPT_NAMES)
    for host in WORKERS:
        ssh(host, "mkdir -p /tmp/qwen_scripts", timeout=60)
        run_local(f"scp {files} {shlex.quote(host)}:/tmp/qwen_scripts/", timeout=120)


def main():
    sync_qwen_scripts()
    write_text(OUT / "images.tsv", f"role\timage\nx86_server_scheduler\t{X86_IMAGE}\nworker\t{WORKER_IMAGE}\ndpu_server\t{DPU_IMAGE}\n")
    results = []
    port_offset = 0
    if "nccl" in RUN_CASES:
        result, port_offset = run_with_retries("nccl", None, port_offset)
        results.append(result)
    if "hostservers" in RUN_CASES:
        result, port_offset = run_with_retries("hostservers", False, port_offset)
        results.append(result)
    if "dpuservers" in RUN_CASES:
        if results:
            time.sleep(20)
        result, port_offset = run_with_retries("dpuservers", True, port_offset)
        results.append(result)

    loss_a = next((r for r in results if r["case"] == "hostservers"), None)
    loss_b = next((r for r in results if r["case"] == "dpuservers"), None)
    loss_label_a, loss_label_b = "host_loss", "dpu_loss"
    if not loss_a or not loss_b:
        if len(results) >= 2:
            loss_a, loss_b = results[0], results[1]
            loss_label_a = f"{loss_a['case']}_loss"
            loss_label_b = f"{loss_b['case']}_loss"
        else:
            loss_a = loss_b = None
    if loss_a and loss_b:
        common, diffs, max_abs = compare_losses(loss_a["rows"], loss_b["rows"])
    else:
        common, diffs, max_abs = [], [], math.nan
    write_text(LOGDIR / "loss_compare.tsv", f"iter\t{loss_label_a}\t{loss_label_b}\tabs_diff\n" + "\n".join(
        f"{i}\t{a:.12g}\t{b:.12g}\t{d:.12g}" for i, a, b, d in diffs
    ) + "\n")

    lines = ["case\tattempt\trail_variant\trun\tstatus\tps_port\tmaster_port\tmean_ms_iter2_20\tstd_ms\tn\ttrain_valid_loss\ttrain_test_loss\trank7_log\titer_file"]
    for r in results:
        mean = "NA" if math.isnan(r["mean"]) else f"{r['mean']:.3f}"
        std = "NA" if math.isnan(r["std"]) else f"{r['std']:.3f}"
        train_valid = r["eval_losses"].get("validation", math.nan)
        train_test = r["eval_losses"].get("test", math.nan)
        fmt = lambda v: "NA" if math.isnan(v) else f"{v:.12g}"
        lines.append(f"{r['case']}\t{r.get('attempt', 1)}\t{r.get('rail_variant', '')}\t{r.get('run', '')}\t{r['status']}\t{r['ps_port']}\t{r['master_port']}\t{mean}\t{std}\t{r['n']}\t{fmt(train_valid)}\t{fmt(train_test)}\t{r['rank7_log']}\t{r['iter_file']}")
    write_text(OUT / "summary.tsv", "\n".join(lines) + "\n")

    status_lines = []
    for r in results:
        status_lines.append(f"## {r['case']}")
        for k in sorted(r["statuses"]):
            status_lines.append(f"{k}\t{r['statuses'][k]}")
    write_text(OUT / "statuses.tsv", "\n".join(status_lines) + "\n")

    md = [
        f"# Megatron host-server vs DPU-server BytePS {NETWORK_MODE.upper()} comparison",
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- workload: `{WORKLOAD}`",
        f"- parallel: `{WORKLOAD_CFG['parallel']}`",
        f"- model: `{WORKLOAD_CFG['model']}`",
        f"- script: `{WORKLOAD_CFG['script']}`",
        f"- worker image: `{WORKER_IMAGE}`",
        f"- x86 scheduler/server image: `{X86_IMAGE}`",
        f"- DPU server image: `{DPU_IMAGE}`",
        f"- task: `bash {WORKLOAD_CFG['script']}`",
        f"- runner script overrides: `{WORKER_SCRIPT_EXPORTS.replace(chr(10), '; ') or 'none'}`",
        f"- runner extra worker exports: `{WORKER_EXTRA_EXPORTS.replace(chr(10), '; ') or 'none'}`",
        f"- qwen script source dir: `{QWEN_SCRIPT_SOURCE_DIR or str(ROOT / 'Megatron-LM' / 'examples' / 'qwen')}`",
        f"- model overrides: `{MODEL_EXPORTS.replace(chr(10), '; ')}`",
        "- worker env policy: BytePS default attempts do not export `USE_DPU*`, `DMLC_ENABLE_UCX`, UCX rail variables, or BytePS tuning variables to workers; worker scripts use their defaults.",
        f"- UCX rail retry variants: `{','.join(UCX_RAIL_VARIANTS)}`; only non-default retries export `UCX_RAIL_MODE` to workers.",
        f"- DMLC_NUM_PORTS: `{DMLC_NUM_PORTS or 'unset'}`",
        "- worker env exceptions: runner exports topology/runtime values (`DMLC_*` root/port/count, `DMLC_NODE_HOST`, `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`); NCCL case also exports `USE_DPU*=0` to select baseline.",
        "- worker CUDA policy: not exported by runner unless `CUDA_VISIBLE_DEVICES_OVERRIDE` is set; scripts default to `CUDA_VISIBLE_DEVICES=1`",
        "- topology: scheduler gpu01, workers gpu01,gpu02,gpu03,gpu04,asus01,asus02,asus03,asus04",
        f"- network mode: `{NETWORK_MODE}`",
        f"- scheduler/server BytePS env: fused push/pull={FUSED_PUSH_PULL}, partition {BYTEPS_PARTITION_BYTES}, address pool {BYTEPS_ADDRESS_POOL_SIZE}, rx depth {BYTEPS_RDMA_RX_DEPTH}, start depth {BYTEPS_RDMA_START_DEPTH}",
        f"- BYTEPS_SERVER_ENABLE_SCHEDULE: `{SERVER_ENABLE_SCHEDULE or 'runner-default-dpu-0'}`",
        f"- max retries per case: `{MAX_RETRIES}`",
        "- worker id policy: no external WORKER_ID; script sets `DMLC_WORKER_ID=$NODE_RANK`",
        "",
        "## Results",
        "",
        "| case | attempt | rail | status | mean ms/iter 2-20 | std ms | n | train valid | train test |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        mean = "NA" if math.isnan(r["mean"]) else f"{r['mean']:.3f}"
        std = "NA" if math.isnan(r["std"]) else f"{r['std']:.3f}"
        train_valid = r["eval_losses"].get("validation", math.nan)
        train_test = r["eval_losses"].get("test", math.nan)
        fmt = lambda v: "NA" if math.isnan(v) else f"{v:.6g}"
        md.append(f"| {r['case']} | {r.get('attempt', 1)} | {r.get('rail_variant', '')} | {r['status']} | {mean} | {std} | {r['n']} | {fmt(train_valid)} | {fmt(train_test)} |")
    md += [
        "",
        "## Loss Check",
        "",
        f"- common loss points: {len(common)}",
        f"- max abs loss diff: {'NA' if math.isnan(max_abs) else f'{max_abs:.12g}'}",
        f"- loss compare TSV: `{LOGDIR / 'loss_compare.tsv'}`",
        "",
        "## Files",
        "",
        f"- summary TSV: `{OUT / 'summary.tsv'}`",
        f"- statuses TSV: `{OUT / 'statuses.tsv'}`",
        f"- logs: `{LOGDIR}`",
    ]
    write_text(OUT / "summary.md", "\n".join(md) + "\n")
    print(f"OUT={OUT}")


if __name__ == "__main__":
    main()
