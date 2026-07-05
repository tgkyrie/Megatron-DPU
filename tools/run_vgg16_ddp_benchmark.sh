#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/home/cmcc/CMCC/Megatron-DPU"
OUT_ROOT="$BASE_DIR/docs/benchmark_runs"
BATCH="$(date +%Y%m%d-%H%M%S)-vgg16-ddp-8w"
OUT_DIR="$OUT_ROOT/$BATCH"
mkdir -p "$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.md"
TSV="$OUT_DIR/results.tsv"

workers=(gpu01 gpu02 gpu03 gpu04 asus01 asus02 asus03 asus04)
container="${MEGASCALE_PS_CONTAINER:-megascale_ps-latest}"
master_addr="192.168.1.10"
master_port="${MASTER_PORT:-29610}"

remote_exec() {
  local host="$1" cmd="$2"
  ssh "$host" "sudo docker exec $container bash -lc $(printf '%q' "$cmd")"
}

remote_exec_d() {
  local host="$1" cmd="$2"
  ssh "$host" "sudo docker exec -d $container bash -lc $(printf '%q' "$cmd")"
}

clean_all() {
  for h in "${workers[@]}"; do
    remote_exec "$h" 'pids=$(pgrep -f "[t]orch_ddp_benchmark|[r]un-vgg16-ddp" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  sleep 3
}

worker_ifname() {
  case "$1" in
    gpu01|gpu02|gpu03|gpu04) echo ens39f1np1 ;;
    asus01|asus02|asus03|asus04) echo ens93f1np1 ;;
    *) return 1 ;;
  esac
}

start_workers() {
  local run="$1"
  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    local ifname
    ifname="$(worker_ifname "$h")"
    local cmd="echo running > /tmp/${run}-worker.status
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=8
export RANK=$idx
export CUDA_VISIBLE_DEVICES=1
export NCCL_SOCKET_IFNAME=$ifname
export NCCL_IB_HCA=mlx5_1
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TC=${NCCL_IB_TC:-106}
cd /usr/local/megascale_ps
timeout 900s bash /usr/local/megascale_ps/sh/worker_ddp.sh --no-comm-log > /tmp/${run}-worker.log 2>&1
echo rc=\$? > /tmp/${run}-worker.status"
    remote_exec_d "$h" "$cmd" >/dev/null || return 1
  done
}

statuses() {
  local run="$1"
  local out=""
  for h in "${workers[@]}"; do
    local one
    one=$(remote_exec "$h" "cat /tmp/${run}-worker.status 2>/dev/null || echo missing" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]*$//')
    out+="$h:$one "
  done
  echo "$out"
}

all_done() {
  local s="$1"
  [[ "$s" != *running* && "$s" != *missing* ]]
}

collect_logs() {
  local run="$1"
  for h in "${workers[@]}"; do
    remote_exec "$h" "cat /tmp/${run}-worker.log 2>/dev/null || true" > "$OUT_DIR/logs/${run}-${h}.log" 2>/dev/null || true
  done
}

parse_rank0() {
  local log="$1"
  python3 - "$log" <<'PY'
import re, sys
text = open(sys.argv[1], errors="ignore").read()
per = re.search(r"Img/sec per GPU:\s*([0-9.]+)\s*\+-\s*([0-9.]+)", text)
tot = re.search(r"Total img/sec on\s+8\s+GPU\(s\):\s*([0-9.]+)\s*\+-\s*([0-9.]+)", text)
if not per or not tot:
    print("NA\tNA\tNA\tNA")
else:
    print(f"{per.group(1)}\t{per.group(2)}\t{tot.group(1)}\t{tot.group(2)}")
PY
}

run="run-vgg16-ddp-8w-$BATCH"
clean_all
echo "[$(date '+%H:%M:%S')] START DDP VGG16 run=$run"
start_workers "$run" || {
  echo "start_workers failed"
  exit 1
}

start_ts=$(date +%s)
while true; do
  sleep 20
  s=$(statuses "$run")
  now=$(date +%s)
  elapsed=$((now - start_ts))
  echo "[$(date '+%H:%M:%S')] statuses=$s"
  if all_done "$s"; then
    break
  fi
  if [[ $elapsed -gt 900 ]]; then
    echo "timeout"
    break
  fi
done

collect_logs "$run"
clean_all

rank0_log="$OUT_DIR/logs/${run}-gpu01.log"
parsed="$(parse_rank0 "$rank0_log")"
per_mean="$(echo "$parsed" | awk '{print $1}')"
per_conf="$(echo "$parsed" | awk '{print $2}')"
total_mean="$(echo "$parsed" | awk '{print $3}')"
total_conf="$(echo "$parsed" | awk '{print $4}')"
final_status="$(statuses "$run")"

echo -e "system\tworkers\tmodel\tstatus\timg_sec_per_gpu\timg_sec_per_gpu_conf\ttotal_img_sec\ttotal_img_sec_conf\trank0_log\tworker_statuses" > "$TSV"
echo -e "DDP\t8\tvgg16\t$final_status\t$per_mean\t$per_conf\t$total_mean\t$total_conf\t$rank0_log\t$final_status" >> "$TSV"

{
  echo "# VGG16 DDP 8-worker benchmark"
  echo
  echo "Run: $BATCH"
  echo "Date: $(date -Is)"
  echo
  echo "## Setup"
  echo
  echo "- Workers: ${workers[*]}"
  echo "- Container: $container"
  echo "- Worker script: /usr/local/megascale_ps/sh/worker_ddp.sh --no-comm-log"
  echo "- Benchmark defaults come from worker_ddp.sh: torch_ddp_benchmark.py --model vgg16 --num-iters 10"
  echo "- Runner exports: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK only"
  echo "- Runner also sets NCCL_SOCKET_IFNAME per worker host so gpu*/asus* use their actual RoCE interface"
  echo "- Runner sets NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} so NCCL uses the 192.168.1.x RoCE GID instead of link-local GID 0"
  echo "- Runner sets NCCL_IB_TC=${NCCL_IB_TC:-106}"
  echo "- Runner uses CUDA_VISIBLE_DEVICES=1 to match the current cluster training default"
  echo "- Master: $master_addr:$master_port"
  echo
  echo "## Result"
  echo
  echo "| Metric | Result |"
  echo "| --- | ---: |"
  echo "| Worker status | $final_status |"
  echo "| Img/sec per GPU | $per_mean +- $per_conf |"
  echo "| Total img/sec on 8 GPUs | $total_mean +- $total_conf |"
  echo
  echo "Raw TSV: $TSV"
  echo "Rank0 log: $rank0_log"
} > "$SUMMARY"

echo "OUT_DIR=$OUT_DIR"
echo "SUMMARY=$SUMMARY"
echo "TSV=$TSV"
