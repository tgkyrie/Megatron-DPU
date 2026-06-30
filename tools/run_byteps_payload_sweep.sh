#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/home/cmcc/CMCC/Megatron-DPU"
OUT_ROOT="$BASE_DIR/docs/benchmark_runs"
BATCH="$(date +%Y%m%d-%H%M%S)-byteps-payload-sweep-8w8s"
OUT_DIR="$OUT_ROOT/$BATCH"
mkdir -p "$OUT_DIR/logs"

SUMMARY="$OUT_DIR/summary.md"
TSV="$OUT_DIR/results.tsv"

workers=(gpu01 gpu02 gpu03 gpu04 asus01 asus02 asus03 asus04)
servers=(R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)
byteps_nodes=(gpu01 R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)
container="${BYTEPS_CONTAINER:-byteps-latest}"
root_uri="192.168.1.10"
root_port="9010"
sizes=(0.25 0.5 1 2 4 8 16)
payload_timeout="${PAYLOAD_TIMEOUT:-120}"

remote_exec() {
  local host="$1" cmd="$2"
  ssh "$host" "sudo docker exec $container bash -lc $(printf '%q' "$cmd")"
}

remote_exec_d() {
  local host="$1" cmd="$2"
  ssh "$host" "sudo docker exec -d $container bash -lc $(printf '%q' "$cmd")"
}

topology_env() {
  cat <<EOF
export DMLC_PS_ROOT_URI=$root_uri
export DMLC_PS_ROOT_PORT=$root_port
export DMLC_NUM_WORKER=8
export DMLC_NUM_SERVER=8
export DMLC_ENABLE_UCX=0
export DMLC_ENABLE_RDMA=ibverbs
EOF
}

worker_ip() {
  case "$1" in
    gpu01) echo 192.168.1.10 ;;
    gpu02) echo 192.168.1.11 ;;
    gpu03) echo 192.168.1.12 ;;
    gpu04) echo 192.168.1.13 ;;
    asus01) echo 192.168.1.20 ;;
    asus02) echo 192.168.1.21 ;;
    asus03) echo 192.168.1.22 ;;
    asus04) echo 192.168.1.23 ;;
    *) return 1 ;;
  esac
}

clean_all() {
  for h in "${workers[@]}"; do
    remote_exec "$h" 'pids=$(pgrep -f "[p]ushpull_bench|[b]pslaunch|[b]yteps-payload" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  for h in "${byteps_nodes[@]}"; do
    remote_exec "$h" 'pids=$(pgrep -f "[s]cheduler.sh|[s]erver.sh|[p]ushpull_bench|[b]pslaunch|[b]enchmark_byteps" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  sleep 3
}

start_byteps() {
  local run="$1" envs
  envs="$(topology_env)"
  remote_exec_d gpu01 "echo running > /tmp/${run}-scheduler.status
$envs
cd /usr/local
bash /usr/local/byteps/sh/scheduler.sh > /tmp/${run}-scheduler.log 2>&1
echo rc=\$? > /tmp/${run}-scheduler.status" >/dev/null || return 1
  sleep 2
  for h in "${servers[@]}"; do
    remote_exec_d "$h" "echo running > /tmp/${run}-server.status
$envs
cd /usr/local
bash /usr/local/byteps/sh/server.sh > /tmp/${run}-server.log 2>&1
echo rc=\$? > /tmp/${run}-server.status" >/dev/null || return 1
  done
  sleep 6
}

start_workers() {
  local run="$1" size="$2" envs
  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    local node_host
    node_host="$(worker_ip "$h")"
    envs="$(topology_env)"
    remote_exec_d "$h" "echo running > /tmp/${run}-worker.status
$envs
export DMLC_ROLE=worker
export DMLC_NODE_HOST=$node_host
export DMLC_WORKER_ID=$idx
export BYTEPS_LOCAL_SIZE=1
export BYTEPS_LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=1
export BYTEPS_PARTITION_BYTES=4194304
export BYTEPS_RDMA_RX_DEPTH=512
export BYTEPS_RDMA_START_DEPTH=32
cd /usr/local/byteps
timeout ${payload_timeout}s bpslaunch python3 /usr/local/byteps/example/pytorch/pushpull_bench.py --size-mb $size --iters 40 --warmup 5 > /tmp/${run}-worker.log 2>&1
echo rc=\$? > /tmp/${run}-worker.status" >/dev/null || return 1
  done
}

statuses() {
  local run="$1" out="" one
  for h in "${workers[@]}"; do
    one=$(remote_exec "$h" "cat /tmp/${run}-worker.status 2>/dev/null || echo missing" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]*$//')
    out+="$h:$one "
  done
  echo "$out"
}

all_done() {
  local s="$1"
  [[ "$s" != *running* && "$s" != *missing* ]]
}

workers_have_failure() {
  local s="$1"
  [[ "$s" =~ rc=([1-9][0-9]*) ]]
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
import re, statistics, sys
text = open(sys.argv[1], errors="ignore").read()
rows = []
for m in re.finditer(r"time_ms=([0-9.]+).*?agg_GB/s=([0-9.]+).*?bus_GB/s=([0-9.]+)", text):
    rows.append(tuple(float(x) for x in m.groups()))
if not rows:
    print("0\tNA\tNA\tNA\tNA\tNA\tNA")
else:
    def mean(i): return statistics.mean(r[i] for r in rows)
    def pstdev(i): return statistics.pstdev(r[i] for r in rows) if len(rows) > 1 else 0.0
    print(f"{len(rows)}\t{mean(0):.3f}\t{pstdev(0):.3f}\t{mean(1):.3f}\t{pstdev(1):.3f}\t{mean(2):.3f}\t{pstdev(2):.3f}")
PY
}

run_one() {
  local size="$1" label
  label="${size//./p}"
  local run="byteps-payload-${label}mb-$BATCH"
  clean_all
  echo "[$(date '+%H:%M:%S')] START payload size=${size}MB"
  start_byteps "$run" || return 1
  start_workers "$run" "$size" || return 1
  local start_ts now elapsed s
  start_ts=$(date +%s)
  while true; do
    sleep 20
    s=$(statuses "$run")
    now=$(date +%s)
    elapsed=$((now - start_ts))
    echo "[$(date '+%H:%M:%S')] size=${size}MB statuses=$s"
    if all_done "$s"; then
      break
    fi
    if workers_have_failure "$s"; then
      break
    fi
    if [[ $elapsed -gt $payload_timeout ]]; then
      echo "timeout size=${size}MB"
      break
    fi
  done
  collect_logs "$run"
  clean_all
  local log="$OUT_DIR/logs/${run}-gpu01.log" parsed
  parsed="$(parse_rank0 "$log")"
  echo -e "${size}\t${parsed}\t${log}\t$(statuses "$run")" >> "$TSV"
}

echo -e "size_mb\tn_iters\tmean_time_ms\tstd_time_ms\tmean_agg_GBps\tstd_agg_GBps\tmean_bus_GBps\tstd_bus_GBps\trank0_log\tworker_statuses" > "$TSV"
trap 'clean_all >/dev/null 2>&1 || true' EXIT

for s in "${sizes[@]}"; do
  run_one "$s"
done

{
  echo "# BytePS payload sweep"
  echo
  echo "Batch: $BATCH"
  echo "Date: $(date -Is)"
  echo
  echo "This is a BytePS push-pull communication microbenchmark, not a Megatron end-to-end training measurement."
  echo
  echo "- Scheduler: gpu01"
  echo "- Workers: ${workers[*]}"
  echo "- Servers: ${servers[*]}"
  echo "- Worker command: bpslaunch python3 /usr/local/byteps/example/pytorch/pushpull_bench.py --iters 40 --warmup 5"
  echo "- Runner sets DMLC_NODE_HOST per worker so the sweep is independent of per-host interface names"
  echo "- Per-size timeout: ${payload_timeout}s"
  echo "- Payload range: 256KB to 16MB"
  echo
  echo "| Size MB | n | Mean time ms | Agg GB/s | Bus GB/s |"
  echo "| ---: | ---: | ---: | ---: | ---: |"
  tail -n +2 "$TSV" | awk -F'\t' '{printf "| %s | %s | %s | %s | %s |\n",$1,$2,$3,$5,$7}'
  echo
  echo "Raw TSV: $TSV"
} > "$SUMMARY"

echo "OUT_DIR=$OUT_DIR"
echo "SUMMARY=$SUMMARY"
echo "TSV=$TSV"
