#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/home/cmcc/CMCC/Megatron-DPU"
OUT_ROOT="$BASE_DIR/docs/benchmark_runs"
BATCH="$(date +%Y%m%d-%H%M%S)-vgg16-byteps-8w8s"
OUT_DIR="$OUT_ROOT/$BATCH"
mkdir -p "$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.md"
TSV="$OUT_DIR/results.tsv"

workers=(gpu01 gpu02 gpu03 gpu04 asus01 asus02 asus03 asus04)
servers=(R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)
byteps_nodes=(gpu01 R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)
container="${BYTEPS_CONTAINER:-byteps-latest}"

remote_exec() {
  local host="$1" cmd="$2"
  ssh "$host" "sudo docker exec $container bash -lc $(printf '%q' "$cmd")"
}

remote_exec_d() {
  local host="$1" cmd="$2"
  ssh "$host" "sudo docker exec -d $container bash -lc $(printf '%q' "$cmd")"
}

topology_env() {
  cat <<'EOF'
export DMLC_NUM_WORKER=8
export DMLC_NUM_SERVER=8
EOF
}

clean_all() {
  for h in "${workers[@]}"; do
    remote_exec "$h" 'pids=$(pgrep -f "[w]orker.sh|[b]enchmark_byteps|[b]pslaunch|[r]un-vgg16-byteps" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  for h in "${byteps_nodes[@]}"; do
    remote_exec "$h" 'pids=$(pgrep -f "[s]cheduler.sh|[s]erver.sh|[b]enchmark_byteps|[b]pslaunch|[r]un-vgg16-byteps" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  sleep 3
}

start_byteps_roles() {
  local run="$1" envs
  envs="$(topology_env)"

  remote_exec_d gpu01 "echo running > /tmp/${run}-scheduler.status
$envs
cd /tmp
bash /usr/local/byteps/sh/scheduler.sh > /tmp/${run}-scheduler.log 2>&1
echo rc=\$? > /tmp/${run}-scheduler.status" >/dev/null || return 1
  sleep 2

  for h in "${servers[@]}"; do
    remote_exec_d "$h" "echo running > /tmp/${run}-server.status
$envs
cd /tmp
bash /usr/local/byteps/sh/server.sh > /tmp/${run}-server.log 2>&1
echo rc=\$? > /tmp/${run}-server.status" >/dev/null || return 1
  done
  sleep 6
}

start_workers() {
  local run="$1" envs
  envs="$(topology_env)"
  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    remote_exec_d "$h" "echo running > /tmp/${run}-worker.status
$envs
export WORKER_ID=$idx
cd /tmp
timeout 900s bash /usr/local/byteps/sh/worker.sh > /tmp/${run}-worker.log 2>&1
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

collect_logs() {
  local run="$1"
  remote_exec gpu01 "cat /tmp/${run}-scheduler.log 2>/dev/null || true" > "$OUT_DIR/logs/${run}-scheduler.log" 2>/dev/null || true
  for h in "${servers[@]}"; do
    remote_exec "$h" "cat /tmp/${run}-server.log 2>/dev/null || true" > "$OUT_DIR/logs/${run}-${h}-server.log" 2>/dev/null || true
  done
  for h in "${workers[@]}"; do
    remote_exec "$h" "cat /tmp/${run}-worker.log 2>/dev/null || true" > "$OUT_DIR/logs/${run}-${h}-worker.log" 2>/dev/null || true
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

run="run-vgg16-byteps-8w8s-$BATCH"
clean_all
echo "[$(date '+%H:%M:%S')] START BytePS VGG16 run=$run"
start_byteps_roles "$run" || {
  echo "start_byteps_roles failed"
  exit 1
}
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
final_status="$(statuses "$run")"
clean_all

rank0_log="$OUT_DIR/logs/${run}-gpu01-worker.log"
parsed="$(parse_rank0 "$rank0_log")"
per_mean="$(echo "$parsed" | awk '{print $1}')"
per_conf="$(echo "$parsed" | awk '{print $2}')"
total_mean="$(echo "$parsed" | awk '{print $3}')"
total_conf="$(echo "$parsed" | awk '{print $4}')"

echo -e "system\tworkers\tservers\tmodel\tstatus\timg_sec_per_gpu\timg_sec_per_gpu_conf\ttotal_img_sec\ttotal_img_sec_conf\trank0_log\tworker_statuses" > "$TSV"
echo -e "BytePS\t8\t8\tvgg16\t$final_status\t$per_mean\t$per_conf\t$total_mean\t$total_conf\t$rank0_log\t$final_status" >> "$TSV"

{
  echo "# VGG16 BytePS 8-worker/8-server benchmark"
  echo
  echo "Run: $BATCH"
  echo "Date: $(date -Is)"
  echo
  echo "## Setup"
  echo
  echo "- Scheduler: gpu01"
  echo "- Workers: ${workers[*]}"
  echo "- Servers: ${servers[*]}"
  echo "- Container: $container"
  echo "- Role scripts: /usr/local/byteps/sh/scheduler.sh, server.sh, worker.sh"
  echo "- Runner exports: DMLC_NUM_WORKER, DMLC_NUM_SERVER, WORKER_ID only"
  echo "- Benchmark defaults come from worker.sh: benchmark_byteps.py --model vgg16 --num-iters 10"
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
