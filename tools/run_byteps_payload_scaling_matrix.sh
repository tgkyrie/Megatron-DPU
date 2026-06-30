#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/home/cmcc/CMCC/Megatron-DPU"
OUT_ROOT="$BASE_DIR/docs/benchmark_runs"
BATCH="$(date +%Y%m%d-%H%M%S)-byteps-payload-scaling-matrix"
OUT_DIR="$OUT_ROOT/$BATCH"
mkdir -p "$OUT_DIR/logs"

SUMMARY="$OUT_DIR/summary.md"
TSV="$OUT_DIR/results.tsv"

all_workers=(gpu01 gpu02 gpu03 gpu04 asus01 asus02 asus03 asus04)
all_servers=(R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)

scales=(${SCALES:-2 4 8})
systems=(${SYSTEMS:-megascale native})
sizes=(${SIZES:-1})

megascale_container="${MEGASCALE_CONTAINER:-byteps-latest}"
native_container="${NATIVE_CONTAINER:-byteps-native}"

root_uri="${DMLC_PS_ROOT_URI:-192.168.1.10}"
root_port="${DMLC_PS_ROOT_PORT:-9010}"
timeout_s="${TIMEOUT_S:-600}"
byteps_extra_env="${BYTEPS_EXTRA_ENV:-}"

remote_exec() {
  local host="$1" container="$2" cmd="$3"
  ssh "$host" "sudo docker exec $container bash -lc $(printf '%q' "$cmd")"
}

remote_exec_d() {
  local host="$1" container="$2" cmd="$3"
  ssh "$host" "sudo docker exec -d $container bash -lc $(printf '%q' "$cmd")"
}

join_by_space() {
  local IFS=' '
  echo "$*"
}

select_hosts() {
  local scale="$1"
  workers=("${all_workers[@]:0:scale}")
  servers=("${all_servers[@]:0:scale}")
}

topology_env() {
  local scale="$1"
  cat <<EOF
export DMLC_PS_ROOT_URI=$root_uri
export DMLC_PS_ROOT_PORT=$root_port
export DMLC_NUM_WORKER=$scale
export DMLC_NUM_SERVER=$scale
export DMLC_ENABLE_UCX=0
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_USE_GDR=0
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

clean_byteps() {
  local container="$1"
  for h in "${workers[@]}"; do
    remote_exec "$h" "$container" 'pids=$(pgrep -f "[p]ushpull_bench|[b]pslaunch|[p]ayload-scaling" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  remote_exec gpu01 "$container" 'pids=$(pgrep -f "[s]cheduler.sh|[p]ayload-scaling" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  for h in "${servers[@]}"; do
    remote_exec "$h" "$container" 'pids=$(pgrep -f "[s]erver.sh|[p]ushpull_bench|[b]pslaunch|[p]ayload-scaling" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  sleep 3
}

worker_statuses() {
  local run="$1" container="$2" out="" one
  for h in "${workers[@]}"; do
    one=$(remote_exec "$h" "$container" "cat /tmp/${run}-worker.status 2>/dev/null || echo missing" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]*$//')
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

logs_have_result() {
  local run="$1" container="$2"
  remote_exec "${workers[0]}" "$container" "grep -q 'agg_GB/s=' /tmp/${run}-worker.log 2>/dev/null" >/dev/null 2>&1
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

wait_workers() {
  local run="$1" container="$2" scale="$3" label="$4" size="$5"
  local start_ts now elapsed s
  start_ts=$(date +%s)
  while true; do
    sleep 20
    s=$(worker_statuses "$run" "$container")
    now=$(date +%s)
    elapsed=$((now - start_ts))
    echo "[$(date '+%H:%M:%S')] system=${label} scale=${scale} size=${size}MB statuses=$s"
    if all_done "$s"; then
      break
    fi
    if logs_have_result "$run" "$container"; then
      echo "[$(date '+%H:%M:%S')] system=${label} scale=${scale} size=${size}MB payload results detected in rank0 log"
      break
    fi
    if workers_have_failure "$s"; then
      break
    fi
    if [[ $elapsed -gt $timeout_s ]]; then
      echo "timeout system=${label} scale=${scale} size=${size}MB"
      break
    fi
  done
}

run_payload() {
  local label="$1" scale="$2" container="$3" size="$4"
  local size_label="${size//./p}" envs role_extra="" worker_extra=""
  local run="payload-scaling-${label}-${scale}w${scale}s-${size_label}mb-$BATCH"
  envs="$(topology_env "$scale")"
  if [[ -n "$byteps_extra_env" ]]; then
    role_extra="$byteps_extra_env"
    worker_extra="$byteps_extra_env"
  fi
  clean_byteps "$container"
  echo "[$(date '+%H:%M:%S')] START ${label} scale=${scale} size=${size}MB workers=$(join_by_space "${workers[@]}") servers=$(join_by_space "${servers[@]}")"
  remote_exec_d gpu01 "$container" "echo running > /tmp/${run}-scheduler.status
$envs
$role_extra
cd /tmp
bash /usr/local/byteps/sh/scheduler.sh > /tmp/${run}-scheduler.log 2>&1
echo rc=\$? > /tmp/${run}-scheduler.status" >/dev/null || return 1
  sleep 2
  for h in "${servers[@]}"; do
    remote_exec_d "$h" "$container" "echo running > /tmp/${run}-server.status
$envs
$role_extra
cd /tmp
bash /usr/local/byteps/sh/server.sh > /tmp/${run}-server.log 2>&1
echo rc=\$? > /tmp/${run}-server.status" >/dev/null || return 1
  done
  sleep 6
  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    local node_host
    node_host="$(worker_ip "$h")"
    remote_exec_d "$h" "$container" "echo running > /tmp/${run}-worker.status
$envs
$worker_extra
export DMLC_ROLE=worker
export DMLC_NODE_HOST=$node_host
export DMLC_WORKER_ID=$idx
export BYTEPS_LOCAL_SIZE=1
export BYTEPS_LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=1
export BYTEPS_PARTITION_BYTES=\${BYTEPS_PARTITION_BYTES:-4194304}
export BYTEPS_RDMA_RX_DEPTH=\${BYTEPS_RDMA_RX_DEPTH:-512}
export BYTEPS_RDMA_START_DEPTH=\${BYTEPS_RDMA_START_DEPTH:-32}
cd /tmp
timeout ${timeout_s}s bpslaunch python3 /usr/local/byteps/example/pytorch/pushpull_bench.py --size-mb $size --iters 40 --warmup 5 > /tmp/${run}-worker.log 2>&1
echo rc=\$? > /tmp/${run}-worker.status" >/dev/null || return 1
  done
  wait_workers "$run" "$container" "$scale" "$label" "$size"
  for h in "${workers[@]}"; do
    remote_exec "$h" "$container" "cat /tmp/${run}-worker.log 2>/dev/null || true" > "$OUT_DIR/logs/${run}-${h}.log" 2>/dev/null || true
  done
  local status rank0_log parsed
  status="$(worker_statuses "$run" "$container")"
  clean_byteps "$container"
  rank0_log="$OUT_DIR/logs/${run}-gpu01.log"
  parsed="$(parse_rank0 "$rank0_log")"
  echo -e "$label\t$scale\t$scale\t$size\t$container\t$status\t$parsed\t$rank0_log" >> "$TSV"
}

echo -e "system\tworkers\tservers\tsize_mb\tcontainer\tstatus\tn_iters\tmean_time_ms\tstd_time_ms\tmean_agg_GBps\tstd_agg_GBps\tmean_bus_GBps\tstd_bus_GBps\trank0_log" > "$TSV"

for scale in "${scales[@]}"; do
  select_hosts "$scale"
  for size in "${sizes[@]}"; do
    for sys in "${systems[@]}"; do
      case "$sys" in
        megascale) run_payload "MegaScale-PS" "$scale" "$megascale_container" "$size" ;;
        native) run_payload "Native-BytePS" "$scale" "$native_container" "$size" ;;
        *) echo "unknown system: $sys" >&2; exit 2 ;;
      esac
    done
  done
done

{
  echo "# BytePS payload scaling matrix"
  echo
  echo "Run: $BATCH"
  echo "Date: $(date -Is)"
  echo
  echo "This is a push-pull communication microbenchmark. BytePS-style systems use equal worker and server counts on disjoint machines."
  echo
  echo "| System | Workers | Servers | Size MB | Agg GB/s | Time ms | Container |"
  echo "| --- | ---: | ---: | ---: | ---: | ---: | --- |"
  tail -n +2 "$TSV" | awk -F'\t' '{printf "| %s | %s | %s | %s | %s ± %s | %s ± %s | %s |\n",$1,$2,$3,$4,$10,$11,$8,$9,$5}'
  echo
  echo "Raw TSV: $TSV"
} > "$SUMMARY"

echo "OUT_DIR=$OUT_DIR"
echo "SUMMARY=$SUMMARY"
echo "TSV=$TSV"
