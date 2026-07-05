#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/home/cmcc/CMCC/Megatron-DPU"
OUT_ROOT="$BASE_DIR/docs/benchmark_runs"
BATCH="$(date +%Y%m%d-%H%M%S)-vgg16-scaling-matrix"
OUT_DIR="$OUT_ROOT/$BATCH"
mkdir -p "$OUT_DIR/logs"

SUMMARY="$OUT_DIR/summary.md"
TSV="$OUT_DIR/results.tsv"

all_workers=(gpu01 gpu02 gpu03 gpu04 asus01 asus02 asus03 asus04)
all_servers=(R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)

scales=(${SCALES:-2 4 8})
systems=(${SYSTEMS:-ddp megascale_ps native})

ddp_container="${DDP_CONTAINER:-megascale_ps-latest}"
megascale_ps_container="${MEGASCALE_PS_CONTAINER:-megascale_ps-latest}"
native_container="${NATIVE_CONTAINER:-megascale_ps-native}"

root_uri="${DMLC_PS_ROOT_URI:-192.168.1.10}"
root_port="${DMLC_PS_ROOT_PORT:-9010}"
master_addr="${MASTER_ADDR:-192.168.1.10}"
master_port_base="${MASTER_PORT_BASE:-29610}"
timeout_s="${TIMEOUT_S:-900}"
megascale_ps_extra_env="${MEGASCALE_PS_EXTRA_ENV:-}"

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
EOF
}

worker_ifname() {
  case "$1" in
    gpu01|gpu02|gpu03|gpu04) echo ens39f1np1 ;;
    asus01|asus02|asus03|asus04) echo ens93f1np1 ;;
    *) return 1 ;;
  esac
}

clean_workers() {
  local container="$1" pattern="$2"
  for h in "${workers[@]}"; do
    remote_exec "$h" "$container" "pids=\$(pgrep -f \"$pattern\" || true); if [ -n \"\$pids\" ]; then kill -TERM \$pids 2>/dev/null || true; fi" >/dev/null 2>&1 || true
  done
}

clean_megascale_ps() {
  local container="$1"
  clean_workers "$container" "[w]orker.sh|[b]enchmark_megascale_ps|[b]pslaunch|[v]gg16-scaling"
  remote_exec gpu01 "$container" 'pids=$(pgrep -f "[s]cheduler.sh|[v]gg16-scaling" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  for h in "${servers[@]}"; do
    remote_exec "$h" "$container" 'pids=$(pgrep -f "[s]erver.sh|[b]enchmark_megascale_ps|[b]pslaunch|[v]gg16-scaling" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
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

logs_have_result() {
  local run="$1" container="$2"
  remote_exec "${workers[0]}" "$container" "grep -q 'Total img/sec on' /tmp/${run}-worker.log 2>/dev/null" >/dev/null 2>&1
}

parse_rank0() {
  local log="$1" scale="$2"
  python3 - "$log" "$scale" <<'PY'
import re, sys
text = open(sys.argv[1], errors="ignore").read()
scale = sys.argv[2]
per = re.search(r"Img/sec per GPU:\s*([0-9.]+)\s*\+-\s*([0-9.]+)", text)
tot = re.search(r"Total img/sec on\s+\d+\s+GPU\(s\):\s*([0-9.]+)\s*\+-\s*([0-9.]+)", text)
if not per or not tot:
    print("NA\tNA\tNA\tNA")
else:
    print(f"{per.group(1)}\t{per.group(2)}\t{tot.group(1)}\t{tot.group(2)}")
PY
}

wait_workers() {
  local run="$1" container="$2" scale="$3" system="$4"
  local start_ts now elapsed s
  start_ts=$(date +%s)
  while true; do
    sleep 20
    s=$(worker_statuses "$run" "$container")
    now=$(date +%s)
    elapsed=$((now - start_ts))
    echo "[$(date '+%H:%M:%S')] system=${system} scale=${scale} statuses=$s"
    if all_done "$s"; then
      break
    fi
    if logs_have_result "$run" "$container"; then
      echo "[$(date '+%H:%M:%S')] system=${system} scale=${scale} final throughput detected in all worker logs"
      break
    fi
    if [[ $elapsed -gt $timeout_s ]]; then
      echo "timeout system=${system} scale=${scale}"
      break
    fi
  done
}

run_ddp() {
  local scale="$1" container="$2" run="vgg16-scaling-ddp-${scale}w-$BATCH"
  local port=$((master_port_base + scale))
  clean_workers "$container" "[t]orch_ddp_benchmark|[v]gg16-scaling-ddp"
  echo "[$(date '+%H:%M:%S')] START DDP scale=${scale} workers=$(join_by_space "${workers[@]}")"
  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    local ifname
    ifname="$(worker_ifname "$h")"
    remote_exec_d "$h" "$container" "echo running > /tmp/${run}-worker.status
export MASTER_ADDR=$master_addr
export MASTER_PORT=$port
export WORLD_SIZE=$scale
export RANK=$idx
export CUDA_VISIBLE_DEVICES=1
export NCCL_SOCKET_IFNAME=$ifname
export NCCL_IB_HCA=mlx5_1
export NCCL_IB_GID_INDEX=\${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TC=\${NCCL_IB_TC:-106}
cd /tmp
timeout ${timeout_s}s bash /usr/local/megascale_ps/sh/worker_ddp.sh --no-comm-log > /tmp/${run}-worker.log 2>&1
echo rc=\$? > /tmp/${run}-worker.status" >/dev/null || return 1
  done
  wait_workers "$run" "$container" "$scale" "DDP"
  for h in "${workers[@]}"; do
    remote_exec "$h" "$container" "cat /tmp/${run}-worker.log 2>/dev/null || true" > "$OUT_DIR/logs/${run}-${h}.log" 2>/dev/null || true
  done
  local status rank0_log parsed
  status="$(worker_statuses "$run" "$container")"
  clean_workers "$container" "[t]orch_ddp_benchmark|[v]gg16-scaling-ddp"
  rank0_log="$OUT_DIR/logs/${run}-gpu01.log"
  parsed="$(parse_rank0 "$rank0_log" "$scale")"
  echo -e "DDP\t$scale\t0\tvgg16\t$container\t$status\t$parsed\t$rank0_log" >> "$TSV"
}

run_megascale_ps() {
  local label="$1" scale="$2" container="$3"
  local run="vgg16-scaling-${label}-${scale}w${scale}s-$BATCH"
  local envs role_extra="" worker_extra=""
  envs="$(topology_env "$scale")"
  if [[ "$label" == "Native-MegaScalePS" ]]; then
    role_extra="$megascale_ps_extra_env"
    worker_extra="$megascale_ps_extra_env"
  fi
  clean_megascale_ps "$container"
  echo "[$(date '+%H:%M:%S')] START ${label} scale=${scale} workers=$(join_by_space "${workers[@]}") servers=$(join_by_space "${servers[@]}")"
  remote_exec_d gpu01 "$container" "echo running > /tmp/${run}-scheduler.status
$envs
$role_extra
cd /tmp
bash /usr/local/megascale_ps/sh/scheduler.sh > /tmp/${run}-scheduler.log 2>&1
echo rc=\$? > /tmp/${run}-scheduler.status" >/dev/null || return 1
  sleep 2
  for h in "${servers[@]}"; do
    remote_exec_d "$h" "$container" "echo running > /tmp/${run}-server.status
$envs
$role_extra
cd /tmp
bash /usr/local/megascale_ps/sh/server.sh > /tmp/${run}-server.log 2>&1
echo rc=\$? > /tmp/${run}-server.status" >/dev/null || return 1
  done
  sleep 6
  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    remote_exec_d "$h" "$container" "echo running > /tmp/${run}-worker.status
$envs
$worker_extra
export WORKER_ID=$idx
cd /tmp
timeout ${timeout_s}s bash /usr/local/megascale_ps/sh/worker.sh > /tmp/${run}-worker.log 2>&1
echo rc=\$? > /tmp/${run}-worker.status" >/dev/null || return 1
  done
  wait_workers "$run" "$container" "$scale" "$label"
  remote_exec gpu01 "$container" "cat /tmp/${run}-scheduler.log 2>/dev/null || true" > "$OUT_DIR/logs/${run}-scheduler.log" 2>/dev/null || true
  for h in "${servers[@]}"; do
    remote_exec "$h" "$container" "cat /tmp/${run}-server.log 2>/dev/null || true" > "$OUT_DIR/logs/${run}-${h}-server.log" 2>/dev/null || true
  done
  for h in "${workers[@]}"; do
    remote_exec "$h" "$container" "cat /tmp/${run}-worker.log 2>/dev/null || true" > "$OUT_DIR/logs/${run}-${h}-worker.log" 2>/dev/null || true
  done
  local status rank0_log parsed
  status="$(worker_statuses "$run" "$container")"
  clean_megascale_ps "$container"
  rank0_log="$OUT_DIR/logs/${run}-gpu01-worker.log"
  parsed="$(parse_rank0 "$rank0_log" "$scale")"
  echo -e "$label\t$scale\t$scale\tvgg16\t$container\t$status\t$parsed\t$rank0_log" >> "$TSV"
}

echo -e "system\tworkers\tservers\tmodel\tcontainer\tstatus\timg_sec_per_gpu\timg_sec_per_gpu_conf\ttotal_img_sec\ttotal_img_sec_conf\trank0_log" > "$TSV"
trap 'true' EXIT

for scale in "${scales[@]}"; do
  select_hosts "$scale"
  for sys in "${systems[@]}"; do
    case "$sys" in
      ddp) run_ddp "$scale" "$ddp_container" ;;
      megascale_ps) run_megascale_ps "MegaScale-PS" "$scale" "$megascale_ps_container" ;;
      native) run_megascale_ps "Native-MegaScalePS" "$scale" "$native_container" ;;
      *) echo "unknown system: $sys" >&2; exit 2 ;;
    esac
  done
done

{
  echo "# VGG16 scaling matrix"
  echo
  echo "Run: $BATCH"
  echo "Date: $(date -Is)"
  echo
  echo "Workers are selected from: ${all_workers[*]}"
  echo "Servers are selected from: ${all_servers[*]}"
  echo "DDP uses workers only. MegaScalePS-style systems use equal worker and server counts on disjoint machines."
  echo
  echo "| System | Workers | Servers | Total img/s | Per-GPU img/s | Container |"
  echo "| --- | ---: | ---: | ---: | ---: | --- |"
  tail -n +2 "$TSV" | awk -F'\t' '{printf "| %s | %s | %s | %s ± %s | %s ± %s | %s |\n",$1,$2,$3,$9,$10,$7,$8,$5}'
  echo
  echo "Raw TSV: $TSV"
} > "$SUMMARY"

echo "OUT_DIR=$OUT_DIR"
echo "SUMMARY=$SUMMARY"
echo "TSV=$TSV"
