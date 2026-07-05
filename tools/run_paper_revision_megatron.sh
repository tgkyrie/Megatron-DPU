#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/home/cmcc/CMCC/Megatron-DPU"
OUT_ROOT="$BASE_DIR/docs/benchmark_runs"
BATCH="$(date +%Y%m%d-%H%M%S)-paper-revision-megatron"
OUT_DIR="$OUT_ROOT/$BATCH"
mkdir -p "$OUT_DIR/logs"

DETAIL="$OUT_DIR/details.md"
SUMMARY="$OUT_DIR/summary.md"
TSV="$OUT_DIR/results.tsv"

workers=(gpu01 gpu02 gpu03 gpu04 asus01 asus02 asus03 asus04)
all_servers=(R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)
megascale_ps_nodes=(gpu01 R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)

worker_container="${WORKER_CONTAINER:-megatron-dpu-latest}"
megascale_ps_container="${MEGASCALE_PS_CONTAINER:-megascale_ps-latest}"
root_uri="192.168.1.10"
root_port="9010"
port=19600

remote_exec() {
  local host="$1" container="$2" cmd="$3"
  ssh "$host" "sudo docker exec $container bash -lc $(printf '%q' "$cmd")"
}

remote_exec_d() {
  local host="$1" container="$2" cmd="$3"
  ssh "$host" "sudo docker exec -d $container bash -lc $(printf '%q' "$cmd")"
}

servers_for_count() {
  local n="$1"
  case "$n" in
    2) echo "R750-1 R750-2" ;;
    4) echo "R750-1 R750-2 R750-3 R750-4" ;;
    8) echo "R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15" ;;
    *) return 1 ;;
  esac
}

clean_all() {
  for h in "${workers[@]}"; do
    remote_exec "$h" "$worker_container" 'pids=$(pgrep -f "[t]orchrun|[p]retrain_gpt|[t]rain_.*megascale_ps|[t]rain_qwen|[t]rain_llama" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  for h in "${megascale_ps_nodes[@]}"; do
    remote_exec "$h" "$megascale_ps_container" 'pids=$(pgrep -f "[s]cheduler.sh|[s]erver.sh|[b]pslaunch|[b]enchmark_megascale_ps|[p]ushpull_bench" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  sleep 3
}

net_env() {
  local mode="$1" server_count="$2" partition="$3" address_pool="$4" rx_depth="$5" fused="$6"
  cat <<EOF
export DMLC_PS_ROOT_URI=$root_uri
export DMLC_PS_ROOT_PORT=$root_port
export DMLC_NUM_SERVER=$server_count
export DMLC_NUM_WORKER=8
export DMLC_USE_GDR=0
export MEGASCALE_PS_ENABLE_FUSED_PUSH_PULL=$fused
export MEGASCALE_PS_PARTITION_BYTES=$partition
export MEGASCALE_PS_ADDRESS_POOL_SIZE=$address_pool
export MEGASCALE_PS_RDMA_RX_DEPTH=$rx_depth
export MEGASCALE_PS_RDMA_START_DEPTH=32
export NCCL_IB_HCA=mlx5_1
EOF
  if [[ "$mode" == "ucx8" ]]; then
    cat <<'EOF'
export DMLC_ENABLE_UCX=1
export DMLC_ENABLE_RDMA=0
export UCX_TLS=rc
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1
export UCX_MAX_EAGER_RAILS=2
export UCX_MAX_RNDV_RAILS=2
export UCX_RNDV_THRESH=8k
export UCX_IB_TRAFFIC_CLASS=106
export PSLITE_UCX_IB_TRAFFIC_CLASS=106
export PSLITE_UCX_USE_MT_MUTEX=y
export PSLITE_UCX_RNDV_SCHEME=put_zcopy
export UCX_WARN_UNUSED_ENV_VARS=n
EOF
  else
    cat <<'EOF'
export DMLC_ENABLE_UCX=0
export DMLC_ENABLE_RDMA=ibverbs
unset UCX_TLS UCX_NET_DEVICES UCX_MAX_EAGER_RAILS UCX_MAX_RNDV_RAILS UCX_RNDV_THRESH UCX_IB_TRAFFIC_CLASS PSLITE_UCX_IB_TRAFFIC_CLASS PSLITE_UCX_USE_MT_MUTEX PSLITE_UCX_RNDV_SCHEME UCX_WARN_UNUSED_ENV_VARS
EOF
  fi
}

workload_info() {
  local workload="$1"
  case "$workload" in
    dp_qwen3b)
      echo "qwen_3b	DP	examples/qwen/train_qwen_3b.sh	1"
      ;;
    tp_llama2_7b)
      echo "llama2_7b	TP	examples/qwen/train_llama_7b_tp_megascale_ps.sh	8"
      ;;
    dptp_qwen3_4b)
      echo "qwen3_4b	DPTP	examples/qwen/train_qwen3_4b_tp_dp_megascale_ps.sh	2"
      ;;
    *)
      return 1
      ;;
  esac
}

hostps_flags() {
  # Host-PS/MegaScalePS is the default in the current training scripts.
  # Keep this empty so the worker-side defaults stay authoritative.
  :
}

nccl_flags() {
  local workload="$1"
  case "$workload" in
    dp_qwen3b) echo "export USE_DPU=0" ;;
    tp_llama2_7b) echo "export USE_DPU=0" ;;
    dptp_qwen3_4b) echo "export USE_DPU_DP=0; export USE_DPU_TP=0" ;;
  esac
}

worker_net_env() {
  local comm="$1" mode="$2" server_count="$3" fused="$4"
  if [[ "$comm" != hostps* ]]; then
    return 0
  fi
  cat <<EOF
export DMLC_PS_ROOT_URI=$root_uri
export DMLC_PS_ROOT_PORT=$root_port
export DMLC_NUM_SERVER=$server_count
export DMLC_NUM_WORKER=8
EOF
  if [[ "$mode" == "rdma8" ]]; then
    cat <<'EOF'
export DMLC_ENABLE_UCX=0
export DMLC_ENABLE_RDMA=ibverbs
EOF
  fi
  if [[ "$fused" != "1" ]]; then
    echo "export MEGASCALE_PS_ENABLE_FUSED_PUSH_PULL=$fused"
  fi
}

start_megascale_ps() {
  local run="$1" mode="$2" server_count="$3" partition="$4" address_pool="$5" rx_depth="$6" fused="$7"
  local envs cmd server_list
  envs="$(net_env "$mode" "$server_count" "$partition" "$address_pool" "$rx_depth" "$fused")"
  cmd="echo running > /tmp/${run}-scheduler.status
$envs
cd /usr/local
bash /usr/local/scheduler.sh > /tmp/${run}-scheduler.log 2>&1
echo rc=\$? > /tmp/${run}-scheduler.status"
  remote_exec_d gpu01 "$megascale_ps_container" "$cmd" >/dev/null || return 1
  sleep 2

  read -r -a server_list <<< "$(servers_for_count "$server_count")"
  for h in "${server_list[@]}"; do
    cmd="echo running > /tmp/${run}-server.status
$envs
cd /usr/local
bash /usr/local/server.sh > /tmp/${run}-server.log 2>&1
echo rc=\$? > /tmp/${run}-server.status"
    remote_exec_d "$h" "$megascale_ps_container" "$cmd" >/dev/null || return 1
  done
  sleep 6
}

start_workers() {
  local run="$1" workload="$2" comm="$3" mode="$4" server_count="$5" partition="$6" address_pool="$7" rx_depth="$8" fused="$9" overlap="${10}" mport="${11}" train_iters="${12}"
  local model parallel script tp_size envs flags timeout_s train_export overlap_export
  IFS=$'\t' read -r model parallel script tp_size <<< "$(workload_info "$workload")"
  envs="$(worker_net_env "$comm" "$mode" "$server_count" "$fused")"
  if [[ "$comm" == hostps* ]]; then flags="$(hostps_flags "$workload")"; else flags="$(nccl_flags "$workload")"; fi
  if [[ "$train_iters" -gt 100 ]]; then timeout_s=7200; else timeout_s=1800; fi
  train_export=""
  if [[ "$train_iters" != "10" ]]; then train_export="export TRAIN_ITERS=$train_iters"; fi
  overlap_export=""
  if [[ "$overlap" != "1" ]]; then overlap_export="export USE_OVERLAP=$overlap"; fi

  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    local cmd="set -o pipefail
echo running > /tmp/${run}-worker.status
$envs
$flags
export MASTER_ADDR=$root_uri
export MASTER_PORT=$mport
export NUM_NODES=8
export GPUS_PER_NODE=1
export TP_SIZE=$tp_size
export NODE_RANK=$idx
export NCCL_IB_HCA=mlx5_1
$train_export
$overlap_export
cd /usr/local/Megatron-LM
timeout ${timeout_s}s bash $script > /tmp/${run}-worker.log 2>&1
echo rc=\$? > /tmp/${run}-worker.status"
    remote_exec_d "$h" "$worker_container" "$cmd" >/dev/null || return 1
  done
}

worker_statuses() {
  local run="$1"
  local s="" one
  for h in "${workers[@]}"; do
    one=$(remote_exec "$h" "$worker_container" "cat /tmp/${run}-worker.status 2>/dev/null || echo missing" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]*$//')
    s+="$h:$one "
  done
  echo "$s"
}

all_workers_done() {
  local statuses="$1"
  [[ "$statuses" != *running* && "$statuses" != *missing* ]]
}

workers_have_failure() {
  local statuses="$1"
  [[ "$statuses" =~ rc=([1-9][0-9]*) ]]
}

rank_log_tail() {
  local run="$1"
  remote_exec asus04 "$worker_container" "tail -220 /tmp/${run}-worker.log 2>/dev/null || true" 2>/dev/null || true
}

collect_log_file() {
  local run="$1"
  local file="$OUT_DIR/logs/${run}-rank7.log"
  remote_exec asus04 "$worker_container" "cat /tmp/${run}-worker.log 2>/dev/null || true" > "$file" 2>/dev/null || true
  echo "$file"
}

parse_stats() {
  local logfile="$1" iter_file="$2"
  perl -ne 'if(/iteration\s+(\d+)\s*\/\s*\d+.*elapsed time per iteration \(ms\):\s*([0-9.]+).*lm loss:\s*([0-9.Ee+-]+)/){print "$1\t$2\t$3\n"} elsif(/iteration\s+(\d+)\s*\/\s*\d+.*elapsed time per iteration \(ms\):\s*([0-9.]+)/){print "$1\t$2\tNA\n"}' "$logfile" > "$iter_file" || true
  awk 'BEGIN{n=0;s=0;ss=0} $1>=2 {n++; s+=$2; ss+=$2*$2} END{if(n>0){m=s/n; v=ss/n-m*m; if(v<0)v=0; printf "%d\t%.3f\t%.3f", n,m,sqrt(v)} else {printf "0\tNA\tNA"}}' "$iter_file"
}

parse_loss_time() {
  local logfile="$1" out_file="$2"
  python3 - "$logfile" "$out_file" <<'PY'
import re, sys
from datetime import datetime
log, out = sys.argv[1], sys.argv[2]
rows = []
base = None
pat = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*iteration\s+(\d+)\s*/\s*\d+.*?elapsed time per iteration \(ms\):\s*([0-9.]+).*?lm loss:\s*([0-9.Ee+-]+)")
with open(log, errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        if base is None:
            base = ts
        rows.append((int(m.group(2)), ts - base, float(m.group(3)), float(m.group(4))))
with open(out, "w") as f:
    f.write("iteration\twall_time_s\telapsed_ms\tlm_loss\n")
    for r in rows:
        f.write(f"{r[0]}\t{r[1]:.3f}\t{r[2]:.3f}\t{r[3]:.8g}\n")
PY
}

write_headers() {
  echo -e "experiment\tcategory\tworkload\tparallel\tmodel\tcomm\tmode\tstatus\tattempts\tmaster_port\tmean_ms_iter2plus\tstd_ms_iter2plus\tn_iters_used\tserver_count\tpartition_bytes\taddress_pool\trdma_rx_depth\tfused_push_pull\tuse_overlap\ttrain_iters\tworker_statuses\trank7_log\titer_file\tloss_time_file" > "$TSV"
  {
    echo "# Paper revision Megatron benchmark details"
    echo
    echo "Batch: $BATCH"
    echo "Start: $(date -Is)"
    echo "Workers: ${workers[*]}"
    echo "Servers: ${all_servers[*]}"
    echo "Worker container: $worker_container"
    echo "MegaScalePS container: $megascale_ps_container"
    echo "Root: $root_uri:$root_port"
    echo "Stats: rank7/asus04, iteration >= 2"
    echo
  } > "$DETAIL"
}

append_detail() {
  local exp="$1" status="$2" logfile="$3" iterfile="$4" lossfile="$5"
  {
    echo "## $exp"
    echo
    grep -P "^${exp}\t" "$TSV" | tail -1 | awk -F'\t' '{print "- row: " $0}'
    echo "- status: $status"
    echo "- rank7_log: $logfile"
    echo "- iter_file: $iterfile"
    if [[ -n "$lossfile" ]]; then echo "- loss_time_file: $lossfile"; fi
    echo
    echo "Iteration sample:"
    echo
    echo '```text'
    tail -20 "$iterfile" 2>/dev/null || true
    echo '```'
    echo
    echo "Rank7 tail:"
    echo
    echo '```text'
    tail -80 "$logfile" 2>/dev/null || true
    echo '```'
    echo
  } >> "$DETAIL"
}

run_job() {
  local exp="$1" category="$2" workload="$3" comm="$4" mode="$5" server_count="$6" partition="$7" address_pool="$8" rx_depth="$9" fused="${10}" overlap="${11}" train_iters="${12}" attempts_max="${13}"
  local model parallel script tp_size
  IFS=$'\t' read -r model parallel script tp_size <<< "$(workload_info "$workload")"
  local attempts=0 status="FAIL" final_port="" statuses="" logfile="" iterfile="" lossfile="" stats=$'0\tNA\tNA'
  while [[ $attempts -lt $attempts_max ]]; do
    attempts=$((attempts + 1))
    local mport=$port
    port=$((port + 1))
    local run="paperrev-${exp}-p${mport}-a${attempts}"
    final_port="$mport"
    echo "[$(date '+%H:%M:%S')] START exp=$exp workload=$workload comm=$comm mode=$mode overlap=$overlap iters=$train_iters port=$mport"
    clean_all
    if [[ "$comm" == hostps* ]]; then
      if ! start_megascale_ps "$run" "$mode" "$server_count" "$partition" "$address_pool" "$rx_depth" "$fused"; then
        status="START_FAIL"
        clean_all
        continue
      fi
    fi
    if ! start_workers "$run" "$workload" "$comm" "$mode" "$server_count" "$partition" "$address_pool" "$rx_depth" "$fused" "$overlap" "$mport" "$train_iters"; then
      status="START_FAIL"
      clean_all
      continue
    fi

    local start_ts now elapsed seen_setup=0 seen_iter=0 failed=0 hang=0 tailtxt
    start_ts=$(date +%s)
    while true; do
      sleep 20
      now=$(date +%s)
      elapsed=$((now - start_ts))
      statuses=$(worker_statuses "$run")
      tailtxt=$(rank_log_tail "$run")
      if echo "$tailtxt" | grep -q "time across ranks"; then seen_setup=1; fi
      if echo "$tailtxt" | grep -q "elapsed time per iteration"; then seen_iter=1; fi
      if echo "$tailtxt" | egrep -q "Traceback|RuntimeError|CUDA out of memory|RDMA WRITE FAILURE|Segmentation fault|Address already in use|DistNetworkError|Endpoint timeout|Check failed"; then failed=1; fi
      if all_workers_done "$statuses"; then break; fi
      if workers_have_failure "$statuses"; then failed=1; break; fi
      if [[ $failed -eq 1 ]]; then break; fi
      if [[ $seen_setup -eq 0 && $elapsed -gt 300 ]]; then hang=1; break; fi
      if [[ $seen_setup -eq 1 && $seen_iter -eq 0 && $elapsed -gt 900 ]]; then hang=1; break; fi
      if [[ $train_iters -gt 100 && $elapsed -gt 7200 ]]; then hang=1; break; fi
      if [[ $train_iters -le 100 && $elapsed -gt 1800 ]]; then hang=1; break; fi
      if (( elapsed % 120 < 20 )); then echo "[$(date '+%H:%M:%S')] RUNNING exp=$exp elapsed=${elapsed}s setup=$seen_setup iter=$seen_iter statuses=$statuses"; fi
    done

    logfile=$(collect_log_file "$run")
    iterfile="$OUT_DIR/logs/${run}-iterations.tsv"
    stats=$(parse_stats "$logfile" "$iterfile")
    lossfile=""
    if [[ "$train_iters" -gt 100 ]]; then
      lossfile="$OUT_DIR/logs/${run}-loss-time.tsv"
      parse_loss_time "$logfile" "$lossfile"
    fi
    statuses=$(worker_statuses "$run")
    clean_all

    if [[ $hang -eq 1 ]]; then
      status="HANG"
      echo "[$(date '+%H:%M:%S')] HANG exp=$exp attempt=$attempts"
      continue
    fi
    if [[ $failed -eq 1 || "$statuses" == *rc=124* || "$statuses" == *rc=1* || "$statuses" == *rc=2* || "$statuses" == *rc=134* || "$statuses" == *rc=139* ]]; then
      status="FAIL"
      echo "[$(date '+%H:%M:%S')] FAIL exp=$exp attempt=$attempts statuses=$statuses"
      if [[ $attempts -lt $attempts_max ]]; then
        continue
      fi
      break
    fi
    local n
    n=$(echo -e "$stats" | awk '{print $1}')
    if [[ "$statuses" == *rc=0* && "$n" != "0" ]]; then
      status="OK"
      break
    fi
    status="UNKNOWN"
    break
  done

  local n mean std
  n=$(echo -e "$stats" | awk '{print $1}')
  mean=$(echo -e "$stats" | awk '{print $2}')
  std=$(echo -e "$stats" | awk '{print $3}')
  echo -e "$exp\t$category\t$workload\t$parallel\t$model\t$comm\t$mode\t$status\t$attempts\t$final_port\t$mean\t$std\t$n\t$server_count\t$partition\t$address_pool\t$rx_depth\t$fused\t$overlap\t$train_iters\t$statuses\t$logfile\t$iterfile\t$lossfile" >> "$TSV"
  append_detail "$exp" "$status" "$logfile" "$iterfile" "$lossfile"
  echo "[$(date '+%H:%M:%S')] DONE exp=$exp status=$status mean=$mean std=$std n=$n"
}

write_summary() {
  {
    echo "# Paper revision Megatron benchmark summary"
    echo
    echo "Batch: $BATCH"
    echo "Generated: $(date -Is)"
    echo "Raw TSV: $TSV"
    echo "Detail: $DETAIL"
    echo
    echo "| Experiment | Category | Workload | Comm | Mode | Status | Mean ms | Std ms | n | Servers | Part | Overlap | Iters |"
    echo "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    tail -n +2 "$TSV" | awk -F'\t' '{printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",$1,$2,$3,$6,$7,$8,$11,$12,$13,$14,$15,$19,$20}'
    echo
    echo "Best completed Host-PS candidate per main workload:"
    awk -F'\t' 'NR>1 && $2=="main" && $8=="OK" && $6 ~ /^hostps/ {print $3 "\t" $1 "\t" $11 "\t" $7}' "$TSV" | sort -k1,1 -k3,3n
  } > "$SUMMARY"
}

write_headers
trap 'clean_all >/dev/null 2>&1 || true' EXIT

# Main end-to-end results: correct model per mode, and candidate Host-PS configs.
run_job "main_dp_qwen3b_nccl" "main" "dp_qwen3b" "nccl" "none" 0 4194304 10240 512 1 1 10 1
run_job "main_dp_qwen3b_hostps_ucx8" "main" "dp_qwen3b" "hostps" "ucx8" 8 4194304 10240 512 1 1 10 2
run_job "main_dp_qwen3b_hostps_rdma8" "main" "dp_qwen3b" "hostps" "rdma8" 8 4194304 10240 512 1 1 10 1

run_job "main_tp_llama2_7b_nccl" "main" "tp_llama2_7b" "nccl" "none" 0 2097152 10240 256 1 1 10 1
run_job "main_tp_llama2_7b_hostps_ucx8" "main" "tp_llama2_7b" "hostps" "ucx8" 8 1048576 10240 512 1 1 10 2
run_job "main_tp_llama2_7b_hostps_rdma8" "main" "tp_llama2_7b" "hostps" "rdma8" 8 2097152 10240 256 1 1 10 1

run_job "main_dptp_qwen4b_nccl" "main" "dptp_qwen3_4b" "nccl" "none" 0 4194304 10240 1024 1 1 10 1
run_job "main_dptp_qwen4b_hostps_ucx8" "main" "dptp_qwen3_4b" "hostps" "ucx8" 8 1048576 10240 512 1 1 10 2
run_job "main_dptp_qwen4b_hostps_rdma8" "main" "dptp_qwen3_4b" "hostps" "rdma8" 8 4194304 10240 1024 1 1 10 1

# Server endpoint scaling on DP Qwen-3B.
run_job "server_dp_qwen3b_2s" "server_count_dp" "dp_qwen3b" "hostps" "rdma8" 2 4194304 10240 512 1 1 10 2
run_job "server_dp_qwen3b_4s" "server_count_dp" "dp_qwen3b" "hostps" "rdma8" 4 4194304 10240 512 1 1 10 1
run_job "server_dp_qwen3b_8s" "server_count_dp" "dp_qwen3b" "hostps" "rdma8" 8 4194304 10240 512 1 1 10 1

# Megatron overlap switch on TP+DP Qwen3-4B.
run_job "overlap_qwen4b_nccl_on" "overlap_qwen4b" "dptp_qwen3_4b" "nccl" "none" 0 4194304 10240 1024 1 1 10 1
run_job "overlap_qwen4b_nccl_off" "overlap_qwen4b" "dptp_qwen3_4b" "nccl" "none" 0 4194304 10240 1024 1 0 10 1
run_job "overlap_qwen4b_hostps_on" "overlap_qwen4b" "dptp_qwen3_4b" "hostps" "ucx8" 8 1048576 10240 512 1 1 10 2
run_job "overlap_qwen4b_hostps_off" "overlap_qwen4b" "dptp_qwen3_4b" "hostps" "ucx8" 8 1048576 10240 512 1 0 10 2

if [[ "${MEGASCALE_PS_RUN_LOSS:-1}" == "1" ]]; then
  run_job "loss_qwen4b_nccl_1000" "loss_qwen4b" "dptp_qwen3_4b" "nccl" "none" 0 4194304 10240 1024 1 1 1000 1
  run_job "loss_qwen4b_hostps_1000" "loss_qwen4b" "dptp_qwen3_4b" "hostps" "ucx8" 8 1048576 10240 512 1 1 1000 2
fi

write_summary
echo "OUT_DIR=$OUT_DIR"
echo "SUMMARY=$SUMMARY"
echo "TSV=$TSV"
