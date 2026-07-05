#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/home/cmcc/CMCC/Megatron-DPU"
OUT_ROOT="$BASE_DIR/docs/benchmark_runs"
BATCH="$(date +%Y%m%d-%H%M%S)-tp-only-ucx-sweep"
OUT_DIR="$OUT_ROOT/$BATCH"
mkdir -p "$OUT_DIR/logs"

DETAIL="$OUT_DIR/details.md"
SUMMARY="$OUT_DIR/summary.md"
TSV="$OUT_DIR/results.tsv"

workers=(gpu01 gpu02 gpu03 gpu04 asus01 asus02 asus03 asus04)
megascale_ps_nodes=(gpu01 R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)
servers=(R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)

worker_container="${WORKER_CONTAINER:-megatron-dpu-latest}"
megascale_ps_container="${MEGASCALE_PS_CONTAINER:-megascale_ps-latest}"
root_uri="192.168.1.10"
root_port="9010"
port=19400

remote_exec() {
  local host="$1" container="$2" cmd="$3"
  ssh "$host" "sudo docker exec $container bash -lc $(printf '%q' "$cmd")"
}

remote_exec_d() {
  local host="$1" container="$2" cmd="$3"
  ssh "$host" "sudo docker exec -d $container bash -lc $(printf '%q' "$cmd")"
}

clean_all() {
  for h in "${workers[@]}"; do
    remote_exec "$h" "$worker_container" 'pids=$(pgrep -f "[t]orchrun|[p]retrain_gpt|[t]rain_.*megascale_ps|[t]rain_qwen_3b" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  for h in "${megascale_ps_nodes[@]}"; do
    remote_exec "$h" "$megascale_ps_container" 'pids=$(pgrep -f "[s]cheduler.sh|[s]erver.sh|[b]pslaunch|[b]enchmark_megascale_ps|[p]ushpull_bench" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  sleep 3
}

ucx_env() {
  local partition="$1" address_pool="$2" rx_depth="$3"
  cat <<EOF
export DMLC_PS_ROOT_URI=$root_uri
export DMLC_PS_ROOT_PORT=$root_port
export DMLC_NUM_SERVER=8
export DMLC_NUM_WORKER=8
export DMLC_USE_GDR=0
export MEGASCALE_PS_ENABLE_FUSED_PUSH_PULL=1
export MEGASCALE_PS_PARTITION_BYTES=$partition
export MEGASCALE_PS_ADDRESS_POOL_SIZE=$address_pool
export MEGASCALE_PS_RDMA_RX_DEPTH=$rx_depth
export MEGASCALE_PS_RDMA_START_DEPTH=32
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
}

nccl_env() {
  cat <<EOF
export MASTER_ADDR=$root_uri
export NUM_NODES=8
export GPUS_PER_NODE=1
export TP_SIZE=8
export NCCL_IB_HCA=mlx5_1
EOF
}

worker_ucx_env() {
  local partition="$1" address_pool="$2" rx_depth="$3"
  cat <<EOF
export DMLC_PS_ROOT_URI=$root_uri
export DMLC_PS_ROOT_PORT=$root_port
export DMLC_NUM_SERVER=8
export DMLC_NUM_WORKER=8
EOF
  [[ "$partition" != "4194304" ]] && echo "export MEGASCALE_PS_PARTITION_BYTES=$partition"
  [[ "$address_pool" != "10240" ]] && echo "export MEGASCALE_PS_ADDRESS_POOL_SIZE=$address_pool"
  [[ "$rx_depth" != "512" ]] && echo "export MEGASCALE_PS_RDMA_RX_DEPTH=$rx_depth"
}

start_megascale_ps() {
  local run="$1" partition="$2" address_pool="$3" rx_depth="$4"
  local envs
  envs="$(ucx_env "$partition" "$address_pool" "$rx_depth")"
  local cmd="echo running > /tmp/${run}-scheduler.status
$envs
cd /usr/local
bash /usr/local/scheduler.sh > /tmp/${run}-scheduler.log 2>&1
echo rc=\$? > /tmp/${run}-scheduler.status"
  remote_exec_d gpu01 "$megascale_ps_container" "$cmd" >/dev/null || return 1
  sleep 2

  for h in "${servers[@]}"; do
    cmd="echo running > /tmp/${run}-server.status
$envs
cd /usr/local
bash /usr/local/server.sh > /tmp/${run}-server.log 2>&1
echo rc=\$? > /tmp/${run}-server.status"
    remote_exec_d "$h" "$megascale_ps_container" "$cmd" >/dev/null || return 1
  done
  sleep 6
}

start_megascale_ps_workers() {
  local run="$1" partition="$2" address_pool="$3" rx_depth="$4" mport="$5"
  local envs
  envs="$(worker_ucx_env "$partition" "$address_pool" "$rx_depth")"
  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    local cmd="echo running > /tmp/${run}-worker.status
$envs
export MASTER_ADDR=$root_uri
export MASTER_PORT=$mport
export NUM_NODES=8
export GPUS_PER_NODE=1
export TP_SIZE=8
export NODE_RANK=$idx
export NCCL_IB_HCA=mlx5_1
cd /usr/local/Megatron-LM
timeout 1200s bash examples/qwen/train_qwen_3b_tp_megascale_ps.sh > /tmp/${run}-worker.log 2>&1
echo rc=\$? > /tmp/${run}-worker.status"
    remote_exec_d "$h" "$worker_container" "$cmd" >/dev/null || return 1
  done
}

start_nccl_workers() {
  local run="$1" mport="$2"
  local envs
  envs="$(nccl_env)"
  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    local cmd="echo running > /tmp/${run}-worker.status
$envs
export USE_DPU=0
export MASTER_PORT=$mport
export NODE_RANK=$idx
cd /usr/local/Megatron-LM
timeout 1200s bash examples/qwen/train_qwen_3b_tp_megascale_ps.sh > /tmp/${run}-worker.log 2>&1
echo rc=\$? > /tmp/${run}-worker.status"
    remote_exec_d "$h" "$worker_container" "$cmd" >/dev/null || return 1
  done
}

worker_statuses() {
  local run="$1"
  local s=""
  for h in "${workers[@]}"; do
    local one
    one=$(remote_exec "$h" "$worker_container" "cat /tmp/${run}-worker.status 2>/dev/null || echo missing" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]*$//')
    s+="$h:$one "
  done
  echo "$s"
}

all_workers_done() {
  local statuses="$1"
  [[ "$statuses" != *running* && "$statuses" != *missing* ]]
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
  local logfile="$1"
  local iter_file="$2"
  perl -ne 'if(/iteration\s+(\d+)\s*\/\s*\d+.*elapsed time per iteration \(ms\):\s*([0-9.]+)/){print "$1\t$2\n"}' "$logfile" > "$iter_file" || true
  awk 'BEGIN{n=0;s=0;ss=0;thr=0}
       $1>=2 {n++; s+=$2; ss+=$2*$2}
       END{if(n>0){m=s/n; v=ss/n-m*m; if(v<0)v=0; printf "%d\t%.3f\t%.3f", n,m,sqrt(v)} else {printf "0\tNA\tNA"}}' "$iter_file"
}

write_headers() {
  echo -e "experiment\tcategory\tmodel\tcomm\tstatus\tattempts\tmaster_port\tmean_ms_iter2plus\tstd_ms_iter2plus\tn_iters_used\tpartition_bytes\taddress_pool\trdma_rx_depth\tworker_statuses\trank7_log\titer_file" > "$TSV"
  {
    echo "# TP-only UCX sweep"
    echo
    echo "Batch: $BATCH"
    echo "Start: $(date -Is)"
    echo "Workers: ${workers[*]}"
    echo "Servers: ${servers[*]}"
    echo "Workload: qwen_3b TP-only, 8 workers, TP_SIZE=8, SEQ_LENGTH=6144, TRAIN_ITERS=10"
    echo "Stats: rank7/asus04, iteration >= 2"
    echo
  } > "$DETAIL"
}

append_detail() {
  local exp="$1" cat="$2" comm="$3" status="$4" attempts="$5" mport="$6" mean="$7" std="$8" n="$9" part="${10}" addr="${11}" rx="${12}" statuses="${13}" logfile="${14}" iterfile="${15}"
  {
    echo "## $exp"
    echo
    echo "- category: $cat"
    echo "- comm: $comm"
    echo "- status: $status"
    echo "- attempts: $attempts"
    echo "- master_port: $mport"
    echo "- params: partition=$part, address_pool=$addr, rdma_rx_depth=$rx"
    echo "- worker_statuses: $statuses"
    echo "- rank7_log: $logfile"
    echo "- iter_file: $iterfile"
    echo "- stats(iter>=2): n=$n mean_ms=$mean std_ms=$std"
    echo
    echo "Iteration times:"
    echo
    echo '```text'
    cat "$iterfile" 2>/dev/null || true
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

wait_and_collect() {
  local run="$1"
  local start_ts now elapsed seen_setup=0 seen_iter=0 hang=0 failed=0 statuses tailtxt
  start_ts=$(date +%s)
  while true; do
    sleep 15
    now=$(date +%s)
    elapsed=$((now - start_ts))
    statuses=$(worker_statuses "$run")
    tailtxt=$(rank_log_tail "$run")
    if echo "$tailtxt" | grep -q "time across ranks"; then seen_setup=1; fi
    if echo "$tailtxt" | grep -q "elapsed time per iteration"; then seen_iter=1; fi
    if echo "$tailtxt" | egrep -q "Traceback|RuntimeError|CUDA out of memory|RDMA WRITE FAILURE|Segmentation fault|Address already in use|DistNetworkError|Check failed|Endpoint timeout"; then failed=1; fi
    if all_workers_done "$statuses"; then break; fi
    if [[ $failed -eq 1 ]]; then break; fi
    if [[ $seen_setup -eq 0 && $elapsed -gt 240 ]]; then hang=1; break; fi
    if [[ $seen_setup -eq 1 && $seen_iter -eq 0 && $elapsed -gt 720 ]]; then hang=1; break; fi
    if [[ $elapsed -gt 1200 ]]; then hang=1; break; fi
    if (( elapsed % 60 < 15 )); then
      echo "[$(date '+%H:%M:%S')] RUNNING run=$run elapsed=${elapsed}s setup=$seen_setup iter=$seen_iter statuses=$statuses"
    fi
  done
  if [[ $hang -eq 1 ]]; then
    echo "HANG"
  elif [[ $failed -eq 1 ]]; then
    echo "FAIL"
  else
    echo "DONE"
  fi
}

record_row() {
  local exp="$1" cat="$2" comm="$3" status="$4" attempts="$5" mport="$6" part="$7" addr="$8" rx="$9" statuses="${10}" logfile="${11}" iterfile="${12}" stats="${13}"
  local n mean std
  n=$(echo "$stats" | cut -f1)
  mean=$(echo "$stats" | cut -f2)
  std=$(echo "$stats" | cut -f3)
  if [[ "$status" == "DONE" && "$n" != "0" ]]; then
    status="OK"
  elif [[ "$status" == "DONE" && "$n" == "0" ]]; then
    status="FAIL"
  fi
  echo -e "$exp\t$cat\tqwen_3b\t$comm\t$status\t$attempts\t$mport\t$mean\t$std\t$n\t$part\t$addr\t$rx\t$statuses\t$logfile\t$iterfile" >> "$TSV"
  append_detail "$exp" "$cat" "$comm" "$status" "$attempts" "$mport" "$mean" "$std" "$n" "$part" "$addr" "$rx" "$statuses" "$logfile" "$iterfile"
}

run_nccl() {
  local exp="tp_qwen3b_nccl_gpu0"
  local attempts=1
  local mport=$port
  port=$((port + 1))
  local run="tpsweep-${exp}-p${mport}-a${attempts}"
  echo "[$(date '+%H:%M:%S')] START exp=$exp comm=nccl port=$mport"
  clean_all
  if ! start_nccl_workers "$run" "$mport"; then
    echo "[$(date '+%H:%M:%S')] START_FAIL exp=$exp"
  fi
  local status
  status=$(wait_and_collect "$run")
  local logfile iterfile stats statuses
  logfile=$(collect_log_file "$run")
  iterfile="$OUT_DIR/logs/${run}-iterations.tsv"
  stats=$(parse_stats "$logfile" "$iterfile")
  statuses=$(worker_statuses "$run")
  clean_all
  record_row "$exp" "baseline" "nccl" "$status" "$attempts" "$mport" "NA" "NA" "NA" "$statuses" "$logfile" "$iterfile" "$stats"
}

run_ucx() {
  local exp="$1" partition="$2" address_pool="$3" rx_depth="$4"
  local attempts=1
  local mport=$port
  port=$((port + 1))
  local run="tpsweep-${exp}-p${mport}-a${attempts}"
  echo "[$(date '+%H:%M:%S')] START exp=$exp comm=hostps_ucx8 partition=$partition pool=$address_pool rx=$rx_depth port=$mport"
  clean_all
  if ! start_megascale_ps "$run" "$partition" "$address_pool" "$rx_depth"; then
    echo "[$(date '+%H:%M:%S')] START_FAIL megascale_ps exp=$exp"
  fi
  if ! start_megascale_ps_workers "$run" "$partition" "$address_pool" "$rx_depth" "$mport"; then
    echo "[$(date '+%H:%M:%S')] START_FAIL workers exp=$exp"
  fi
  local status
  status=$(wait_and_collect "$run")
  local logfile iterfile stats statuses
  logfile=$(collect_log_file "$run")
  iterfile="$OUT_DIR/logs/${run}-iterations.tsv"
  stats=$(parse_stats "$logfile" "$iterfile")
  statuses=$(worker_statuses "$run")
  clean_all
  record_row "$exp" "tp_param_sweep" "hostps_ucx8" "$status" "$attempts" "$mport" "$partition" "$address_pool" "$rx_depth" "$statuses" "$logfile" "$iterfile" "$stats"
}

summarize() {
  {
    echo "# TP-only UCX sweep summary"
    echo
    echo "Batch: $BATCH"
    echo
    echo '```tsv'
    cat "$TSV"
    echo '```'
    echo
    echo "Best Host-PS UCX8 row by mean_ms_iter2plus:"
    awk -F'\t' 'NR>1 && $5=="OK" && $4=="hostps_ucx8" && $8!="NA" {print $0}' "$TSV" | sort -t$'\t' -k8,8n | head -1
    echo
    echo "End: $(date -Is)"
  } > "$SUMMARY"
}

write_headers
trap 'clean_all >/dev/null 2>&1 || true' EXIT

run_nccl
run_ucx "tp_qwen3b_ucx_1mb_pool10240_rx512" 1048576 10240 512
run_ucx "tp_qwen3b_ucx_2mb_pool10240_rx256" 2097152 10240 256
run_ucx "tp_qwen3b_ucx_2mb_pool10240_rx512" 2097152 10240 512
run_ucx "tp_qwen3b_ucx_4mb_pool10240_rx512" 4194304 10240 512
run_ucx "tp_qwen3b_ucx_4mb_pool20480_rx512" 4194304 20480 512

summarize
echo "$OUT_DIR"
