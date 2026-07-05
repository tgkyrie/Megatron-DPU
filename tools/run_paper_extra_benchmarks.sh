#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/home/cmcc/CMCC/Megatron-DPU"
OUT_ROOT="$BASE_DIR/docs/benchmark_runs"
BATCH="$(date +%Y%m%d-%H%M%S)-paper-extra"
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
port=19300

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
    remote_exec "$h" "$worker_container" 'pids=$(pgrep -f "[t]orchrun|[p]retrain_gpt|[t]rain_.*megascale_ps|[t]rain_qwen_3b.sh" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  for h in "${megascale_ps_nodes[@]}"; do
    remote_exec "$h" "$megascale_ps_container" 'pids=$(pgrep -f "[s]cheduler.sh|[s]erver.sh|[b]pslaunch|[b]enchmark_megascale_ps|[p]ushpull_bench" || true); if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi' >/dev/null 2>&1 || true
  done
  sleep 3
}

net_env() {
  local server_count="$1" partition="$2" address_pool="$3" rx_depth="$4" fused="$5"
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
export DMLC_ENABLE_UCX=0
export DMLC_ENABLE_RDMA=ibverbs
unset UCX_TLS UCX_NET_DEVICES UCX_MAX_EAGER_RAILS UCX_MAX_RNDV_RAILS UCX_RNDV_THRESH UCX_IB_TRAFFIC_CLASS PSLITE_UCX_IB_TRAFFIC_CLASS PSLITE_UCX_USE_MT_MUTEX PSLITE_UCX_RNDV_SCHEME UCX_WARN_UNUSED_ENV_VARS
EOF
}

worker_net_env() {
  local server_count="$1" partition="$2" address_pool="$3" rx_depth="$4" fused="$5"
  cat <<EOF
export DMLC_PS_ROOT_URI=$root_uri
export DMLC_PS_ROOT_PORT=$root_port
export DMLC_NUM_SERVER=$server_count
export DMLC_NUM_WORKER=8
export DMLC_ENABLE_UCX=0
export DMLC_ENABLE_RDMA=ibverbs
EOF
  [[ "$partition" != "4194304" ]] && echo "export MEGASCALE_PS_PARTITION_BYTES=$partition"
  [[ "$address_pool" != "10240" ]] && echo "export MEGASCALE_PS_ADDRESS_POOL_SIZE=$address_pool"
  [[ "$rx_depth" != "1024" ]] && echo "export MEGASCALE_PS_RDMA_RX_DEPTH=$rx_depth"
  [[ "$fused" != "1" ]] && echo "export MEGASCALE_PS_ENABLE_FUSED_PUSH_PULL=$fused"
}

start_megascale_ps() {
  local run="$1" server_count="$2" partition="$3" address_pool="$4" rx_depth="$5" fused="$6"
  local envs
  envs="$(net_env "$server_count" "$partition" "$address_pool" "$rx_depth" "$fused")"
  local cmd="echo running > /tmp/${run}-scheduler.status
$envs
cd /usr/local
bash /usr/local/scheduler.sh > /tmp/${run}-scheduler.log 2>&1
echo rc=\$? > /tmp/${run}-scheduler.status"
  remote_exec_d gpu01 "$megascale_ps_container" "$cmd" >/dev/null || return 1
  sleep 2

  local server_list
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
  local run="$1" server_count="$2" partition="$3" address_pool="$4" rx_depth="$5" fused="$6" mport="$7"
  local envs
  envs="$(worker_net_env "$server_count" "$partition" "$address_pool" "$rx_depth" "$fused")"
  for idx in "${!workers[@]}"; do
    local h="${workers[$idx]}"
    local cmd="echo running > /tmp/${run}-worker.status
$envs
export MASTER_ADDR=$root_uri
export MASTER_PORT=$mport
export NUM_NODES=8
export GPUS_PER_NODE=1
export TP_SIZE=2
export NODE_RANK=$idx
export NCCL_IB_HCA=mlx5_1
cd /usr/local/Megatron-LM
timeout 1200s bash examples/qwen/train_qwen_3b_tp_dp_megascale_ps.sh > /tmp/${run}-worker.log 2>&1
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

cleanup_after_run() {
  clean_all >/dev/null 2>&1 || true
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
  awk 'BEGIN{n=0;s=0;ss=0} $1>=2 {n++; s+=$2; ss+=$2*$2} END{if(n>0){m=s/n; v=ss/n-m*m; if(v<0)v=0; printf "%d\t%.3f\t%.3f", n,m,sqrt(v)} else {printf "0\tNA\tNA"}}' "$iter_file"
}

write_headers() {
  echo -e "experiment\tcategory\tmodel\tstatus\tattempts\tmaster_port\tmean_ms_iter2plus\tstd_ms_iter2plus\tn_iters_used\tserver_count\tpartition_bytes\taddress_pool\trdma_rx_depth\tfused_push_pull\tworker_statuses\trank7_log\titer_file" > "$TSV"
  {
    echo "# Paper extra Megatron benchmark details"
    echo
    echo "Batch: $BATCH"
    echo "Start: $(date -Is)"
    echo "Workers: ${workers[*]}"
    echo "Servers: ${all_servers[*]}"
    echo "Worker container: $worker_container"
    echo "MegaScalePS container: $megascale_ps_container"
    echo "Root: $root_uri:$root_port"
    echo "Workload: qwen_3b TP+DP, 8 workers, TP_SIZE=2, TRAIN_ITERS=10"
    echo "Stats: rank7/asus04, iteration >= 2"
    echo
  } > "$DETAIL"
}

append_detail() {
  local exp="$1" cat="$2" status="$3" attempts="$4" mport="$5" mean="$6" std="$7" n="$8" sc="$9" part="${10}" addr="${11}" rx="${12}" fused="${13}" statuses="${14}" logfile="${15}" iterfile="${16}"
  {
    echo "## $exp"
    echo
    echo "- category: $cat"
    echo "- status: $status"
    echo "- attempts: $attempts"
    echo "- master_port: $mport"
    echo "- params: servers=$sc, partition=$part, address_pool=$addr, rdma_rx_depth=$rx, fused_push_pull=$fused"
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

run_one() {
  local exp="$1" cat="$2" server_count="$3" partition="$4" address_pool="$5" rx_depth="$6" fused="$7"
  local attempts=0 status="FAIL" final_port="" statuses="" logfile="" iterfile="" stats="0	NA	NA"
  while [[ $attempts -lt 2 ]]; do
    attempts=$((attempts + 1))
    local mport=$port
    port=$((port + 1))
    local run="paper-${exp}-p${mport}-a${attempts}"
    final_port="$mport"
    echo "[$(date '+%H:%M:%S')] START exp=$exp category=$cat servers=$server_count partition=$partition fused=$fused attempt=$attempts port=$mport"
    clean_all
    if ! start_megascale_ps "$run" "$server_count" "$partition" "$address_pool" "$rx_depth" "$fused"; then
      status="START_FAIL"
      cleanup_after_run
      continue
    fi
    if ! start_workers "$run" "$server_count" "$partition" "$address_pool" "$rx_depth" "$fused" "$mport"; then
      status="START_FAIL"
      cleanup_after_run
      continue
    fi

    local start_ts now elapsed seen_setup=0 seen_iter=0 hang=0 failed=0
    start_ts=$(date +%s)
    while true; do
      sleep 15
      now=$(date +%s)
      elapsed=$((now - start_ts))
      statuses=$(worker_statuses "$run")
      local tailtxt
      tailtxt=$(rank_log_tail "$run")
      if echo "$tailtxt" | grep -q "time across ranks"; then seen_setup=1; fi
      if echo "$tailtxt" | grep -q "elapsed time per iteration"; then seen_iter=1; fi
      if echo "$tailtxt" | egrep -q "Traceback|RuntimeError|CUDA out of memory|RDMA WRITE FAILURE|Segmentation fault|Address already in use|DistNetworkError|Check failed"; then failed=1; fi
      if all_workers_done "$statuses"; then break; fi
      if [[ $failed -eq 1 ]]; then break; fi
      if [[ $seen_setup -eq 0 && $elapsed -gt 180 ]]; then hang=1; break; fi
      if [[ $seen_setup -eq 1 && $seen_iter -eq 0 && $elapsed -gt 600 ]]; then hang=1; break; fi
      if [[ $elapsed -gt 1200 ]]; then hang=1; break; fi
      if (( elapsed % 60 < 15 )); then echo "[$(date '+%H:%M:%S')] RUNNING exp=$exp elapsed=${elapsed}s setup=$seen_setup iter=$seen_iter statuses=$statuses"; fi
    done

    logfile=$(collect_log_file "$run")
    iterfile="$OUT_DIR/logs/${run}-iterations.tsv"
    stats=$(parse_stats "$logfile" "$iterfile")
    statuses=$(worker_statuses "$run")
    cleanup_after_run

    if [[ $hang -eq 1 ]]; then
      status="HANG"
      echo "[$(date '+%H:%M:%S')] HANG exp=$exp attempt=$attempts"
      continue
    fi
    if [[ $failed -eq 1 || "$statuses" == *rc=124* || "$statuses" == *rc=1* || "$statuses" == *rc=2* || "$statuses" == *rc=134* || "$statuses" == *rc=139* ]]; then
      status="FAIL"
      break
    fi
    if [[ "$statuses" == *rc=0* && "$stats" != $'0	NA	NA' ]]; then
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
  echo -e "$exp\t$cat\tqwen_3b\t$status\t$attempts\t$final_port\t$mean\t$std\t$n\t$server_count\t$partition\t$address_pool\t$rx_depth\t$fused\t$statuses\t$logfile\t$iterfile" >> "$TSV"
  append_detail "$exp" "$cat" "$status" "$attempts" "$final_port" "$mean" "$std" "$n" "$server_count" "$partition" "$address_pool" "$rx_depth" "$fused" "$statuses" "$logfile" "$iterfile"
  echo "[$(date '+%H:%M:%S')] DONE exp=$exp status=$status mean=$mean std=$std n=$n"
}

write_summary() {
  {
    echo "# Paper extra Megatron benchmark summary"
    echo
    echo "Batch: $BATCH"
    echo "Generated: $(date -Is)"
    echo "Detail: $DETAIL"
    echo "Raw TSV: $TSV"
    echo
    echo "| Experiment | Category | Status | Attempts | Port | Mean ms | Std ms | n | Servers | Partition | Fused |"
    echo "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    tail -n +2 "$TSV" | awk -F'\t' '{printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",$1,$2,$4,$5,$6,$7,$8,$9,$10,$11,$14}'
  } > "$SUMMARY"
}

write_headers

run_one "server_count_2" "server_count" 2 4194304 10240 1024 1
run_one "server_count_4" "server_count" 4 4194304 10240 1024 1
run_one "server_count_8" "server_count" 8 4194304 10240 1024 1
run_one "partition_1mb" "partition" 8 1048576 10240 1024 1
run_one "partition_4mb" "partition" 8 4194304 10240 1024 1
run_one "partition_16mb" "partition" 8 16777216 10240 1024 1
run_one "overlap_off" "fused_overlap" 8 4194304 10240 1024 0
run_one "overlap_on" "fused_overlap" 8 4194304 10240 1024 1

write_summary

echo "OUT_DIR=$OUT_DIR"
echo "DETAIL=$DETAIL"
echo "SUMMARY=$SUMMARY"
echo "TSV=$TSV"
