#!/usr/bin/env bash
set -u

ROOT="/home/cmcc/CMCC/Megatron-DPU"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="$ROOT/docs/benchmark_runs/${STAMP}-server-count-ucx-matrix"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/server_count_summary.tsv"

HOSTS=(R750-1 R750-2 R750-3 R750-4 server12 server13 server14 server15)
IPS=(192.168.1.40 192.168.1.41 192.168.1.42 192.168.1.43 192.168.1.30 192.168.1.31 192.168.1.32 192.168.1.33)
WORKLOADS="${WORKLOADS_OVERRIDE:-dp_qwen3b dptp_qwen3_4b}"
COUNTS="${SERVER_COUNTS_OVERRIDE:-1 2 3 4 5 6 7 8}"

echo -e "label\tworkload\tserver_count\thosts\tips\trunner_out\tcase\tattempt\trail_variant\trun\tstatus\tps_port\tmaster_port\tmean_ms\tstd_ms\tn\tvalid_loss\ttest_loss\tmetric_worker_log\titer_file" > "$SUMMARY"

join_prefix() {
  local n="$1"
  local -n arr="$2"
  local out=""
  local i
  for ((i = 0; i < n; i++)); do
    if [[ -n "$out" ]]; then
      out+=","
    fi
    out+="${arr[$i]}"
  done
  printf '%s' "$out"
}

run_one() {
  local workload="$1"
  local count="$2"
  local hosts ips label log rc runner_out
  hosts="$(join_prefix "$count" HOSTS)"
  ips="$(join_prefix "$count" IPS)"
  label="${workload}_${count}server"
  log="$OUT_ROOT/${label}.driver.log"
  echo "[$(date '+%H:%M:%S')] START workload=$workload servers=$count hosts=$hosts"

  (
    cd "$ROOT" || exit 1
    env \
      WORKLOAD="$workload" \
      RUN_CASES=hostservers \
      TRAIN_ITERS="${TRAIN_ITERS_OVERRIDE:-20}" \
      EVAL_INTERVAL=1000 \
      EVAL_ITERS=0 \
      NETWORK_MODE=ucx \
      UCX_RAIL_VARIANTS=single \
      HOST_SERVERS_OVERRIDE="$hosts" \
      HOST_SERVER_IPS_OVERRIDE="$ips" \
      DMLC_NUM_SERVER_OVERRIDE="$count" \
      MAX_RETRIES="${MAX_RETRIES_OVERRIDE:-2}" \
      MONITOR_TIMEOUT=1200 \
      NO_ITER_TIMEOUT=420 \
      WORKER_TIMEOUT=1800 \
      ROLE_TIMEOUT=1200 \
      python3 tools/run_megatron_ucx_semantic.py
  ) > "$log" 2>&1
  rc=$?

  runner_out="$(grep -o 'OUT=/.*' "$log" | tail -1 | sed 's/^OUT=//')"
  if [[ -z "$runner_out" || ! -f "$runner_out/summary.tsv" ]]; then
    echo -e "$label\t$workload\t$count\t$hosts\t$ips\t${runner_out:-NA}\tNA\tNA\tNA\tNA\tDRIVER_RC_$rc\tNA\tNA\tNA\tNA\t0\tNA\tNA\t$log\tNA" >> "$SUMMARY"
    echo "[$(date '+%H:%M:%S')] FAIL workload=$workload servers=$count rc=$rc log=$log"
    return 0
  fi

  tail -n +2 "$runner_out/summary.tsv" | awk -v label="$label" -v workload="$workload" -v count="$count" -v hosts="$hosts" -v ips="$ips" -v out="$runner_out" -F'\t' 'BEGIN{OFS="\t"} {print label,workload,count,hosts,ips,out,$0}' >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] DONE workload=$workload servers=$count rc=$rc out=$runner_out"
}

for workload in $WORKLOADS; do
  for count in $COUNTS; do
    run_one "$workload" "$count"
  done
done

cat > "$OUT_ROOT/run_notes.md" <<EOF
# Server Count UCX Matrix

Date: $(date -Is)

Workloads:

- $WORKLOADS

Server counts:

- $COUNTS

Summary:

- \`server_count_summary.tsv\`
EOF

echo "OUT_ROOT=$OUT_ROOT"
echo "SUMMARY=$SUMMARY"
