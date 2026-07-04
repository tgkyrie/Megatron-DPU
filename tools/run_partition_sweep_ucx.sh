#!/usr/bin/env bash
set -u

ROOT="/home/cmcc/CMCC/Megatron-DPU"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="$ROOT/docs/benchmark_runs/${STAMP}-partition-sweep-ucx"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/partition_sweep_summary.tsv"

WORKLOADS="${WORKLOADS_OVERRIDE:-dp_qwen3b dptp_qwen3_4b}"
PARTITIONS="${PARTITION_BYTES_LIST:-512K:524288 1M:1048576 2M:2097152 4M:4194304 6M:6291456 8M:8388608 12M:12582912 16M:16777216}"

echo -e "label\tworkload\tpartition_label\tpartition_bytes\trunner_out\tcase\tattempt\trail_variant\trun\tstatus\tps_port\tmaster_port\tmean_ms\tstd_ms\tn\tvalid_loss\ttest_loss\tmetric_worker_log\titer_file" > "$SUMMARY"

run_one() {
  local workload="$1"
  local label="$2"
  local bytes="$3"
  local run_label="${workload}_partition_${label}"
  local log="$OUT_ROOT/${run_label}.driver.log"
  echo "[$(date '+%H:%M:%S')] START workload=$workload label=$label bytes=$bytes"

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
      BYTEPS_PARTITION_BYTES_DEFAULT="$bytes" \
      MAX_RETRIES="${MAX_RETRIES_OVERRIDE:-2}" \
      MONITOR_TIMEOUT=1200 \
      NO_ITER_TIMEOUT=420 \
      WORKER_TIMEOUT=1800 \
      ROLE_TIMEOUT=1200 \
      python3 tools/run_megatron_ucx_semantic.py
  ) > "$log" 2>&1
  local rc=$?

  local runner_out
  runner_out="$(grep -o 'OUT=/.*' "$log" | tail -1 | sed 's/^OUT=//')"
  if [[ -z "$runner_out" || ! -f "$runner_out/summary.tsv" ]]; then
    echo -e "$run_label\t$workload\t$label\t$bytes\t${runner_out:-NA}\tNA\tNA\tNA\tNA\tDRIVER_RC_$rc\tNA\tNA\tNA\tNA\t0\tNA\tNA\t$log\tNA" >> "$SUMMARY"
    echo "[$(date '+%H:%M:%S')] FAIL workload=$workload label=$label rc=$rc log=$log"
    return 0
  fi

  tail -n +2 "$runner_out/summary.tsv" | awk -v run_label="$run_label" -v workload="$workload" -v label="$label" -v bytes="$bytes" -v out="$runner_out" -F'\t' 'BEGIN{OFS="\t"} {print run_label,workload,label,bytes,out,$0}' >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] DONE workload=$workload label=$label rc=$rc out=$runner_out"
}

for workload in $WORKLOADS; do
  for spec in $PARTITIONS; do
    IFS=: read -r label bytes <<< "$spec"
    run_one "$workload" "$label" "$bytes"
  done
done

cat > "$OUT_ROOT/run_notes.md" <<EOF
# Partition Sweep, UCX

Date: $(date -Is)

Workloads:

- $WORKLOADS

Partitions:

- $PARTITIONS

Summary:

- \`partition_sweep_summary.tsv\`
EOF

echo "OUT_ROOT=$OUT_ROOT"
echo "SUMMARY=$SUMMARY"
