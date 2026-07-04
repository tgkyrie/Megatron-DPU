#!/usr/bin/env bash
set -u

ROOT="/home/cmcc/CMCC/Megatron-DPU"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="$ROOT/docs/benchmark_runs/${STAMP}-dptp-partition-sweep-ucx"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/dptp_partition_sweep_summary.tsv"

echo -e "label\tworkload\tpartition_label\tpartition_bytes\trunner_out\tcase\tattempt\trail_variant\trun\tstatus\tps_port\tmaster_port\tmean_ms\tstd_ms\tn\tvalid_loss\ttest_loss\tmetric_worker_log\titer_file" > "$SUMMARY"

run_one() {
  local label="$1"
  local bytes="$2"
  local run_label="partition_${label}"
  local log="$OUT_ROOT/${run_label}.driver.log"
  echo "[$(date '+%H:%M:%S')] START label=$run_label bytes=$bytes"

  (
    cd "$ROOT" || exit 1
    env \
      WORKLOAD=dptp_qwen3_4b \
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
    echo -e "$run_label\tdptp_qwen3_4b\t$label\t$bytes\t${runner_out:-NA}\tNA\tNA\tNA\tNA\tDRIVER_RC_$rc\tNA\tNA\tNA\tNA\t0\tNA\tNA\t$log\tNA" >> "$SUMMARY"
    echo "[$(date '+%H:%M:%S')] FAIL label=$run_label rc=$rc log=$log"
    return 0
  fi

  tail -n +2 "$runner_out/summary.tsv" | awk -v run_label="$run_label" -v label="$label" -v bytes="$bytes" -v out="$runner_out" -F'\t' 'BEGIN{OFS="\t"} {print run_label,"dptp_qwen3_4b",label,bytes,out,$0}' >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] DONE label=$run_label rc=$rc out=$runner_out"
}

if [[ -n "${PARTITION_BYTES_LIST:-}" ]]; then
  for spec in ${PARTITION_BYTES_LIST}; do
    IFS=: read -r label bytes <<< "$spec"
    run_one "$label" "$bytes"
  done
else
  run_one 512K 524288
  run_one 1M 1048576
  run_one 2M 2097152
  run_one 4M 4194304
  run_one 6M 6291456
  run_one 8M 8388608
  run_one 12M 12582912
  run_one 16M 16777216
fi

cat > "$OUT_ROOT/run_notes.md" <<EOF
# TP+DP Partition Sweep, UCX

Date: $(date -Is)

This run sweeps BytePS partition size for the TP+DP Qwen3-4B workload using
short runs by default.

Summary:

- \`dptp_partition_sweep_summary.tsv\`
EOF

echo "OUT_ROOT=$OUT_ROOT"
echo "SUMMARY=$SUMMARY"
