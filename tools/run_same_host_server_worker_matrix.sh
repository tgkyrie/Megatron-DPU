#!/usr/bin/env bash
set -u

ROOT="/home/cmcc/CMCC/Megatron-DPU"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="$ROOT/docs/benchmark_runs/${STAMP}-same-host-server-worker"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/same_host_server_worker_summary.tsv"

WORKER_HOSTS="gpu01,gpu02,gpu03,gpu04,asus01,asus02,asus03,asus04"
WORKER_IPS="192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13,192.168.1.20,192.168.1.21,192.168.1.22,192.168.1.23"

echo -e "label\tworkload\tplacement\trunner_out\tcase\tattempt\trail_variant\trun\tstatus\tps_port\tmaster_port\tmean_ms\tstd_ms\tn\tvalid_loss\ttest_loss\tmetric_worker_log\titer_file" > "$SUMMARY"

run_one() {
  local workload="$1"
  local label="${workload}_same_host"
  local log="$OUT_ROOT/${label}.driver.log"
  echo "[$(date '+%H:%M:%S')] START label=$label workload=$workload"

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
      HOST_SERVERS_OVERRIDE="$WORKER_HOSTS" \
      HOST_SERVER_IPS_OVERRIDE="$WORKER_IPS" \
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
    echo -e "$label\t$workload\tsame-host\t${runner_out:-NA}\tNA\tNA\tNA\tNA\tDRIVER_RC_$rc\tNA\tNA\tNA\tNA\t0\tNA\tNA\t$log\tNA" >> "$SUMMARY"
    echo "[$(date '+%H:%M:%S')] FAIL label=$label rc=$rc log=$log"
    return 0
  fi

  tail -n +2 "$runner_out/summary.tsv" | awk -v label="$label" -v workload="$workload" -v out="$runner_out" -F'\t' 'BEGIN{OFS="\t"} {print label,workload,"same-host",out,$0}' >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] DONE label=$label rc=$rc out=$runner_out"
}

run_one dp_qwen3b
run_one tp_llama2_7b
run_one dptp_qwen3_4b

cat > "$OUT_ROOT/run_notes.md" <<EOF
# Same-host Server/Worker Placement Matrix

Date: $(date -Is)

This run starts Host-PS server roles on the same eight GPU worker hosts instead
of the non-worker CPU server pool. It is a placement/control experiment and
uses short runs by default.

Summary:

- \`same_host_server_worker_summary.tsv\`

Server hosts:

- $WORKER_HOSTS
EOF

echo "OUT_ROOT=$OUT_ROOT"
echo "SUMMARY=$SUMMARY"
