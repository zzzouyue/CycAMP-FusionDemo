#!/bin/bash
# HighFold2 batch runner for CycAMP-FusionDemo.
# Consumes manifest.csv produced by scripts/build_highfold_inputs.py.
# One prediction.py invocation per peptide because disulfide pairs differ per row.
# Resume-safe: a peptide whose output dir contains .done is skipped.
#
# Usage on server:
#   mkdir -p /root/autodl-tmp/highfold_cycamp
#   # upload manifest.csv and fasta/ into that directory first
#   cd /root/autodl-tmp/highfold_cycamp
#   nohup bash run_highfold_batch.sh > batch_nohup.log 2>&1 &
set -u

HF_DIR=/root/autodl-tmp/HighFold2-main
PY=/root/autodl-tmp/envs/highfold2/bin/python
# tleap (AmberTools) lives in the env's bin; prediction.py's amber relax
# invokes it by bare name via subprocess, so it must be on PATH.
export PATH="/root/autodl-tmp/envs/highfold2/bin:$PATH"
# Override RUN_DIR when launching a corrected rerun so prior results remain
# immutable, for example: RUN_DIR=/root/autodl-tmp/highfold_cycamp_corrected.
RUN_DIR=${RUN_DIR:-/root/autodl-tmp/highfold_cycamp}
MANIFEST="$RUN_DIR/manifest.csv"
OUT_ROOT="$RUN_DIR/results"
SEED=42
LOG="$RUN_DIR/batch_progress.log"

mkdir -p "$OUT_ROOT"
cd "$HF_DIR" || exit 1

total=$(($(wc -l < "$MANIFEST") - 1))
done_count=0
fail_count=0
echo "BATCH_START $(date +%F' '%T) total=$total" >> "$LOG"

tail -n +2 "$MANIFEST" | while IFS=, read -r pid ffasta seq slen ctype fcyc fnc dsb ndsb label rgroup; do
  outdir="$OUT_ROOT/$pid"
  if [ -f "$outdir/.done" ]; then
    echo "SKIP $pid" >> "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  start=$(date +%s)

  # HighFold2 documents every disulfide-containing peptide as cyclic.  The
  # manifest records DSB positions as one-based residue positions, while
  # --index-ss requires a zero-based chain index followed by zero-based
  # residue positions.  Do not substitute --disulfide-bond-pairs here: it
  # does not provide Amber relaxation with the explicit S-S topology.
  set -- --model-type alphafold2_multimer_v3 \
         --msa-mode mmseqs2_uniref_env \
         --num-models 5 \
         --amber --num-relax 1 \
         --random-seed "$SEED" \
         --rank plddt \
         "$RUN_DIR/$ffasta" "$outdir" \
         --flag-cyclic-peptide 1
  if [ "$fcyc" = "1" ]; then
    set -- "$@" --flag-nc 1
  fi
  if [ -n "$dsb" ]; then
    read -r -a dsb_indices <<< "$dsb"
    if (( ${#dsb_indices[@]} % 2 != 0 )); then
      echo "FAIL $pid invalid_odd_dsb_index_count" >> "$LOG"
      continue
    fi
    for ((pair_index = 0; pair_index < ${#dsb_indices[@]}; pair_index += 2)); do
      left=${dsb_indices[pair_index]}
      right=${dsb_indices[pair_index + 1]}
      if (( left < 1 || right < 1 )); then
        echo "FAIL $pid invalid_one_based_dsb_index:${left}-${right}" >> "$LOG"
        continue 2
      fi
      set -- "$@" --index-ss 0 "$((left - 1))" "$((right - 1))"
    done
  fi

  if "$PY" prediction.py "$@" > "$outdir/run.log" 2>&1; then
    touch "$outdir/.done"
    echo "DONE $pid $(( $(date +%s) - start ))s" >> "$LOG"
  else
    echo "FAIL $pid $(( $(date +%s) - start ))s log=$outdir/run.log" >> "$LOG"
  fi
done

echo "ALL_FINISHED $(date +%F' '%T)" >> "$LOG"
