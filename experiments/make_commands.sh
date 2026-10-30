#!/bin/bash
# Writes one command per (config, fold, seed) without an eval.npz yet (and null evaluations for the trained ones).
# Usage: GPU_MEM=6000 bash experiments/make_commands.sh <dataset> <n_folds> [config-name-regex] > cmds.txt
# Run the commands with experiments/run_commands.sh inside run.sbatch.
D=$1; K=${2:-5}; FILTER=${3:-.}; MEM=${GPU_MEM:-6000}
ROOT=/shared/home/david.calhas/eeg_to_fmri/uncertainty
source ${CONFIGS_FILE:-$(dirname $0)/configs.sh}
O="-n_folds $K -out_dir $ROOT/runs/cv -cache_dir $ROOT/cache -gpu_mem $MEM"
for entry in "${CONFIGS[@]}"; do
  name=${entry%%|*}; flags=${entry#*|}
  [[ $name =~ $FILTER ]] || continue
  seeds=("${SEEDS[@]}"); [ "$name" = "${ENSEMBLE%%|*}" ] && seeds=("${ENSEMBLE_SEEDS[@]}")
  for fold in $(seq 0 $((K-1))); do for s in "${seeds[@]}"; do
    out=$ROOT/runs/cv/$D/$name/fold$fold/seed$s
    [ -f $out/eval_null_swap.npz ] && continue
    echo "mkdir -p $out && python -u train_cv.py $D $flags -name $name -fold $fold -seed $s $O > $out/log.txt 2>&1"
  done; done
done
for entry in "${CONFIGS_LIU[@]}"; do
  name=${entry%%|*}; flags=${entry#*|}
  [[ $name =~ $FILTER ]] || continue
  for fold in $(seq 0 $((K-1))); do for s in "${SEEDS[@]}"; do
    out=$ROOT/runs/cv/$D/$name/fold$fold/seed$s
    [ -f $out/eval_null_swap.npz ] && continue
    echo "mkdir -p $out && python -u train_cv_liu.py $D $flags -name $name -fold $fold -seed $s $O > $out/log.txt 2>&1"
  done; done
done
for entry in "${CONFIGS_POSTHOC[@]}"; do
  name=${entry%%|*}; flags=${entry#*|}; base=$(sed -E 's/.*-init_from ([^ ]+).*/\1/' <<<"$flags")
  [[ $name =~ $FILTER ]] || continue
  for fold in $(seq 0 $((K-1))); do for s in "${SEEDS[@]}"; do
    out=$ROOT/runs/cv/$D/$name/fold$fold/seed$s
    [ -f $out/eval_null_swap.npz ] && continue
    [ -f $ROOT/runs/cv/$D/$base/fold$fold/seed$s/weights.index ] || continue
    echo "mkdir -p $out && python -u train_cv.py $D $flags -name $name -fold $fold -seed $s $O > $out/log.txt 2>&1"
  done; done
done
name=${ENSEMBLE%%|*}; flags=${ENSEMBLE#*|}
if [[ ${name}_ensemble =~ $FILTER ]]; then for fold in $(seq 0 $((K-1))); do
  out=$ROOT/runs/cv/$D/${name}_ensemble/fold$fold/seed0
  [ -f $out/eval.npz ] && continue
  ok=1; for s in "${ENSEMBLE_SEEDS[@]}"; do [ -f $ROOT/runs/cv/$D/$name/fold$fold/seed$s/weights.index ] || ok=0; done
  [ $ok = 1 ] || continue
  echo "mkdir -p $out && python -u evaluate_ensemble.py $D $flags -name $name -members $(IFS=,; echo "${ENSEMBLE_SEEDS[*]}") -fold $fold -n_folds $K -out_dir $ROOT/runs/cv -cache_dir $ROOT/cache -gpu_mem 12000 > $out/log.txt 2>&1"
done; fi
# EEG-free predictive distributions (template, own-mean oracle, stimulus-locked), one set per fold
if [[ eeg_free =~ $FILTER ]]; then for fold in $(seq 0 $((K-1))); do
  [ -f $ROOT/runs/cv/$D/${BASELINE_PREFIX}stimulus_locked${BASELINE_SUFFIX}/fold$fold/seed0/eval.npz ] && continue
  out=$ROOT/runs/cv/$D/${BASELINE_PREFIX}eeg_free_template${BASELINE_SUFFIX}/fold$fold/seed0
  echo "mkdir -p $out && python -u baseline_eeg_free.py $D $BASELINE_FLAGS -name eeg_free -fold $fold $O > $out/log.txt 2>&1"
done; fi
