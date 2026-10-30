#!/bin/bash
# Reproduce the original implementation's tables (8 rows x 5 seeds) for one dataset with the unmodified code.
# Usage (inside run.sbatch): bash ../experiments/repro_legacy.sh <dataset> [parallel] [row-name regex]
D=$1; P=${2:-8}; FILTER=${3:-.}
OUT=$EEG_FMRI/../runs/repro_legacy/$D; mkdir -p $OUT
BASE="-fourier_features -random_fourier -topographical_attention -conditional_attention_style -verbose -gpu_mem 8000 -batch_size 4 -save_metrics -metrics_path $OUT/calhas"
declare -A CFG=(
  [calhas_original]="main.py metrics $D $BASE"
  [calhas_zerofill]="main.py metrics $D $BASE -padded -resolution_decoder 2"
  [calhas_flipout]="main.py metrics $D $BASE -aleatoric_uncertainty"
  [calhas_spectral]="main.py metrics $D $BASE -variational -variational_dependent_h 20 -aleatoric_uncertainty -variational_dist VonMises -variational_coefs 32,32,15 -resolution_decoder 2"
  [liu_original]="benchmarks/liu.py metrics $D -interval_eeg 1 -save -memory_limit 8000 -save_path $OUT/liu"
  [liu_zerofill]="benchmarks/liu.py metrics $D -interval_eeg 1 -save -memory_limit 8000 -save_path $OUT/liu -padded"
  [liu_flipout]="benchmarks/liu.py metrics $D -interval_eeg 1 -save -memory_limit 8000 -save_path $OUT/liu -variational"
  [liu_spectral]="benchmarks/liu.py metrics $D -interval_eeg 1 -save -memory_limit 8000 -save_path $OUT/liu -variationalDCT"
)
for name in "${!CFG[@]}"; do [[ $name =~ $FILTER ]] || continue; for s in 2 3 5 7 11; do
  while [ $(jobs -rp | wc -l) -ge $P ]; do sleep 20; done
  [ -f $OUT/$name.s$s.log ] && grep -q "^RMSE" $OUT/$name.s$s.log && continue
  (cd $( [[ ${CFG[$name]} == benchmarks* ]] && echo benchmarks || echo . ) && python -u ${CFG[$name]#benchmarks/} -seed $s) > $OUT/$name.s$s.log 2>&1 &
done; done
wait
grep -H -E "^(RMSE|SSIM)" $OUT/*.log
