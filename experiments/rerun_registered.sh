#!/bin/bash
# Rerun every registered-volume analysis after the 2026-10-31 registration fix (register_xp2_bold.py -max_refine_mm).
# Usage on the Frida login node (file management + sbatch only), after unc_reg2 / unc_xp1reg2 passed QC:
#   bash experiments/rerun_registered.sh swap     # old registrations and outputs -> slid_20261031/, new ones in place
#   bash experiments/rerun_registered.sh lists    # write the command files
#   bash experiments/rerun_registered.sh submit   # GPU (dev, A100) and CPU (amd) jobs, with the later stages chained
set -e
R=/shared/home/david.calhas/eeg_to_fmri/uncertainty
B=$R/slid_20261031
cd $R/repo
case $1 in
swap)
  mkdir -p $B/cache $B/runs/cv/05 $B/runs/cv/04
  mv $R/cache/xp2_mni $B/cache/ && mv $R/cache/xp2_mni_v2 $R/cache/xp2_mni
  mv $R/cache/xp1_mni $B/cache/ && mv $R/cache/xp1_mni_v2 $R/cache/xp1_mni
  for f in $R/cache/04_xp1_*; do [ -e "$f" ] && mv "$f" $B/cache/; done
  for d in $R/runs/cv/05/mni_* $R/runs/cv/05/probe_mni_fluct*; do [ -e "$d" ] && mv "$d" $B/runs/cv/05/; done
  for d in $R/runs/cv/04/*; do [ -e "$d" ] && mv "$d" $B/runs/cv/04/; done
  for d in ridge_roi_mni ridge_roi_xp1 mehta_sgl region_maps reeval_xp1 mni_report.json mni_fluct_report.json \
           realfix_mni_fluct_report.json xp1_report.json xp1_fluct_report.json xp1_effect_report.json; do
    if [ -e $R/runs/$d ]; then mv $R/runs/$d $B/runs/; fi
  done
  mkdir -p $R/runs/ridge_roi_mni $R/runs/ridge_roi_xp1 $R/runs/mehta_sgl $R/runs/region_maps $R/runs/reeval_xp1
  echo SWAPPED;;
lists)
  #stage 1: training (with its null evaluations), EEG-free references and linear probes; an A100 holds ~8 of these
  CONFIGS_FILE=experiments/configs_review.sh GPU_MEM=7000 bash experiments/make_commands.sh 05 5 '^mni_' > $R/rr_05a.txt
  #EEG-free references on registered volumes: the exact commands of the first run (configs.sh fixes BASELINE_FLAGS)
  /usr/bin/grep -h "baseline_eeg_free.py 05 .*-fmri_space mni" $R/rev*.txt | sort -u >> $R/rr_05a.txt
  for k in 0 1 2 3 4; do
    echo "python -u linear_probe.py 05 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -target fluct -fmri_space mni -name probe_mni_fluct -fold $k -n_folds 5 -out_dir $R/runs/cv -cache_dir $R/cache -gpu_mem 9000 > $R/runs/probe_mni_fluct_f$k.log 2>&1" >> $R/rr_05a.txt
  done
  CONFIGS_FILE=experiments/configs_xp1.sh GPU_MEM=6000 bash experiments/make_commands.sh 04 5 '^xp1_' > $R/rr_04a.txt
  /usr/bin/grep -hE "baseline_eeg_free|linear_probe" $R/xp1b.txt | sort -u >> $R/rr_04a.txt
  printf '%s\n' "python -u mehta_sgl.py -features eeg -gpu_mem 15000 -out $R/runs/mehta_sgl > $R/runs/mehta_sgl/eeg.log 2>&1" \
                "python -u mehta_sgl.py -features ecg -gpu_mem 15000 -out $R/runs/mehta_sgl > $R/runs/mehta_sgl/ecg.log 2>&1" > $R/rr_mehta.txt
  #ridge (CPU): the exact commands of the first run, same output directories
  /usr/bin/grep -h "ridge_roi.py" $R/*.txt $R/*.sh | /usr/bin/grep -E "roi_source mni|dataset xp1" | /usr/bin/grep -v smoke \
    | sed -E 's# > .*##; s#^.*python -u #python -u #' | sort -u > $R/rr_ridge_raw.txt
  awk '{o=$0; sub(/.*-out /,"",o); t=$0; sub(/python -u ridge_roi.py /,"",t); sub(/ -n_surrogates.*/,"",t); gsub(/[- +]+/,"_",t); print $0" > "o"/"t".log 2>&1"}' \
    $R/rr_ridge_raw.txt > $R/rr_ridge.txt
  wc -l $R/rr_05a.txt $R/rr_04a.txt $R/rr_mehta.txt $R/rr_ridge.txt;;
submit)
  #A100-SXM4-40GB nodes: 7 models x 5 GB (8 x 7 GB ran out of memory on 2026-10-31)
  sed -i 's/-gpu_mem [0-9]*/-gpu_mem 5000/' $R/rr_05a.txt $R/rr_04a.txt
  a=$(sbatch --parsable -J unc_rr05a experiments/run.sbatch bash ../experiments/run_commands.sh $R/rr_05a.txt 7)
  #CHUR-XP1 timed out at 12 h on one GPU in the first run: two GPUs, half of the commands each, after a one-time cache build
  rm -f $R/rr_04a_part*; split -n l/2 -d $R/rr_04a.txt $R/rr_04a_part
  #train_cv/baseline_eeg_free and ridge_roi require a GPU (utils/tf_config asserts it), so these run through run.sbatch
  /usr/bin/grep -m1 'baseline_eeg_free.py 04 .*-target run_zscore .*-fold 0 ' $R/rr_04a.txt > $R/rr_cache04.txt
  k=$(sbatch --parsable -J unc_rrcache04 --time=3:00:00 experiments/run.sbatch bash ../experiments/run_commands.sh $R/rr_cache04.txt 1)
  b=$(sbatch --parsable -J unc_rr04a --dependency=afterok:$k experiments/run.sbatch bash ../experiments/run_commands.sh $R/rr_04a_part00 7)
  b2=$(sbatch --parsable -J unc_rr04b --dependency=afterok:$k experiments/run.sbatch bash ../experiments/run_commands.sh $R/rr_04a_part01 7)
  c=$(sbatch --parsable -J unc_rrridge experiments/run.sbatch bash ../experiments/run_commands.sh $R/rr_ridge.txt 4)
  #stage 2: retry any training that failed, then ensembles and the XP1 re-evaluation
  e=$(sbatch --parsable -J unc_rrens --dependency=afterany:$a:$b:$b2 experiments/run.sbatch bash ../experiments/rerun_registered.sh ens)
  #stage 3 on a CPU node: reports and significance maps, the exact commands of the first run
  cat > $R/rr_reports.sh <<EOS
#!/bin/bash
cd $R/repo/src
P=~/eeg_to_fmri/envs/tf29/bin/python
\$P -u gate_report.py $R/runs/cv/05 -configs mni_hetero,mni_spectral,mni_hetero_ensemble -prefix mni_ -out $R/runs/mni_report.json > $R/runs/rep_mni.log 2>&1
\$P -u gate_report.py $R/runs/cv/05 -configs mni_hetero_fluct -prefix mni_ -suffix _fluct -out $R/runs/mni_fluct_report.json > $R/runs/rep_mni_fluct.log 2>&1
\$P -u gate_report.py $R/runs/cv/05 -configs mni_hetero_lr1e-4_b32_fluct -prefix mni_ -suffix _fluct -out $R/runs/realfix_mni_fluct_report.json > $R/runs/rep_realfix_mni.log 2>&1
bash $R/rep_xp1.sh
\$P -u effect_report.py $R/runs/cv/04 -configs xp1_hetero,xp1_point,xp1_spectral,xp1_hetero_lr1e-4_b32,xp1_hetero_ensemble -prefix mni_ -members xp1_hetero -out $R/runs/xp1_effect_report.json > $R/runs/xp1_effect.log 2>&1
bash $R/region.sh > $R/runs/region_maps/region.log 2>&1
echo RR_REPORTS_DONE
EOS
  m=${MEHTA_JOB:-}
  f=$(sbatch --parsable -J unc_rrrep -p amd --dependency=afterany:$e:$c${m:+:$m} --cpus-per-task=8 --mem=96G --time=4:00:00 --output=$R/logs/slurm-%j.out $R/rr_reports.sh)
  echo "submitted cache04:$k 05:$a 04:$b,$b2 ridge:$c ensembles+retry:$e reports:$f (after mehta ${m:-none})";;
ens)
  #runs inside run.sbatch: first retry any training that failed (make_commands lists only runs without their null
  #evaluations), then the ensembles, whose commands can only be written once their members exist
  CONFIGS_FILE=experiments/configs_review.sh GPU_MEM=5000 bash experiments/make_commands.sh 05 5 '^mni_(hetero|spectral|hetero_fluct|hetero_lr1e-4_b32_fluct)$' > $R/rr_retry.txt
  #a fold with a single test subject has no swap null, so make_commands lists its finished runs again: keep only runs
  #without their shift-null evaluation
  CONFIGS_FILE=experiments/configs_xp1.sh GPU_MEM=5000 bash experiments/make_commands.sh 04 5 '^xp1_[a-z0-9_.-]*$' | /usr/bin/grep -v evaluate_ensemble \
    | while read -r l; do d=$(sed -E 's#^mkdir -p ([^ ]+) .*#\1#' <<<"$l"); [ -f $d/eval_null_shift.npz ] || echo "$l"; done >> $R/rr_retry.txt || true
  (cd src; wc -l $R/rr_retry.txt; bash ../experiments/run_commands.sh $R/rr_retry.txt 4 || true)
  CONFIGS_FILE=experiments/configs_review.sh bash experiments/make_commands.sh 05 5 '^mni_hetero_ensemble$' > $R/rr_ens.txt
  CONFIGS_FILE=experiments/configs_xp1.sh bash experiments/make_commands.sh 04 5 '^xp1_hetero_ensemble$' >> $R/rr_ens.txt
  cd src; wc -l $R/rr_ens.txt
  bash ../experiments/run_commands.sh $R/rr_ens.txt 2 || true  #ensembles reserve 12 GB each: 2 per 40 GB A100
  bash ../experiments/run_commands.sh $R/xp1d.txt 1;;
esac
