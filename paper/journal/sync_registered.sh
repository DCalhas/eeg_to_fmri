#!/bin/bash
# Copy the registered-volume results of the 2026-11-01 rerun (rerun_registered.sh) from Frida into results/.
# JSON summaries and the significance-map npz only; run from paper/journal.
set -e
F=login-frida:/shared/home/david.calhas/eeg_to_fmri/uncertainty/runs
J="--include=*/ --include=*.json --exclude=*"
for c in $(ssh login-frida 'ls -d /shared/home/david.calhas/eeg_to_fmri/uncertainty/runs/cv/05/mni_*' | xargs -n1 basename); do
  rsync -a --prune-empty-dirs $J $F/cv/05/$c/ results/summaries/$c/
done
rsync -a --prune-empty-dirs $J $F/cv/05/probe_mni_fluct/ results/probe/probe_mni_fluct/
for c in $(ssh login-frida 'ls -d /shared/home/david.calhas/eeg_to_fmri/uncertainty/runs/cv/04/*/' | xargs -n1 basename); do
  case $c in probe_*) rsync -a --prune-empty-dirs $J $F/cv/04/$c/ results/probe_xp1/$c/;; *) rsync -a --prune-empty-dirs $J $F/cv/04/$c/ results/summaries_xp1/$c/;; esac
done
rsync -a $F/ridge_roi_mni/*.json results/ridge_mni/
rsync -a $F/ridge_roi_xp1/*.json results/ridge_xp1/
rsync -a $F/mehta_sgl/*.json results/mehta/
for r in mni_report mni_fluct_report realfix_mni_fluct_report xp1_report xp1_fluct_report xp1_effect_report; do
  rsync -a $F/$r.json results/ 2>/dev/null || echo "not yet: $r.json"
done
rsync -a $F/region_maps/*.npz results/region_maps/ 2>/dev/null || echo "not yet: region maps"
