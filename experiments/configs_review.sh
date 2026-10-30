# Runs requested by the mock review (2026-10-28). Use with CONFIGS_FILE=experiments/configs_review.sh.
#  #1 registered inputs: the main configurations on MNI-registered, motion-corrected volumes (-fmri_space mni)
#  #2b positive control: a known C3/C4 mu-power signal injected into a fixed box of the fluct target, three SNRs
source $(dirname ${BASH_SOURCE[0]})/configs.sh
CONFIGS=(
  "mni_hetero|$HETERO -fmri_space mni"
  "mni_spectral|$BACKBONE $SPECTRAL -aleatoric_uncertainty -aleatoric_head spectral -fmri_space mni"
  "mni_hetero_fluct|$HETERO -fmri_space mni -target fluct"
  "inject0.1_hetero_fluct|$HETERO -target fluct -inject_snr 0.1"
  "inject0.3_hetero_fluct|$HETERO -target fluct -inject_snr 0.3"
  "inject1_hetero_fluct|$HETERO -target fluct -inject_snr 1"
  # training-recipe ablation on the strongest injection (the ladder lost the signal only with the deep loop)
  "inject1_point_fluct|$BACKBONE -padded -resolution_decoder 2 -target fluct -inject_snr 1"
  "inject1_hetero_lr1e-4_b32_fluct|$HETERO -target fluct -inject_snr 1 -learning_rate 1e-4 -batch_size 32 -epochs 30"
  "inject1_point_lr1e-4_b32_fluct|$BACKBONE -padded -resolution_decoder 2 -target fluct -inject_snr 1 -learning_rate 1e-4 -batch_size 32 -epochs 30"
  # the recipes that recover the injected signal, on the real data (review follow-up: would a sensitive model find EEG information?)
  "real_hetero_lr1e-4_b32|$HETERO -learning_rate 1e-4 -batch_size 32 -epochs 30"
  "real_hetero_lr1e-4_b32_fluct|$HETERO -target fluct -learning_rate 1e-4 -batch_size 32 -epochs 30"
  "real_point_fluct|$BACKBONE -padded -resolution_decoder 2 -target fluct"
  "mni_hetero_lr1e-4_b32_fluct|$HETERO -fmri_space mni -target fluct -learning_rate 1e-4 -batch_size 32 -epochs 30"
)
SEEDS=(2 3 5)
ENSEMBLE="mni_hetero|$HETERO -fmri_space mni"
ENSEMBLE_SEEDS=(2 3 5 7 11)
CONFIGS_POSTHOC=()
CONFIGS_LIU=()
