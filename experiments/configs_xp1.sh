# CHUR-XP1 replication (dataset 04: ds002336 motor localizer, 9 subjects, registered volumes, pulse-aligned EEG).
# Use with CONFIGS_FILE=experiments/configs_xp1.sh and dataset 04.
source $(dirname ${BASH_SOURCE[0]})/configs.sh
X="-fmri_space mni"
CONFIGS=(
  "xp1_hetero|$HETERO $X"
  "xp1_point|$BACKBONE -padded -resolution_decoder 2 $X"
  "xp1_spectral|$BACKBONE $SPECTRAL -aleatoric_uncertainty -aleatoric_head spectral $X"
  "xp1_hetero_lr1e-4_b32|$HETERO $X -learning_rate 1e-4 -batch_size 32 -epochs 30"
  "xp1_point_fluct|$BACKBONE -padded -resolution_decoder 2 $X -target fluct"
  "xp1_inject1_hetero_fluct|$HETERO $X -target fluct -inject_snr 1"
  "xp1_inject1_point_fluct|$BACKBONE -padded -resolution_decoder 2 $X -target fluct -inject_snr 1"
)
SEEDS=(2 3 5)
ENSEMBLE="xp1_hetero|$HETERO $X"
ENSEMBLE_SEEDS=(2 3 5 7 11)
CONFIGS_POSTHOC=()
CONFIGS_LIU=()
