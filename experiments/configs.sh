# UQ grid of the journal version (uncertainty-evaluation framing), Calhas et al. backbone, run-z-score target.
# Sourced by make_commands.sh; each entry is "name|flags". Every method uses the same folds, preprocessing
# (EEG channels only, 1-50 Hz STFT, time-normalized) and evaluation, and is compared against the EEG-free
# predictive distributions of baseline_eeg_free.py.
BACKBONE="-fourier_features -random_fourier -topographical_attention -conditional_attention_style"
HETERO="$BACKBONE -padded -resolution_decoder 2 -aleatoric_uncertainty -bnn_layer none -aleatoric_head spectral"
SPECTRAL="-variational -variational_dependent_h 20 -variational_dist VonMises -variational_coefs 32,32,15 -resolution_decoder 2"
CONFIGS=(
  # stochastic spectral decoder: original voxel scale head (ablation) and with the smooth aleatoric head
  "spectral_voxelhead|$BACKBONE $SPECTRAL -aleatoric_uncertainty"
  "spectral|$BACKBONE $SPECTRAL -aleatoric_uncertainty -aleatoric_head spectral"
  # Bayesian weights: the legacy layer (single shared epsilon, no KL) and corrected Flipout with the ELBO KL
  "legacy_dense|$BACKBONE -aleatoric_uncertainty"
  "flipout|$BACKBONE -aleatoric_uncertainty -bnn_layer flipout -kl_scale 1 -aleatoric_head spectral"
  "zerofill_flipout|$BACKBONE -padded -resolution_decoder 2 -aleatoric_uncertainty -bnn_layer flipout -kl_scale 1 -aleatoric_head spectral"
  # aleatoric only, sampling-based and distribution-free methods on the same relaxed decoder
  "zerofill_hetero|$HETERO"
  "zerofill_mcdropout|$HETERO -mc_dropout 0.2"
  "zerofill_specdropout|$HETERO -spectral_dropout 0.1"
  "zerofill_evidential|$BACKBONE -padded -resolution_decoder 2 -aleatoric_uncertainty -bnn_layer none -aleatoric_head evidential"
  "zerofill_cqr|$BACKBONE -padded -resolution_decoder 2 -aleatoric_uncertainty -bnn_layer none -aleatoric_head quantile"
  "zerofill|$BACKBONE -padded -resolution_decoder 2"
  # input confound controls (audit A1/A2/A14)
  "zerofill_hetero_ecg_only|$HETERO -channels ecg"
  "zerofill_hetero_minprep|$HETERO -channels all -eeg_norm channels -eeg_fmax none"
)
SEEDS=(2 3 5)
# zerofill_hetero is also the base of the post-hoc methods and of the 5-member deep ensemble
ENSEMBLE_SEEDS=(2 3 5 7 11)
CONFIGS_LIU=()
# post-hoc methods on the trained zerofill_hetero weights (generated only once those weights exist)
CONFIGS_POSTHOC=(
  "zerofill_swag|$HETERO -init_from zerofill_hetero -swag_epochs 3"
  "zerofill_laplace|$HETERO -init_from zerofill_hetero -laplace"
)
ENSEMBLE="zerofill_hetero|$HETERO"
# EEG-free predictive distributions (template, own-mean oracle, stimulus-locked), one set per fold
BASELINE_FLAGS="$BACKBONE"
