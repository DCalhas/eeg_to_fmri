"""
Shared pieces for the cross-validated experiments (train_cv.py, evaluate_uq.py).

The model is built exactly as in main.py, so a configuration here maps onto the original implementation's settings.
"""
import os
import pickle

import numpy as np

from utils import data_utils


def add_model_args(parser):
	parser.add_argument('dataset', choices=['01', '02', '03', '04', '05'], help="Which dataset to load")
	parser.add_argument('-topographical_attention', action="store_true")
	parser.add_argument('-conditional_attention_style', action="store_true")
	parser.add_argument('-fourier_features', action="store_true")
	parser.add_argument('-random_fourier', action="store_true")
	parser.add_argument('-padded', action="store_true", help="Zero-filling DCT relaxation")
	parser.add_argument('-variational', action="store_true", help="Stochastic spectral coefficients")
	parser.add_argument('-variational_coefs', default=None, type=str)
	parser.add_argument('-variational_dependent_h', default=1, type=int)
	parser.add_argument('-variational_dist', default="Normal", type=str)
	parser.add_argument('-resolution_decoder', default=None, type=float)
	parser.add_argument('-aleatoric_uncertainty', action="store_true")
	parser.add_argument('-bnn_layer', default="legacy", choices=["legacy", "flipout", "none"], help="Decoder dense layer when -aleatoric_uncertainty is set")
	parser.add_argument('-aleatoric_head', default="voxel", choices=["voxel", "spectral", "evidential", "quantile"])
	parser.add_argument('-kl_scale', default=0., type=float, help="KL weight in units of 1/(N_train*voxels), i.e. 1 gives the ELBO matching the voxel-averaged likelihood")
	parser.add_argument('-mc_dropout', default=0., type=float)
	parser.add_argument('-spectral_dropout', default=0., type=float, help="MC dropout on the DCT coefficients (zero-filling decoder)")
	parser.add_argument('-epochs', default=10, type=int)
	parser.add_argument('-batch_size', default=4, type=int)
	parser.add_argument('-learning_rate', default=None, type=float, help="override the backbone's searched learning rate (training-recipe ablation)")
	parser.add_argument('-na_path_eeg', default=os.environ['EEG_FMRI']+"/na_models_eeg/na_specification_2", type=str)
	parser.add_argument('-na_path_fmri', default=os.environ['EEG_FMRI']+"/na_models_fmri/na_specification_2", type=str)
	parser.add_argument('-gpu_mem', default=8000, type=int)
	parser.add_argument('-interval_eeg', default=10, type=int)
	parser.add_argument('-channels', default="eeg", choices=["eeg", "ecg", "all"], help="EEG input channels (all = legacy setting, includes ECG)")
	parser.add_argument('-eeg_norm', default="time", choices=["time", "channels"], help="EEG STFT standardization (channels = legacy setting)")
	parser.add_argument('-target', default="run_zscore", choices=["run_zscore", "fluct", "fluct_resid"], help="fMRI target: z-scored run (legacy setting), voxel fluctuations, or fluctuations minus the training stimulus-locked average")
	parser.add_argument('-eeg_fmax', default=50., type=lambda v: None if v == "none" else float(v), help="Highest STFT frequency in Hz; none = the legacy 134 bins")
	parser.add_argument('-fmri_space', default="native", choices=["native", "mni"], help="native EPI geometry (legacy setting) or registered to MNI (review #1)")
	parser.add_argument('-inject_snr', default=0., type=float, help="positive control: inject a known C3/C4 mu-power signal into a fixed box (fluct targets)")
	parser.add_argument('-n_folds', default=5, type=int)
	parser.add_argument('-fold', default=0, type=int)
	parser.add_argument('-n_calib', default=2, type=int, help="Training subjects held out for conformal calibration")
	parser.add_argument('-cache_dir', default=os.environ.get('EEG_FMRI_CACHE'), type=str)
	parser.add_argument('-seed', default=42, type=int)
	parser.add_argument('-name', required=True, type=str, help="Configuration name, used for the output directory")
	parser.add_argument('-out_dir', required=True, type=str)
	parser.add_argument('-verbose', action="store_true")
	parser.add_argument('-debug_subset', default=0, type=int, help="Keep only the first N volumes of every subject (pre-submission smoke test)")
	return parser


def folds(dataset, n_folds, fold, n_calib, n_individuals=None):
	"""
	Subject-level folds with a fixed permutation (independent of the training seed),
	so every configuration and seed sees the same test and calibration subjects.
	Calibration subjects are taken from the start of the next fold.
	"""
	if(n_individuals is None):
		n_individuals=getattr(data_utils, "n_individuals_"+dataset)
	if(dataset=="04"):
		from utils import preprocess_data
		n_individuals=preprocess_data.XP1_N_SUBJECTS  #CHUR-XP1 replication subset (preprocess_data.xp1_motorloc)
	perm=np.random.RandomState(1234).permutation(n_individuals)
	groups=np.array_split(perm, n_folds)
	test=sorted(groups[fold].tolist())
	calib=sorted(groups[(fold+1)%n_folds][:n_calib].tolist())
	return test, calib


def run_dir(opt, name=None):
	return os.path.join(opt.out_dir, opt.dataset, opt.name if name is None else name, "fold"+str(opt.fold), "seed"+str(opt.seed))


def loss_fn(opt):
	from utils import losses_utils
	if(not opt.aleatoric_uncertainty):
		return losses_utils.LOSS_FNS["regression"]
	return {"evidential": losses_utils.Evidential_Loss, "quantile": losses_utils.Quantile_Loss}.get(opt.aleatoric_head, losses_utils.Laplacian_Loss)


def is_stochastic(opt):
	"""Whether repeated decoder calls give different outputs."""
	return opt.variational or opt.mc_dropout > 0 or opt.spectral_dropout > 0 or (opt.aleatoric_uncertainty and opt.bnn_layer != "none")


def build_model(opt, eeg_shape, fmri_shape, n_train=1):
	"""Same construction as main.py (original hyperparameters)."""
	if(opt.padded or opt.variational):
		assert opt.resolution_decoder is not None, "the DCT relaxation needs -resolution_decoder"
	if(opt.bnn_layer != "legacy" or opt.aleatoric_head != "voxel"):
		assert opt.aleatoric_uncertainty, "-bnn_layer/-aleatoric_head only act with -aleatoric_uncertainty"
	if(opt.bnn_layer == "flipout"):
		assert not opt.variational, "the stochastic spectral decoder has no dense layer to make Bayesian"
	assert opt.spectral_dropout == 0. or opt.padded, "spectral dropout acts on the zero-filled DCT coefficients"
	from models.eeg_to_fmri import EEG_to_fMRI

	learning_rate, weight_decay, kernel_size, stride_size, batch_size, latent_dimension, n_channels, max_pool, batch_norm, skip_connections, dropout, n_stacks, outfilter, local = (0.002980911194116198, 0.0, (9, 9, 4), (1, 1, 1), 4, (7, 7, 7), 4, True, True, True, True, 3, 1, True)

	with open(opt.na_path_eeg, "rb") as f:
		na_specification_eeg = pickle.load(f)
	with open(opt.na_path_fmri, "rb") as f:
		na_specification_fmri = pickle.load(f)

	variational_coefs=None
	if(opt.variational_coefs is not None):
		variational_coefs=tuple(map(int, opt.variational_coefs.split(",")))
	resolution_decoder=None
	if(opt.resolution_decoder is not None):
		resolution_decoder=(int(fmri_shape[1]/opt.resolution_decoder), int(fmri_shape[2]/opt.resolution_decoder), int(fmri_shape[3]/opt.resolution_decoder))

	model = EEG_to_fMRI(latent_dimension, eeg_shape[1:], na_specification_eeg, n_channels, weight_decay=weight_decay, consistency=False,
						batch_norm=batch_norm, local=local, fourier_features=opt.fourier_features, random_fourier=opt.random_fourier,
						conditional_attention_style=opt.conditional_attention_style, topographical_attention=opt.topographical_attention,
						conditional_attention_style_prior=False, skip_connections=skip_connections,
						organize_channels=False, inverse_DFT=opt.variational or opt.padded, DFT=opt.variational or opt.padded,
						variational_dist=opt.variational_dist, variational_iDFT=opt.variational, variational_coefs=variational_coefs,
						variational_iDFT_dependent=opt.variational_dependent_h>1, variational_iDFT_dependent_dim=opt.variational_dependent_h,
						aleatoric_uncertainty=opt.aleatoric_uncertainty, low_resolution_decoder=resolution_decoder is not None,
						variational_random_padding=False, resolution_decoder=resolution_decoder,
						bnn_layer=opt.bnn_layer, aleatoric_head=opt.aleatoric_head, mc_dropout=opt.mc_dropout, spectral_dropout=opt.spectral_dropout,
						kl_weight=opt.kl_scale/(n_train*int(np.prod(fmri_shape[1:]))), seed=None,
						fmri_args=(latent_dimension, fmri_shape[1:],
								kernel_size, stride_size, n_channels, max_pool, batch_norm, weight_decay, skip_connections,
								n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
	model.build(eeg_shape, fmri_shape)
	if(getattr(opt, "learning_rate", None) is not None):
		learning_rate = opt.learning_rate
	return model, learning_rate


def load_split(opt):
	from utils import preprocess_data
	test, calib = folds(opt.dataset, opt.n_folds, opt.fold, opt.n_calib)
	splits = preprocess_data.dataset_subjects(opt.dataset, test, calib, interval_eeg=opt.interval_eeg, cache_dir=opt.cache_dir,
											channels=opt.channels, eeg_norm=opt.eeg_norm, eeg_fmax=opt.eeg_fmax, target=opt.target,
											fmri_space=getattr(opt, "fmri_space", "native"), inject_snr=getattr(opt, "inject_snr", 0.), verbose=opt.verbose)
	if(getattr(opt, "debug_subset", 0)):
		for name, split in splits.items():
			if(split is not None and name not in ("mask", "inject_mask")):
				keep = np.concatenate([np.where(split[2] == s)[0][:opt.debug_subset] for s in np.unique(split[2])])
				splits[name] = tuple(a[keep] for a in split)
	return splits, test, calib
