"""
Cross-validated training and evaluation of the Liu et al. (2019) benchmark, same folds as train_cv.py.

Pairs are built per subject block here: Liu_et_al.load_data allocates n_volumes rows per subject but fills
n_volumes-1, so its contiguous train/test slices put the first instances of a test subject into training
and score uninitialized rows as test volumes.

python train_cv_liu.py 05 -name liu_zerofill -padded -fold 0 -seed 2 -out_dir RUNS
"""
import argparse
import json
import os
import time

import numpy as np

import experiment

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['01', '02', '03', '04', '05'])
parser.add_argument('-variational', action="store_true", help="Flipout temporal convolution")
parser.add_argument('-variationalDCT', action="store_true", help="Stochastic spectral coefficients")
parser.add_argument('-padded', action="store_true", help="Zero-filling DCT relaxation")
parser.add_argument('-aleatoric', default=None, type=int, choices=[0, 1], help="Laplace head; default as the original implementation (only with -variational)")
parser.add_argument('-kl_scale', default=None, type=float, help="KL weight in units of 1/(N_train*voxels); default as the original implementation (unscaled Flipout KL)")
parser.add_argument('-channels', default="eeg", choices=["eeg", "ecg", "all"])
parser.add_argument('-eeg_norm', default="time", choices=["time", "channels"])
parser.add_argument('-lag', default=5, type=int, help="Haemodynamic lag in TRs: EEG at t predicts the volume at t+lag (original implementation: 0)")
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-batch_size', default=2, type=int)
parser.add_argument('-gpu_mem', default=4000, type=int)
parser.add_argument('-n_folds', default=5, type=int)
parser.add_argument('-fold', default=0, type=int)
parser.add_argument('-n_calib', default=2, type=int)
parser.add_argument('-cache_dir', default=os.environ.get('EEG_FMRI_CACHE'), type=str)
parser.add_argument('-seed', default=42, type=int)
parser.add_argument('-T', default=50, type=int)
parser.add_argument('-name', required=True)
parser.add_argument('-out_dir', required=True)
opt = parser.parse_args()

import sys
sys.path.append("benchmarks")
from utils import data_utils, fmri_utils, eeg_utils, tf_config, train
import tensorflow as tf
import liu
import evaluate_uq

out = experiment.run_dir(opt)
os.makedirs(out, exist_ok=True)
tf_config.set_seed(seed=opt.seed)
tf_config.setup_tensorflow(device="GPU", memory_limit=opt.gpu_mem)

n_individuals = getattr(data_utils, "n_individuals_"+opt.dataset)
n_volumes = getattr(fmri_utils, "n_volumes_"+opt.dataset)
from utils import preprocess_data
cache_file = None if opt.cache_dir is None else os.path.join(opt.cache_dir, opt.dataset+"_liu_n"+str(n_individuals)+"_"+opt.channels+"_"+opt.eeg_norm+".npz")
if(cache_file is not None and os.path.exists(cache_file)):
	with np.load(cache_file) as f:
		eeg_all, fmri_all = f["eeg"], f["fmri"]
else:
	with tf.device('/CPU:0'):
		eeg_all, fmri_all, _ = data_utils.get_data(range(n_individuals), dataset=opt.dataset, raw_eeg=True, standardize_fmri=True,
												ind_volume_fit=False, iqr_outlier=False, raw_eeg_resample=True,
												bold_shift=getattr(fmri_utils, "bold_shift_"+opt.dataset),
												eeg_resample=getattr(fmri_utils, "TR_"+opt.dataset), fmri_resolution_factor=1,
												channel_types=preprocess_data.CHANNELS[opt.channels], eeg_norm=opt.eeg_norm)
	eeg_all, fmri_all = eeg_all.astype('float32'), fmri_all.astype('float32')
	if(cache_file is not None):
		os.makedirs(opt.cache_dir, exist_ok=True)
		np.savez(cache_file, eeg=eeg_all, fmri=fmri_all)
assert eeg_all.shape[0] == fmri_all.shape[0] == n_individuals*n_volumes, (eeg_all.shape, fmri_all.shape)

test, calib = experiment.folds(opt.dataset, opt.n_folds, opt.fold, opt.n_calib)
def block(individuals):
	"""(eeg (N, 1, channels), fmri (N, X, Y, Z, 1), subject ids): the EEG resampled at TR t paired with volume t+lag.
	The resampled input is one sample per channel and TR, as in the original benchmark setting (disclosed)."""
	if(not len(individuals)):
		return None
	n = n_volumes-opt.lag
	eeg = np.concatenate([eeg_all[i*n_volumes:i*n_volumes+n] for i in individuals])[:, None, :]
	fmri = np.concatenate([fmri_all[i*n_volumes+opt.lag:(i+1)*n_volumes] for i in individuals])[..., None]
	return eeg, fmri, np.repeat(individuals, n)
train_subjects = [i for i in range(n_individuals) if i not in test+calib]
splits = {"train": block(train_subjects), "calib": block(calib), "test": block(test)}
print("I: train subjects", train_subjects, "calib", calib, "test", test, flush=True)

eeg_train, fmri_train, _ = splits["train"]
kl_weight = None if opt.kl_scale is None else opt.kl_scale/(eeg_train.shape[0]*int(np.prod(fmri_train.shape[1:])))
with tf.device('/CPU:0'):
	model = liu.Liu_et_al((eeg_train.shape[-1], 1), getattr(fmri_utils, "fmri_shape_"+opt.dataset)+(1,),
						variational=opt.variational, variationalDCT=opt.variationalDCT, padded=opt.padded,
						variational_coefs=(32, 32, 15), variational_dependent=True, variational_dependent_h=20, variational_dist="VonMises",
						aleatoric=None if opt.aleatoric is None else bool(opt.aleatoric), kl_weight=kl_weight)
	optimizer = tf.keras.optimizers.Adam(0.001)
	train_set = tf.data.Dataset.from_tensor_slices((eeg_train, fmri_train)).shuffle(eeg_train.shape[0], seed=opt.seed, reshuffle_each_iteration=True).batch(opt.batch_size)

t0 = time.time()
train.train(train_set, model, optimizer, model.get_loss(), epochs=opt.epochs, u_architecture=False, verbose=True)
model.save_weights(os.path.join(out, "weights"))
with open(os.path.join(out, "config.json"), "w") as f:
	json.dump(dict(vars(opt), test_subjects=test, calib_subjects=calib, train_seconds=time.time()-t0), f, indent=1)

model.decoder = model.nn
res = evaluate_uq.evaluate([model], splits, T=opt.T, stochastic=opt.variational or opt.variationalDCT, seed=opt.seed)
evaluate_uq.save(res, out)
evaluate_uq.print_summary(res)
for null in ["shift", "swap"]:
	res = evaluate_uq.evaluate([model], splits, T=20, stochastic=opt.variational or opt.variationalDCT, seed=opt.seed, null=null)
	evaluate_uq.save(res, out, tag="_null_"+null)
