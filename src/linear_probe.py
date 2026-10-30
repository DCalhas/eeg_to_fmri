"""
Linear probe on exactly the deep models' inputs and targets (review #2a/#2d): kernel ridge from the EEG input window
(band power of the STFT input, 63 channels x 5 bands x 10 one-second windows) to every fMRI voxel of the fluctuation
target, on the same subject folds. It answers two questions:
	* positive control (-inject_snr > 0): is the injected C3/C4 mu-power signal recoverable from the deep models' input
	  at all? (median voxel r inside the injected box; validates the injection and gives the reachable r)
	* intermediate model: does a voxelwise linear model on the same input predict the real fluctuations, beyond the
	  shift and swap nulls (test EEG circularly shifted >= 30 TRs / swapped between test subjects, no refit)?
The ridge penalty is chosen on the calibration subjects (median voxel r), one penalty for all voxels.

python linear_probe.py 05 -target fluct -inject_snr 1 -fold 0 -name probe -out_dir RUNS -cache_dir CACHE
"""
import argparse
import json
import os

import numpy as np

import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
opt = parser.parse_args()

from utils import process_utils
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=0)
import tensorflow as tf

from linear_probe_utils import features as _features, voxel_r
from utils import fmri_utils as _fu
ALPHAS = 10.**np.arange(-1, 8)


class KRidge:
	"""Dual ridge: one eigendecomposition of the training Gram matrix serves every voxel and penalty."""
	def __init__(self, Xtr):
		self.mx, self.sx = Xtr.mean(0), Xtr.std(0)+1e-6
		Xs = tf.constant((Xtr-self.mx)/self.sx, tf.float64)
		lam, self.Q = tf.linalg.eigh(tf.matmul(Xs, Xs, transpose_b=True)); self.lam = tf.maximum(lam, 0.); self.Xs = Xs

	def predict(self, Ytr, Xte, alpha):
		Xt = tf.constant((Xte-self.mx)/self.sx, tf.float64)
		my = Ytr.mean(0); QtY = tf.matmul(self.Q, tf.constant(Ytr-my, tf.float64), transpose_a=True)
		KtQ = tf.matmul(tf.matmul(Xt, self.Xs, transpose_b=True), self.Q)
		return (tf.matmul(KtQ, QtY/(self.lam[:, None]+alpha))).numpy().astype(np.float32)+my


features = lambda e: _features(e, getattr(_fu, "TR_"+opt.dataset))
splits, test, calib = experiment.load_split(opt)
mask = splits["mask"].reshape(-1)
box = splits["inject_mask"].reshape(-1)[mask] if splits.get("inject_mask") is not None else None
get = lambda name: (features(splits[name][0]), splits[name][1][..., 0].reshape(len(splits[name][1]), -1)[:, mask], splits[name][2])
Xtr, Ytr, _ = get("train"); Xca, Yca, sca = get("calib"); Xte, Yte, ste = get("test")
model = KRidge(Xtr)
score = lambda P, Y, s: np.nanmedian(np.concatenate([[np.nanmedian(voxel_r(P[s == k], Y[s == k]))] for k in np.unique(s)]))
alpha = ALPHAS[int(np.argmax([score(model.predict(Ytr, Xca, a), Yca, sca) for a in ALPHAS]))]
P = model.predict(Ytr, Xte, alpha)


def per_subject(P, sel=None):
	out = {}
	for k in np.unique(ste):
		r = voxel_r(P[ste == k], Yte[ste == k])
		out[int(k)] = float(np.nanmedian(r if sel is None else r[sel]))
	return out


from evaluate_uq import eeg_null
res = {"alpha": float(alpha), "real": {"all": per_subject(P)}}
if(box is not None):
	res["real"]["box"] = per_subject(P, box); res["real"]["outside"] = per_subject(P, ~box)
for null in ["shift", "swap"]:
	eeg_null_te, _ = eeg_null(splits["test"][0], ste, null, seed=0)
	Pn = model.predict(Ytr, features(eeg_null_te), alpha)
	res[null] = {"all": per_subject(Pn)}
	if(box is not None):
		res[null]["box"] = per_subject(Pn, box)
out = experiment.run_dir(opt); os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "probe.json"), "w") as f:
	json.dump(res, f, indent=1)
print("PROBE", opt.name, "fold", opt.fold, json.dumps({k: {kk: float(np.mean(list(vv.values()))) for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in res.items()}), flush=True)
