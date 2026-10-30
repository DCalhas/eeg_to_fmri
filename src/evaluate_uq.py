"""
Point, distributional and conformal evaluation of EEG-to-fMRI synthesis models, computed on the GPU.

All per-volume computations (predictive moments, NLL, sampling, sorting, CRPS, intervals, ranks, sparsification,
conformal quantiles, voxel temporal r) are TensorFlow ops on the batch of predictions; only per-volume scalars
are copied back. evaluate_uq_np.py is the numpy reference the GPU code is tested against
(test_evaluate_equivalence.py).

A predictor is a model (its decoder is called) or a callable eeg -> decoder output (used by SWAG and Laplace,
which resample weights at every call). The decoder output defines the predictive family:
	* tensor                      -> point prediction
	* [mu, b]                     -> Laplace(mu, b)
	* [gamma, nu, alpha, beta]    -> Normal-Inverse-Gamma, predictive Student-t (deep evidential regression)
	* [median, lower, upper]      -> 5%/95% quantiles (conformalized quantile regression)
	* {"family": "gaussian", "params": [mu, sigma]} -> Normal (EEG-free voxelwise template, baseline_eeg_free.py)
A predictor with attribute wants_index=True is called as predictor(eeg, split, start) with the split name and the
index of the first instance of the batch (oracles that use the held-out subject's own volumes).
With S stochastic calls the predictive distribution is the mixture over the S calls.

One fixed brain mask per fold, independent of the test data (Otsu threshold of the training subjects' mean volume),
is used for calibration, coverage and every in-mask metric; point metrics are also reported on the whole volume.
Coverage is empirical: the exchangeable unit is the subject and there are only 2 calibration subjects (report its
spread over subjects), the (n+1) correction is computed over volumes x voxels.
"""
import json
import math
import os

import numpy as np
import tensorflow as tf

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
SPARSE_FRACTIONS = np.linspace(0., 0.95, 20)
N_PRED_SAMPLES = 200
QUANTILE_LEVEL = 0.9  #central interval given by the 5%/95% quantile heads
N_SPEARMAN = 5000
FAMILIES = {1: "point", 2: "laplace", 3: "quantile", 4: "nig"}


def otsu_mask(volume, bins=256):
	"""Computed once per fold on the training mean volume (a 64x64x30 histogram)."""
	hist, edges = np.histogram(volume, bins=bins)
	centers = (edges[:-1]+edges[1:])/2
	w0 = np.cumsum(hist); w1 = w0[-1]-w0
	m0 = np.cumsum(hist*centers)/np.maximum(w0, 1)
	m1 = (np.sum(hist*centers)-np.cumsum(hist*centers))/np.maximum(w1, 1)
	threshold = centers[np.argmax(w0*w1*(m0-m1)**2)]
	return volume > threshold


def ssim(y, y_hat, factor=3):
	"""Same slice-wise SSIM as utils.metrics.ssim (every factor-th axial slice, max_val from the pair); y: (X, Y, Z) tensors."""
	values = []
	for axis in range(y.shape[2]//factor):
		im1 = y[None, :, :, axis*factor, None]+1e-9
		im2 = y_hat[None, :, :, axis*factor, None]+1e-9
		values.append(tf.image.ssim(im1, im2, max_val=tf.maximum(tf.reduce_max(im1), tf.reduce_max(im2)))[0])
	return float(tf.reduce_mean(values))


def _compiled_decoder(model):
	"""
	The decoder call compiled once per model. Called eagerly, the decoder leaks GPU memory on every call
	(~20 MB/call for relaxed Flipout, ~160 MB/call at full resolution), which exhausts the GPU over T samples x batches;
	compiled, memory is flat. Stateful random ops (Flipout, dropout, spectral coefficients) still draw new samples per call.
	"""
	if(getattr(model, "_eval_fn", None) is None):
		model._eval_fn = tf.function(lambda e: model.decoder(e, training=False), reduce_retracing=True)
	return model._eval_fn


def sample_predictions(predictors, eeg, T, stochastic, split="test", start=0):
	"""Returns (family, [params]) with every param a float32 GPU tensor of shape (S, batch, X, Y, Z)."""
	calls, family = [], None
	per_predictor = max(1, T//len(predictors)) if stochastic else 1
	for predictor in predictors:
		for _ in range(per_predictor):
			if(getattr(predictor, "wants_index", False)):
				out = predictor(eeg, split, start)
			else:
				out = _compiled_decoder(predictor)(tf.convert_to_tensor(eeg)) if hasattr(predictor, "decoder") else predictor(eeg)
			if(isinstance(out, dict)):
				family, out = out["family"], out["params"]
			out = list(out) if isinstance(out, (list, tuple)) else [out]
			calls.append([tf.convert_to_tensor(o, tf.float32)[..., 0] for o in out])
	family = FAMILIES[len(calls[0])] if family is None else family
	return family, [tf.stack([c[k] for c in calls]) for k in range(len(calls[0]))]


def _logmeanexp(logp):
	return tf.reduce_logsumexp(logp, axis=0)-math.log(int(logp.shape[0]))


class Predictive:
	"""Predictive distributions of a batch over V voxels, from params of shape (S, B, V); every attribute is (B, V)."""
	def __init__(self, family, params):
		self.family, self.p = family, params
		loc = params[0]
		self.mu = tf.reduce_mean(loc, 0)
		self.var_epistemic = tf.math.reduce_variance(loc, 0)
		if(family == "laplace"):
			self.var_aleatoric = tf.reduce_mean(2*params[1]**2, 0)
		elif(family == "nig"):
			_, nu, alpha, beta = params
			self.var_aleatoric = tf.reduce_mean(beta/(alpha-1), 0)
			self.var_epistemic = self.var_epistemic+tf.reduce_mean(beta/(nu*(alpha-1)), 0)
		elif(family == "gaussian"):
			self.var_aleatoric = tf.reduce_mean(params[1]**2, 0)
		elif(family == "quantile"):
			#Gaussian-equivalent variance of the 90% interval, only used to rank voxels
			self.var_aleatoric = tf.reduce_mean(((params[2]-params[1])/(2*1.645))**2, 0)
		else:
			self.var_aleatoric = tf.zeros_like(self.mu)

	@property
	def distributional(self):
		return self.family in ("laplace", "nig", "gaussian")

	@property
	def point(self):
		return self.family == "point" and int(self.p[0].shape[0]) == 1

	def std(self):
		if(self.point):
			return tf.ones_like(self.mu)
		return tf.sqrt(self.var_epistemic+self.var_aleatoric)+1e-6

	def _student(self):
		gamma, nu, alpha, beta = self.p
		return 2*alpha, gamma, tf.sqrt(beta*(1+nu)/(nu*alpha))

	def nll(self, y):
		if(self.family == "laplace"):
			mu, b = self.p
			return -_logmeanexp(-tf.abs(y[None]-mu)/b-tf.math.log(2*b))
		if(self.family == "gaussian"):
			mu, s = self.p
			return -_logmeanexp(-0.5*tf.math.log(2*np.pi*s**2)-0.5*((y[None]-mu)/s)**2)
		df, loc, scale = self._student()
		z = (y[None]-loc)/scale
		logp = tf.math.lgamma((df+1)/2)-tf.math.lgamma(df/2)-0.5*tf.math.log(df*np.pi)-tf.math.log(scale)-(df+1)/2*tf.math.log1p(z**2/df)
		return -_logmeanexp(logp)

	def samples(self, seed):
		"""Sorted samples (>= N_PRED_SAMPLES, B, V) of the predictive mixture."""
		S = int(self.p[0].shape[0]); k = int(np.ceil(N_PRED_SAMPLES/S))
		rep = lambda t: tf.repeat(t, k, axis=0)
		shape = tf.shape(rep(self.p[0]))
		if(self.family == "laplace"):
			u = tf.random.stateless_uniform(shape, seed=[seed, 1], minval=-0.5+1e-7, maxval=0.5)
			z = rep(self.p[0])-rep(self.p[1])*tf.sign(u)*tf.math.log1p(-2*tf.abs(u))
		elif(self.family == "gaussian"):
			z = rep(self.p[0])+rep(self.p[1])*tf.random.stateless_normal(shape, seed=[seed, 2])
		else:
			df, loc, scale = self._student()
			g = tf.random.stateless_gamma(shape, seed=[seed, 3], alpha=rep(df)/2)*2  #chi2(df)
			z = rep(loc)+rep(scale)*tf.random.stateless_normal(shape, seed=[seed, 4])/tf.sqrt(g/rep(df))
		return tf.sort(z, axis=0)

	def conformity(self, y):
		"""Conformity score: CQR score for quantile heads, |y-mu|/std otherwise."""
		if(self.family == "quantile"):
			return tf.maximum(tf.reduce_mean(self.p[1], 0)-y, y-tf.reduce_mean(self.p[2], 0))
		return tf.abs(y-self.mu)/self.std()

	def conformal_interval(self, q):
		if(self.family == "quantile"):
			return tf.reduce_mean(self.p[1], 0)-q, tf.reduce_mean(self.p[2], 0)+q
		return self.mu-q*self.std(), self.mu+q*self.std()


def crps_samples(y, z):
	"""Unbiased (U-statistic) sample CRPS per voxel; z sorted along axis 0, shape (n, B, V); y (B, V)."""
	n = int(z.shape[0])
	weights = tf.reshape(tf.cast(2*tf.range(1, n+1)-n-1, z.dtype), (n, 1, 1))
	return tf.reduce_mean(tf.abs(z-y[None]), 0)-tf.reduce_sum(weights*z, 0)/(n*(n-1))


def sorted_quantile(z, q, axis=0):
	"""Linear interpolation between order statistics of z sorted along axis (numpy's default quantile)."""
	n = int(z.shape[axis]); h = q*(n-1); lo = int(np.floor(h)); hi = min(lo+1, n-1)
	a, b = tf.gather(z, lo, axis=axis), tf.gather(z, hi, axis=axis)
	return a+(h-lo)*(b-a)


def quantile(x, q):
	"""np.quantile(x, q) of a flat tensor."""
	return sorted_quantile(tf.sort(tf.reshape(x, [-1])), q)


def conformal_level(n, level):
	return min(1., np.ceil((n+1)*level)/n)


def _ranks(x):
	"""Ranks along the last axis (ties broken by order, as continuous data has none)."""
	return tf.cast(tf.argsort(tf.argsort(x, axis=-1), axis=-1), tf.float32)


def _pearson(a, b):
	a = a-tf.reduce_mean(a, -1, keepdims=True); b = b-tf.reduce_mean(b, -1, keepdims=True)
	return tf.reduce_sum(a*b, -1)/tf.sqrt(tf.reduce_sum(a**2, -1)*tf.reduce_sum(b**2, -1)+1e-12)


def sparsification(err, unc):
	"""(B, V) -> RMSE of the voxels kept after removing the most uncertain fraction, and the oracle by error: (B, F) each."""
	n = int(err.shape[-1])
	ks = [int(f*n) for f in SPARSE_FRACTIONS]
	def curve(order):
		e2 = tf.gather(err**2, order, batch_dims=1)
		total = tf.reduce_sum(e2, -1, keepdims=True)
		removed = tf.concat([tf.zeros_like(total), tf.cumsum(e2, axis=-1)], -1)
		return tf.sqrt((total-tf.gather(removed, ks, axis=-1))/tf.constant([n-k for k in ks], tf.float32))
	return curve(tf.argsort(unc, axis=-1, direction="DESCENDING")), curve(tf.argsort(err, axis=-1, direction="DESCENDING"))


def r2_temporal(pred, target):
	"""
	Out-of-sample R^2 of the BOLD time courses, same definition as fidelity_shared.r2_temporal (paper S):
	per voxel R2_v = 1 - sum_t (y-yhat)^2 / sum_t (y - mean_t y)^2, nan where the target has no temporal variance
	(a static prediction scores <= 0), and R2_global = 1 - sum SSE / sum SST over the voxels with variance.
	pred, target: (T, V) tensors of one subject. Also returns the squared correlation r^2 per voxel and the centred
	R2c (audit A19): the prediction centred on its own temporal mean, so static predictions score exactly 0 and the
	anatomy mismatch of a run-z-score target does not count as unexplained temporal variance.
	"""
	sse = tf.reduce_sum((target-pred)**2, 0)
	t = target-tf.reduce_mean(target, 0); p = pred-tf.reduce_mean(pred, 0)
	sst = tf.reduce_sum(t**2, 0)
	valid = (sst > 0).numpy()
	r2 = (1-sse/tf.where(sst > 0, sst, tf.ones_like(sst))).numpy(); r2[~valid] = np.nan
	r = (tf.reduce_sum(p*t, 0)/tf.sqrt(tf.reduce_sum(p**2, 0)*sst+1e-12)).numpy(); r[~valid] = np.nan
	sse_c = tf.reduce_sum((t-p)**2, 0)
	r2c = (1-sse_c/tf.where(sst > 0, sst, tf.ones_like(sst))).numpy(); r2c[~valid] = np.nan
	sse_v, sst_v, sse_cv = sse.numpy()[valid], sst.numpy()[valid], sse_c.numpy()[valid]
	#pooled r of the linearly detrended series (detrended after prediction): insensitive to shared run trends, which can
	#make real > shift without any EEG-specific timing (audit post-hoc v2); compare it against the shift and swap nulls
	tt = tf.cast(tf.range(tf.shape(target)[0]), tf.float32)[:, None]; tt = tt-tf.reduce_mean(tt)
	detrend = lambda x: (x-tf.reduce_mean(x, 0))-tt*tf.reduce_sum(tt*(x-tf.reduce_mean(x, 0)), 0)/tf.reduce_sum(tt**2)
	pd, td = tf.boolean_mask(detrend(pred), valid, axis=1), tf.boolean_mask(detrend(target), valid, axis=1)
	pooled_r_dt = float(tf.reduce_sum(pd*td)/tf.sqrt(tf.reduce_sum(pd**2)*tf.reduce_sum(td**2)+1e-12))
	return {"voxel_R2_mean": float(np.nanmean(r2)), "voxel_R2_median": float(np.nanmedian(r2)), "voxel_R2_frac_pos": float(np.nanmean(r2[valid] > 0)),
			"R2_global": float(1-sse_v.sum()/max(sst_v.sum(), 1e-12)), "voxel_r2corr_mean": float(np.nanmean(r**2)),
			"voxel_R2c_median": float(np.nanmedian(r2c)), "R2c_global": float(1-sse_cv.sum()/max(sst_v.sum(), 1e-12)),
			"pooled_r_dt": pooled_r_dt}


def _batches(eeg, batch):
	for i in range(0, eeg.shape[0], batch):
		yield i, eeg[i:i+batch]


def conformal_levels(family):
	"""CQR is calibrated only at the level of its quantile heads."""
	return [QUANTILE_LEVEL] if family == "quantile" else list(LEVELS)


def eeg_null(eeg, subjects, kind, seed=0, min_shift=30):
	"""Evaluation-time nulls (audit A9), applied to the test EEG only: "shift" circularly shifts each subject's
	sequence of EEG windows by >= min_shift TRs, "swap" gives every test subject another test subject's EEG."""
	from utils import fidelity_shared
	out = eeg.copy(); info = {}
	ids = list(np.unique(subjects))
	if(kind == "shift"):
		g = np.random.default_rng(seed)
		for s in ids:
			out[subjects == s], info[int(s)] = fidelity_shared.circular_shift(eeg[subjects == s], min_shift, g)
	elif(kind == "swap"):
		swapped, source = fidelity_shared.cross_subject_swap({s: eeg[subjects == s] for s in ids}, rng=seed)
		for s in ids:
			out[subjects == s] = swapped[s]
		info = {int(k): int(v) for k, v in source.items()}
	else:
		raise ValueError(kind)
	return out, info


def evaluate(predictors, splits, T=50, stochastic=True, seed=0, batch=8, n_maps=1, null=None):
	"""null: None, "shift" or "swap" (see eeg_null); calibration always uses the real EEG."""
	rng = np.random.RandomState(seed)
	eeg_test, fmri_test, sub_test = splits["test"]
	null_info = None
	if(null is not None):
		eeg_test, null_info = eeg_null(eeg_test, sub_test, null, seed=seed)
	shape = fmri_test.shape[1:4]
	mean_train_np = splits["train"][1][..., 0].mean(0)
	#with the fluctuation target the training mean is ~0 everywhere, so the anatomical mask comes with the data
	mask_np = splits["mask"] if splits.get("mask") is not None else otsu_mask(mean_train_np)
	idx = tf.constant(np.where(mask_np.reshape(-1))[0], tf.int32)
	n_mask = int(idx.shape[0])
	mean_train = tf.constant(mean_train_np, tf.float32)
	flat = lambda t: tf.reshape(t, tf.concat([tf.shape(t)[:-3], [-1]], 0))
	masked = lambda t: tf.gather(flat(t), idx, axis=-1)
	spearman_idx = tf.constant(rng.choice(n_mask, size=min(N_SPEARMAN, n_mask), replace=False), tf.int32)

	#split conformal: global quantile (pooled over calibration volumes and in-mask voxels) and per-voxel quantile
	conformal = None
	if(splits.get("calib") is not None):
		eeg_cal, fmri_cal, _ = splits["calib"]
		scores = []
		for i, x in _batches(eeg_cal, batch):
			family, params = sample_predictions(predictors, x, T, stochastic, "calib", i)
			pred = Predictive(family, [flat(p) for p in params])
			scores.append(pred.conformity(flat(tf.constant(fmri_cal[i:i+x.shape[0], ..., 0], tf.float32))))
		scores = tf.concat(scores, 0)
		levels = conformal_levels(family)
		in_mask = tf.sort(tf.reshape(tf.gather(scores, idx, axis=1), [-1]))
		by_voxel = tf.sort(scores, axis=0)
		conformal = {"levels": levels,
					"global": np.array([float(sorted_quantile(in_mask, conformal_level(int(in_mask.shape[0]), l))) for l in levels]),
					"voxel": np.stack([tf.reshape(sorted_quantile(by_voxel, conformal_level(int(scores.shape[0]), l)), shape).numpy() for l in levels])}
		del scores, in_mask, by_voxel

	keys = ["rmse", "rmse_brain", "ssim", "base_rmse", "base_rmse_brain", "base_ssim",
			"nll", "crps", "spearman", "ause", "epistemic", "aleatoric"]
	res = {k: np.full(len(sub_test), np.nan) for k in keys}
	for k in ["picp", "width", "conf_global_cov", "conf_global_width", "conf_voxel_cov", "conf_voxel_width"]:
		res[k] = np.full((len(sub_test), len(LEVELS)), np.nan)
	res["sparse_unc"] = np.full((len(sub_test), len(SPARSE_FRACTIONS)), np.nan)
	res["sparse_oracle"] = np.full((len(sub_test), len(SPARSE_FRACTIONS)), np.nan)
	res["subject"] = sub_test
	mu_mask, y_mask, mu_flat, y_flat = [], [], [], []
	maps = {}
	if(conformal is not None):
		q_voxel = [tf.gather(tf.reshape(tf.constant(q, tf.float32), [-1]), idx) for q in conformal["voxel"]]

	for i0, x in _batches(eeg_test, batch):
		family, params = sample_predictions(predictors, x, T, stochastic, "test", i0)
		B = x.shape[0]; rows = slice(i0, i0+B)
		y_full = tf.constant(fmri_test[i0:i0+B, ..., 0], tf.float32)
		full = Predictive(family, params)
		err_full = tf.abs(y_full-full.mu)
		res["rmse"][rows] = tf.sqrt(tf.reduce_mean(flat(err_full)**2, -1)).numpy()
		res["base_rmse"][rows] = tf.sqrt(tf.reduce_mean(flat(y_full-mean_train[None])**2, -1)).numpy()
		for j in range(B):
			res["ssim"][i0+j] = ssim(y_full[j], full.mu[j]); res["base_ssim"][i0+j] = ssim(y_full[j], mean_train)

		brain = Predictive(family, [masked(p) for p in params])
		y = masked(y_full); err = tf.abs(y-brain.mu); std = brain.std()
		mu_mask.append(brain.mu); y_mask.append(y); mu_flat.append(flat(full.mu).numpy()); y_flat.append(flat(y_full).numpy())
		res["rmse_brain"][rows] = tf.sqrt(tf.reduce_mean(err**2, -1)).numpy()
		res["base_rmse_brain"][rows] = tf.sqrt(tf.reduce_mean((y-masked(mean_train[None]))**2, -1)).numpy()
		if(conformal is not None):
			for l, level in enumerate(conformal["levels"]):
				col = int(np.argmin(np.abs(LEVELS-level)))
				for kind, q in [("global", float(conformal["global"][l])), ("voxel", q_voxel[l][None])]:
					lo, hi = brain.conformal_interval(q)
					res["conf_"+kind+"_cov"][rows, col] = tf.reduce_mean(tf.cast((y >= lo) & (y <= hi), tf.float32), -1).numpy()
					res["conf_"+kind+"_width"][rows, col] = tf.reduce_mean(hi-lo, -1).numpy()
		if(brain.point):
			continue
		res["epistemic"][rows] = tf.reduce_mean(brain.var_epistemic, -1).numpy(); res["aleatoric"][rows] = tf.reduce_mean(brain.var_aleatoric, -1).numpy()
		u = tf.gather(std, spearman_idx, axis=-1)
		rho = _pearson(_ranks(u), _ranks(tf.gather(err, spearman_idx, axis=-1))).numpy()
		#a constant uncertainty map has no ranking (scipy.stats.spearmanr returns nan)
		res["spearman"][rows] = np.where((tf.reduce_max(u, -1)-tf.reduce_min(u, -1)).numpy() > 0, rho, np.nan)
		cu, co = sparsification(err, std)
		res["sparse_unc"][rows], res["sparse_oracle"][rows] = cu.numpy(), co.numpy()
		res["ause"][rows] = np.trapz(res["sparse_unc"][rows]-res["sparse_oracle"][rows], SPARSE_FRACTIONS, axis=-1)
		if(brain.distributional):
			res["nll"][rows] = tf.reduce_mean(brain.nll(y), -1).numpy()
			z = brain.samples(seed*100003+i0)
			res["crps"][rows] = tf.reduce_mean(crps_samples(y, z), -1).numpy()
			for l, level in enumerate(LEVELS):
				lo, hi = sorted_quantile(z, (1-level)/2), sorted_quantile(z, (1+level)/2)
				res["picp"][rows, l] = tf.reduce_mean(tf.cast((y >= lo) & (y <= hi), tf.float32), -1).numpy()
				res["width"][rows, l] = tf.reduce_mean(hi-lo, -1).numpy()
			del z
		elif(family == "quantile"):
			col = int(np.argmin(np.abs(LEVELS-QUANTILE_LEVEL)))
			lo, hi = tf.reduce_mean(brain.p[1], 0), tf.reduce_mean(brain.p[2], 0)
			res["picp"][rows, col] = tf.reduce_mean(tf.cast((y >= lo) & (y <= hi), tf.float32), -1).numpy()
			res["width"][rows, col] = tf.reduce_mean(hi-lo, -1).numpy()
		for j in range(B):
			s = sub_test[i0+j]
			if(s not in maps and len(maps) < n_maps):
				maps[s] = np.stack([y_full[j].numpy(), full.mu[j].numpy(), tf.sqrt(full.var_epistemic[j]).numpy(), tf.sqrt(full.var_aleatoric[j]).numpy()]).astype(np.float16)

	#temporal fidelity: per-voxel Pearson r over time inside the mask, per test subject
	#(static predictions have no temporal variance: r = 0; fidelity_shared.voxel_r returns nan there). Voxels whose target is
	#constant over the subject's run (e.g. outside its own anatomical mask with -target fluct) have no r: nan, skipped by the medians
	mu_mask, y_mask = tf.concat(mu_mask, 0), tf.concat(y_mask, 0)
	mu_flat, y_flat = np.concatenate(mu_flat, 0), np.concatenate(y_flat, 0)  #host memory; one subject at a time goes to the GPU
	res["temporal_r"] = {}
	#R^2 of the time courses per test subject: whole volume (shared convention, voxels with temporal variance) and in the mask
	res["R2"] = {}
	for s in np.unique(sub_test):
		rows = tf.constant(np.where(sub_test == s)[0], tf.int32)
		a = tf.gather(mu_mask, rows); t = tf.gather(y_mask, rows)
		a = a-tf.reduce_mean(a, 0); t = t-tf.reduce_mean(t, 0)
		r = (tf.reduce_sum(a*t, 0)/tf.sqrt(tf.reduce_sum(a**2, 0)*tf.reduce_sum(t**2, 0)+1e-12)).numpy().astype(np.float32)
		r[(tf.reduce_sum(t**2, 0) <= 1e-12).numpy()] = np.nan
		res["temporal_r"][int(s)] = r
		res["R2"][int(s)] = dict(r2_temporal(tf.constant(mu_flat[sub_test == s]), tf.constant(y_flat[sub_test == s])),
								**{k+"_brain": v for k, v in r2_temporal(tf.gather(mu_mask, rows), tf.gather(y_mask, rows)).items()})
	res["maps"] = maps
	res["conformal"] = conformal
	res["null_info"] = null_info
	res["mask_fraction"] = np.array([mask_np.mean()])
	#positive control (review #2b): voxels where a known EEG-derived signal was injected, as a selection of the mask voxels
	if(splits.get("inject_mask") is not None):
		res["inject_sel"] = splits["inject_mask"].reshape(-1)[np.where(mask_np.reshape(-1))[0]].astype(bool)
	return res


def save(res, out, tag=""):
	"""tag: suffix of the files, e.g. "_null_shift"."""
	arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray) and k != "inject_sel"}
	if(res.get("inject_sel") is not None):
		arrays["inject_sel"] = res["inject_sel"].astype(np.int8)
	arrays.update({"temporal_r_"+str(s): r for s, r in res["temporal_r"].items()})
	if(res.get("R2")):
		keys = sorted(next(iter(res["R2"].values())))
		arrays["R2_subjects"] = np.array(sorted(res["R2"])); arrays["R2_keys"] = np.array(keys)
		arrays["R2_values"] = np.array([[res["R2"][s][k] for k in keys] for s in sorted(res["R2"])])
	arrays.update({"map_"+str(s): m for s, m in res["maps"].items()})
	if(res["conformal"] is not None):
		arrays["conformal_q_global"] = res["conformal"]["global"]; arrays["conformal_levels"] = np.array(res["conformal"]["levels"])
	arrays["levels"] = LEVELS; arrays["sparse_fractions"] = SPARSE_FRACTIONS
	np.savez_compressed(os.path.join(out, "eval"+tag+".npz"), **arrays)
	with open(os.path.join(out, "summary"+tag+".json"), "w") as f:
		json.dump(dict(summary(res), null_info=res.get("null_info")), f, indent=1)


def summary(res):
	"""Subject-level means (each subject weighs the same), then the mean over subjects."""
	subjects = np.unique(res["subject"])
	out = {}
	for k, v in res.items():
		if(not isinstance(v, np.ndarray) or k == "subject" or v.dtype.kind != "f" or v.shape[0] != len(res["subject"])):
			continue
		per_subject = np.stack([np.nanmean(v[res["subject"] == s], axis=0) for s in subjects])
		out[k] = np.nanmean(per_subject, axis=0).tolist()
	out["temporal_r_median"] = float(np.mean([np.nanmedian(r) for r in res["temporal_r"].values()]))
	if(res.get("inject_sel") is not None):
		out["temporal_r_inject"] = float(np.mean([np.nanmedian(r[res["inject_sel"]]) for r in res["temporal_r"].values()]))
		out["temporal_r_outside_inject"] = float(np.mean([np.nanmedian(r[~res["inject_sel"]]) for r in res["temporal_r"].values()]))
	if(res.get("R2")):
		for k in next(iter(res["R2"].values())):
			out[k] = float(np.mean([v[k] for v in res["R2"].values()]))
	out["subjects"] = subjects.tolist()
	out["mask_fraction"] = float(res["mask_fraction"][0]) if "mask_fraction" in res else None
	return out


def print_summary(res):
	s = summary(res)
	for k in ["rmse", "rmse_brain", "ssim", "base_rmse", "base_rmse_brain", "base_ssim", "nll", "crps", "spearman", "ause", "temporal_r_median", "R2_global", "voxel_R2_median", "R2_global_brain", "R2c_global", "R2c_global_brain"]:
		print(k.upper()+":", s[k])
	for k in ["picp", "conf_global_cov", "conf_voxel_cov"]:
		print(k.upper()+" @"+",".join(map(str, LEVELS))+":", np.round(s[k], 3).tolist())
