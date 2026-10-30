"""
REFERENCE (numpy) implementation, kept only for test_evaluate_equivalence.py; experiments use evaluate_uq.py (GPU).

Point, distributional and conformal evaluation of EEG-to-fMRI synthesis models.

A predictor is a model (its decoder is called) or a callable eeg -> decoder output (used by SWAG and Laplace,
which resample weights at every call). The decoder output defines the predictive family:
	* tensor                      -> point prediction
	* [mu, b]                     -> Laplace(mu, b)
	* [gamma, nu, alpha, beta]    -> Normal-Inverse-Gamma, predictive Student-t (deep evidential regression)
	* [median, lower, upper]      -> 5%/95% quantiles (conformalized quantile regression)
	* {"family": "gaussian", "params": [mu, sigma]} -> Normal (EEG-free voxelwise template, baseline_eeg_free.py)
A predictor with attribute wants_index=True is called as predictor(eeg, split, start) with the split name and the
index of the first instance of the batch (oracles that use the held-out subject's own volumes).
With S stochastic calls (spectral coefficients, variational weights, dropout masks, SWAG/Laplace samples or
ensemble members) the predictive distribution is the mixture over the S calls.

One fixed brain mask per fold, independent of the test data (Otsu threshold of the training subjects' mean volume),
is used for calibration, coverage and every in-mask metric; point metrics are also reported on the whole volume.
Coverage is empirical: the exchangeable unit is the subject and there are only 2 calibration subjects (report its
spread over subjects), the (n+1) correction is computed over volumes x voxels.
"""
import json
import os

import numpy as np

from scipy.special import gammaln
from scipy.stats import spearmanr

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
SPARSE_FRACTIONS = np.linspace(0., 0.95, 20)
N_PRED_SAMPLES = 200
QUANTILE_LEVEL = 0.9  #central interval given by the 5%/95% quantile heads


def otsu_mask(volume, bins=256):
	hist, edges = np.histogram(volume, bins=bins)
	centers = (edges[:-1]+edges[1:])/2
	w0 = np.cumsum(hist); w1 = w0[-1]-w0
	m0 = np.cumsum(hist*centers)/np.maximum(w0, 1)
	m1 = (np.sum(hist*centers)-np.cumsum(hist*centers))/np.maximum(w1, 1)
	threshold = centers[np.argmax(w0*w1*(m0-m1)**2)]
	return volume > threshold


def ssim(y, y_hat, factor=3):
	"""Same slice-wise SSIM as utils.metrics.ssim (every factor-th axial slice, max_val from the pair)."""
	import tensorflow as tf
	values = []
	for axis in range(y.shape[2]//factor):
		im1 = y[None, :, :, axis*factor, None].astype(np.float32)+1e-9
		im2 = y_hat[None, :, :, axis*factor, None].astype(np.float32)+1e-9
		values.append(tf.image.ssim(im1, im2, max_val=float(max(im1.max(), im2.max()))).numpy()[0])
	return float(np.mean(values))


FAMILIES = {1: "point", 2: "laplace", 3: "quantile", 4: "nig"}


def sample_predictions(predictors, eeg, T, stochastic, split="test", start=0):
	"""Returns (family, [params]) with every param of shape (S, batch, X, Y, Z)."""
	calls, family = [], None
	per_predictor = max(1, T//len(predictors)) if stochastic else 1
	for predictor in predictors:
		for _ in range(per_predictor):
			if(getattr(predictor, "wants_index", False)):
				out = predictor(eeg, split, start)
			else:
				out = predictor.decoder(eeg, training=False) if hasattr(predictor, "decoder") else predictor(eeg)
			if(isinstance(out, dict)):
				family, out = out["family"], out["params"]
			out = list(out) if isinstance(out, (list, tuple)) else [out]
			calls.append([np.asarray(o)[..., 0] for o in out])
	family = FAMILIES[len(calls[0])] if family is None else family
	return family, [np.stack([c[k] for c in calls]) for k in range(len(calls[0]))]


def predictive_samples(mu, b, rng):
	"""Draws N_PRED_SAMPLES from the Laplace mixture; mu, b have shape (S, V)."""
	k = int(np.ceil(N_PRED_SAMPLES/mu.shape[0]))
	return np.repeat(mu, k, axis=0)+rng.laplace(size=(mu.shape[0]*k,)+mu.shape[1:])*np.repeat(b, k, axis=0)


def mixture_nll(y, mu, b):
	logp = -np.abs(y[None]-mu)/b-np.log(2*b)
	m = logp.max(0)
	return -(m+np.log(np.mean(np.exp(logp-m), axis=0)))


def crps_samples(y, z):
	"""Unbiased (U-statistic) sample CRPS; z must be sorted along axis 0."""
	n = z.shape[0]
	weights = (2*np.arange(1, n+1)-n-1)[:, None]
	return np.mean(np.abs(z-y[None]), axis=0)-np.sum(weights*z, axis=0)/(n*(n-1))


def sorted_quantile(z, q):
	"""Linear interpolation between order statistics of samples z sorted along axis 0 (numpy's default quantile)."""
	h = q*(z.shape[0]-1); lo = int(np.floor(h))
	return z[lo]+(h-lo)*(z[min(lo+1, z.shape[0]-1)]-z[lo])


class Predictive:
	"""Predictive distribution of one instance over V voxels, from params of shape (S, V)."""
	def __init__(self, family, params):
		self.family, self.p = family, params
		loc = params[0]
		self.mu = loc.mean(0)
		self.var_epistemic = loc.var(0)
		if(family == "laplace"):
			self.var_aleatoric = np.mean(2*params[1]**2, 0)
		elif(family == "nig"):
			_, nu, alpha, beta = params
			self.var_aleatoric = np.mean(beta/(alpha-1), 0)
			self.var_epistemic = self.var_epistemic+np.mean(beta/(nu*(alpha-1)), 0)
		elif(family == "gaussian"):
			self.var_aleatoric = np.mean(params[1]**2, 0)
		elif(family == "quantile"):
			#Gaussian-equivalent variance of the 90% interval, only used to rank voxels
			self.var_aleatoric = np.mean(((params[2]-params[1])/(2*1.645))**2, 0)
		else:
			self.var_aleatoric = np.zeros_like(self.mu)

	@property
	def distributional(self):
		return self.family in ("laplace", "nig", "gaussian")

	def std(self):
		if(self.family == "point" and self.p[0].shape[0] == 1):
			return np.ones_like(self.mu)
		return np.sqrt(self.var_epistemic+self.var_aleatoric)+1e-6

	def _student(self):
		gamma, nu, alpha, beta = self.p
		return 2*alpha, gamma, np.sqrt(beta*(1+nu)/(nu*alpha))

	def nll(self, y):
		if(self.family == "laplace"):
			return mixture_nll(y, self.p[0], self.p[1])
		if(self.family == "gaussian"):
			logp = -0.5*np.log(2*np.pi*self.p[1]**2)-0.5*((y[None]-self.p[0])/self.p[1])**2
			m = logp.max(0)
			return -(m+np.log(np.mean(np.exp(logp-m), axis=0)))
		df, loc, scale = self._student()
		z = (y[None]-loc)/scale
		logp = gammaln((df+1)/2)-gammaln(df/2)-0.5*np.log(df*np.pi)-np.log(scale)-(df+1)/2*np.log1p(z**2/df)
		m = logp.max(0)
		return -(m+np.log(np.mean(np.exp(logp-m), axis=0)))

	def samples(self, rng):
		"""Sorted samples (N_PRED_SAMPLES, V) of the predictive mixture."""
		if(self.family == "laplace"):
			z = predictive_samples(self.p[0], self.p[1], rng)
		elif(self.family == "gaussian"):
			k = int(np.ceil(N_PRED_SAMPLES/self.p[0].shape[0]))
			z = np.repeat(self.p[0], k, 0)+np.repeat(self.p[1], k, 0)*rng.standard_normal((self.p[0].shape[0]*k,)+self.p[0].shape[1:])
		else:
			df, loc, scale = self._student()
			k = int(np.ceil(N_PRED_SAMPLES/loc.shape[0]))
			z = np.repeat(loc, k, 0)+np.repeat(scale, k, 0)*rng.standard_t(np.repeat(df, k, 0))
		return np.sort(z, axis=0)

	def conformity(self, y):
		"""Conformity score: CQR score for quantile heads, |y-mu|/std otherwise."""
		if(self.family == "quantile"):
			return np.maximum(self.p[1].mean(0)-y, y-self.p[2].mean(0))
		return np.abs(y-self.mu)/self.std()

	def conformal_interval(self, q):
		if(self.family == "quantile"):
			return self.p[1].mean(0)-q, self.p[2].mean(0)+q
		return self.mu-q*self.std(), self.mu+q*self.std()


def sparsification(err, unc):
	"""RMSE of the voxels kept after removing the most uncertain fraction (and the oracle by error)."""
	order_u, order_e = np.argsort(-unc), np.argsort(-err)
	n = len(err); curve_u, curve_e = [], []
	for f in SPARSE_FRACTIONS:
		k = int(f*n)
		curve_u.append(np.sqrt(np.mean(err[order_u[k:]]**2))); curve_e.append(np.sqrt(np.mean(err[order_e[k:]]**2)))
	return np.array(curve_u), np.array(curve_e)


def conformal_quantile(scores, level, axis=None):
	n = scores.shape[0] if axis == 0 else scores.size
	q = min(1., np.ceil((n+1)*level)/n)
	return np.quantile(scores, q, axis=axis)


def _batches(eeg, batch):
	for i in range(0, eeg.shape[0], batch):
		yield i, eeg[i:i+batch]


def _instance(family, params, j, mask=None):
	return Predictive(family, [p[:, j] if mask is None else p[:, j][:, mask] for p in params])


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
	fmri_test = fmri_test[..., 0]
	mean_train = splits["train"][1][..., 0].mean(0)
	mask = otsu_mask(mean_train)
	mask_flat = mask.reshape(-1)

	#split conformal: global quantile (pooled over calibration volumes and voxels) and per-voxel quantile
	conformal = None
	if(splits.get("calib") is not None):
		eeg_cal, fmri_cal, _ = splits["calib"]
		scores = np.full((eeg_cal.shape[0], int(np.prod(fmri_cal.shape[1:4]))), np.nan, dtype=np.float32)
		for i, x in _batches(eeg_cal, batch):
			family, params = sample_predictions(predictors, x, T, stochastic, "calib", i)
			params = [p.reshape(p.shape[:2]+(-1,)) for p in params]
			for j in range(x.shape[0]):
				scores[i+j] = _instance(family, params, j).conformity(fmri_cal[i+j, ..., 0].reshape(-1))
		levels = conformal_levels(family)
		#global quantile over the same brain mask the coverage is evaluated in; per-voxel quantiles everywhere
		conformal = {"levels": levels, "global": np.array([conformal_quantile(scores[:, mask_flat], l) for l in levels]),
					"voxel": np.stack([conformal_quantile(scores, l, axis=0) for l in levels]).reshape((len(levels),)+fmri_cal.shape[1:4])}
		del scores

	keys = ["rmse", "rmse_brain", "ssim", "base_rmse", "base_rmse_brain", "base_ssim",
			"nll", "crps", "spearman", "ause", "epistemic", "aleatoric"]
	res = {k: np.full(len(sub_test), np.nan) for k in keys}
	for k in ["picp", "width", "conf_global_cov", "conf_global_width", "conf_voxel_cov", "conf_voxel_width"]:
		res[k] = np.full((len(sub_test), len(LEVELS)), np.nan)
	res["sparse_unc"] = np.full((len(sub_test), len(SPARSE_FRACTIONS)), np.nan)
	res["sparse_oracle"] = np.full((len(sub_test), len(SPARSE_FRACTIONS)), np.nan)
	res["subject"] = sub_test
	mu_all = np.empty_like(fmri_test)
	maps = {}

	for i0, x in _batches(eeg_test, batch):
		family, params = sample_predictions(predictors, x, T, stochastic, "test", i0)
		for j in range(x.shape[0]):
			i = i0+j; y = fmri_test[i]
			full = _instance(family, params, j)
			mu_all[i] = full.mu
			err = np.abs(y-full.mu)
			res["rmse"][i] = np.sqrt(np.mean(err**2)); res["rmse_brain"][i] = np.sqrt(np.mean(err[mask]**2))
			res["ssim"][i] = ssim(y, full.mu)
			res["base_rmse"][i] = np.sqrt(np.mean((y-mean_train)**2)); res["base_rmse_brain"][i] = np.sqrt(np.mean((y-mean_train)[mask]**2))
			res["base_ssim"][i] = ssim(y, mean_train)
			if(conformal is not None):
				for l, level in enumerate(conformal["levels"]):
					col = int(np.argmin(np.abs(LEVELS-level)))
					for kind, q in [("global", conformal["global"][l]), ("voxel", conformal["voxel"][l])]:
						lo, hi = full.conformal_interval(q)
						res["conf_"+kind+"_cov"][i, col] = np.mean(((y >= lo) & (y <= hi))[mask])
						res["conf_"+kind+"_width"][i, col] = np.mean((hi-lo)[mask])
			if(family == "point" and params[0].shape[0] == 1):
				continue
			brain = _instance(family, params, j, mask)
			yb, eb, sb = y[mask], err[mask], brain.std()
			res["epistemic"][i] = np.mean(brain.var_epistemic); res["aleatoric"][i] = np.mean(brain.var_aleatoric)
			sub = rng.choice(len(eb), size=min(5000, len(eb)), replace=False)
			res["spearman"][i] = spearmanr(sb[sub], eb[sub]).correlation
			cu, co = sparsification(eb, sb)
			res["sparse_unc"][i], res["sparse_oracle"][i] = cu, co
			res["ause"][i] = np.trapz(cu-co, SPARSE_FRACTIONS)
			if(brain.distributional):
				res["nll"][i] = np.mean(brain.nll(yb))
				z = brain.samples(rng)
				res["crps"][i] = np.mean(crps_samples(yb, z))
				for l, level in enumerate(LEVELS):
					lo, hi = sorted_quantile(z, (1-level)/2), sorted_quantile(z, (1+level)/2)
					res["picp"][i, l] = np.mean((yb >= lo) & (yb <= hi)); res["width"][i, l] = np.mean(hi-lo)
			elif(family == "quantile"):
				col = int(np.argmin(np.abs(LEVELS-QUANTILE_LEVEL)))
				lo, hi = brain.p[1].mean(0), brain.p[2].mean(0)
				res["picp"][i, col] = np.mean((yb >= lo) & (yb <= hi)); res["width"][i, col] = np.mean(hi-lo)
			if(sub_test[i] not in maps and len(maps) < n_maps):
				maps[sub_test[i]] = np.stack([y, full.mu, np.sqrt(full.var_epistemic), np.sqrt(full.var_aleatoric)]).astype(np.float16)

	#temporal fidelity: per-voxel Pearson r over time inside the mask, per test subject
	res["temporal_r"] = {}
	for s in np.unique(sub_test):
		idx = sub_test == s
		a = mu_all[idx][:, mask]; t = fmri_test[idx][:, mask]
		a = a-a.mean(0); t = t-t.mean(0)
		#static predictions have no temporal variance: r = 0 (fidelity_shared.voxel_r returns nan there)
		res["temporal_r"][int(s)] = (np.sum(a*t, 0)/np.sqrt(np.sum(a**2, 0)*np.sum(t**2, 0)+1e-12)).astype(np.float32)
	res["maps"] = maps
	res["conformal"] = conformal
	res["null_info"] = null_info
	res["mask_fraction"] = np.array([mask.mean()])
	return res


def save(res, out, tag=""):
	"""tag: suffix of the files, e.g. "_null_shift"."""
	arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
	arrays.update({"temporal_r_"+str(s): r for s, r in res["temporal_r"].items()})
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
	out["temporal_r_median"] = float(np.mean([np.median(r) for r in res["temporal_r"].values()]))
	out["subjects"] = subjects.tolist()
	out["mask_fraction"] = float(res["mask_fraction"][0]) if "mask_fraction" in res else None
	return out


def print_summary(res):
	s = summary(res)
	for k in ["rmse", "rmse_brain", "ssim", "base_rmse", "base_rmse_brain", "base_ssim", "nll", "crps", "spearman", "ause", "temporal_r_median"]:
		print(k.upper()+":", s[k])
	for k in ["picp", "conf_global_cov", "conf_voxel_cov"]:
		print(k.upper()+" @"+",".join(map(str, LEVELS))+":", np.round(s[k], 3).tolist())
