"""
GPU evaluator (evaluate_uq) vs numpy reference (evaluate_uq_np) on tiny synthetic data: exact for the deterministic
metrics, Monte Carlo tolerance for the sampled ones (CRPS, PICP). Runs in seconds; called first in the GPU smoke job.
"""
import sys
import numpy as np
import tensorflow as tf

import evaluate_uq as G
import evaluate_uq_np as N

rng = np.random.RandomState(0); X = (12, 12, 6); n = 70
def subj(sid):
	anat = np.zeros(X); anat[2:10, 2:10, 1:5] = 1.0+0.05*sid; d = rng.normal(size=n)
	y = anat[None]+0.5*d[:, None, None, None]*anat[None]+0.3*rng.normal(size=(n,)+X)
	return d[:, None].astype(np.float32), y[..., None].astype(np.float32), np.full(n, sid)
def split(ids):
	p = [subj(i) for i in ids]; return tuple(np.concatenate([q[k] for q in p]) for k in range(3))
splits = {"train": split([0, 1, 2, 3]), "calib": split([4, 5]), "test": split([6, 7, 8])}
anat = splits["train"][1][..., 0].mean(0)

class Fixed:
	"""Deterministic predictors per family (S = 1), so both implementations see identical parameters."""
	def __init__(self, family):
		self.family = family
	def __call__(self, eeg):
		e = np.asarray(eeg)[:, 0, None, None, None]
		mu = (anat[None]*(1+0.5*e)).astype(np.float32); one = np.ones_like(mu)
		scale = (0.2+0.1*np.abs(mu)+0.05*np.sin(np.arange(mu.size).reshape(mu.shape))).astype(np.float32)  #voxel-varying, no ties
		if(self.family == "laplace"):
			out = [mu, scale]
		elif(self.family == "nig"):
			out = [mu, 2*one, 3*one, scale/2]
		elif(self.family == "quantile"):
			out = [mu, mu-2*scale, mu+2.5*scale]
		elif(self.family == "gaussian"):
			return {"family": "gaussian", "params": [mu[..., None], scale[..., None]]}
		else:
			out = mu
		return [o[..., None] for o in out] if isinstance(out, list) else out[..., None]

N.ssim = lambda y, yh, factor=3: G.ssim(tf.constant(y, tf.float32), tf.constant(yh, tf.float32), factor)
N.N_SPEARMAN = G.N_SPEARMAN = 10**6  #all in-mask voxels, so both rank the same set
ok = True
for family in ["point", "laplace", "nig", "quantile", "gaussian"]:
	g = G.summary(G.evaluate([Fixed(family)], splits, T=1, stochastic=False, batch=16))
	r = N.summary(N.evaluate([Fixed(family)], splits, T=1, stochastic=False, batch=16))
	for k in ["rmse", "rmse_brain", "ssim", "base_rmse", "base_rmse_brain", "nll", "spearman", "ause", "conf_global_cov", "conf_voxel_cov", "conf_global_width", "conf_voxel_width", "temporal_r_median"]:
		a, b = np.array(g[k], float), np.array(r[k], float)
		if(not np.allclose(a, b, rtol=2e-3, atol=2e-3, equal_nan=True)):
			ok = False; print("MISMATCH", family, k, a, b)
	for k in ["crps", "picp"]:
		a, b = np.array(g[k], float), np.array(r[k], float)
		if(not np.allclose(a, b, atol=0.02, equal_nan=True)):
			ok = False; print("MISMATCH (MC)", family, k, a, b)
	print(family, "checked")
#R^2: numpy port of fidelity_shared.r2_temporal (paper S, torch) on random time courses with some constant voxels
def r2_reference(pred, target):
	mask = target.std(0) > 0
	p, t = pred[:, mask], target[:, mask]
	sse = ((t-p)**2).sum(0); sst = ((t-t.mean(0))**2).sum(0)
	r2 = np.where(sst > 0, 1-sse/np.where(sst > 0, sst, 1), np.nan)
	pc, tc = p-p.mean(0), t-t.mean(0); ssec = ((tc-pc)**2).sum(0)
	r2c = np.where(sst > 0, 1-ssec/np.where(sst > 0, sst, 1), np.nan)
	return {"voxel_R2_mean": np.nanmean(r2), "voxel_R2_median": np.nanmedian(r2), "voxel_R2_frac_pos": np.nanmean(r2 > 0), "R2_global": 1-sse.sum()/max(sst.sum(), 1e-12),
			"voxel_R2c_median": np.nanmedian(r2c), "R2c_global": 1-ssec.sum()/max(sst.sum(), 1e-12)}  #fidelity_shared.r2_temporal @b49121f incl. c_* (A19)
t = rng.normal(size=(120, 500)).astype(np.float32); t[:, :20] = 1.0
p = (0.6*t+0.8*rng.normal(size=t.shape)).astype(np.float32)
g, r = G.r2_temporal(tf.constant(p), tf.constant(t)), r2_reference(p, t)
for k in r:
	if(not np.isclose(g[k], r[k], rtol=1e-4, atol=1e-5)):
		ok = False; print("MISMATCH r2", k, g[k], r[k])
static = G.r2_temporal(tf.constant(np.broadcast_to(t.mean(0), t.shape).copy()), tf.constant(t))
if(not static["R2_global"] <= 1e-6):
	ok = False; print("MISMATCH static R2_global should be <= 0", static["R2_global"])
anat = G.r2_temporal(tf.constant(np.broadcast_to(t.mean(0)+3., t.shape).copy()), tf.constant(t))
if(not (abs(anat["R2c_global"]) < 1e-6 and anat["R2_global"] < -1)):
	ok = False; print("MISMATCH centred R2 of a shifted static prediction should be 0", anat["R2c_global"], anat["R2_global"])
#detrended pooled r: numpy polyfit detrend reference; a shared linear trend alone gives pooled_r_dt = 0
def pooled_dt_reference(pred, target):
	m = target.std(0) > 0; t_ = np.arange(len(target))
	dt = lambda x: x-np.outer(t_, np.polyfit(t_, x, 1)[0])-np.polyfit(t_, x, 1)[1]
	pd, td = dt(pred[:, m]), dt(target[:, m]); return (pd*td).sum()/np.sqrt((pd**2).sum()*(td**2).sum())
if(not np.isclose(G.r2_temporal(tf.constant(p), tf.constant(t))["pooled_r_dt"], pooled_dt_reference(p, t), atol=1e-4)):
	ok = False; print("MISMATCH pooled_r_dt")
trend = np.arange(120, dtype=np.float32)[:, None]*np.ones((1, 500), np.float32)/50
tr_only = G.r2_temporal(tf.constant(trend+rng.normal(size=t.shape).astype(np.float32)*0.01), tf.constant(trend+t))
if(not abs(tr_only["pooled_r_dt"]) < 0.05):
	ok = False; print("MISMATCH a shared trend alone should give pooled_r_dt ~ 0", tr_only["pooled_r_dt"])
print("r2 checked")
print("EQUIVALENCE_OK" if ok else "EQUIVALENCE_FAILED")
sys.exit(0 if ok else 1)
