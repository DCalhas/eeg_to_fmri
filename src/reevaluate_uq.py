"""
Re-evaluation of trained cross-validated runs (no training): restores the saved weights of every fold and seed of one
configuration and computes, on the test subjects, the analyses requested in the journal review.

A. Uncertainty-specific EEG dependence (the nulls of evaluate_uq test the predictive mean; this tests the spread).
   For the real EEG and the shift/swap nulls (evaluate_uq.eeg_null, same null seed as the existing evaluation), per test
   subject, inside the fold's brain mask, with sigma[t, v] = Predictive.std() (sqrt of epistemic + aleatoric variance):
     * temporal/spatial variance ratio of sigma: mean_v Var_t(sigma[:, v]) / Var_v(mean_t sigma[:, v]);
     * real vs null: spatial Pearson of the time-averaged sigma maps, and the mean over volumes of the per-volume spatial
       Pearson of sigma;
     * real vs null: 2-Wasserstein distance between the moment-matched Normals of the two predictive distributions,
       W2^2 = (mu_r - mu_n)^2 + (sigma_r - sigma_n)^2, averaged over volumes and voxels; mean and spread terms reported
       separately and relative to the average predictive variance of the real EEG, mean(sigma_r^2);
     * error tracking (each condition with its own error e = y - mu): anatomical part a[v] = mean_t e, temporal part
       d = e - a; Spearman(mean_t sigma[v], |a[v]|) across voxels and the mean (and median) over voxels of the
       within-voxel Spearman over time of sigma[t, v] and |d[t, v]| (mid-ranks as scipy; voxels with constant sigma or
       constant |d| are skipped). A point model (constant sigma) gets only the W2 mean term.
B. Cross-conformal calibration, pass 1: normalized conformity scores (Predictive.conformity: |y-mu|/sigma, or the CQR
   score of the quantile head) of every test subject on a FIXED common mask (intersection over folds of the fold masks),
   float16, to <run dir>/oof_scores.npz, together with the scores of the fold's calibration subjects under the same
   model and the per-voxel time-mean interval width terms (width = base + slope*q).
   Pass 2 (-combine, numpy only): for every subject s and seed, calibrate with the out-of-fold scores of all other
   subjects, global (pooled over volumes x voxels) and per-voxel quantiles at LEVELS with the (n+1) correction
   (evaluate_uq.conformal_level, numpy's linear quantile = evaluate_uq.sorted_quantile), and report the empirical
   coverage and mean width on s; for comparison the same with only the fold's calibration subjects, both under the
   fold's own model (as evaluate_uq) and with their out-of-fold scores. Coverage is averaged over seeds per subject.
   Caveat: the other subjects' scores come from different fold models; normalized scores make them approximately,
   not exactly, exchangeable with the test subject's (subject is the exchangeable unit).
C. Mask fraction: Otsu mask of each test subject's own mean volume (run_zscore target; with -target fluct the voxels
   with temporal variance, which is the subject's own Otsu mask by construction; not recoverable for fluct_resid),
   fraction of it inside the fold mask and inside the common mask, and fraction of the fold mask inside it.
D. -mode template_mixture: leave-one-subject-out EEG-free reference, the equal-weight mixture over the training
   subjects of each subject's voxelwise Normal(own mean, own SD + 1e-4). The mixture is formed exactly as the S stochastic
   calls / ensemble members of evaluate_uq.sample_predictions: one component callable per training subject, each
   returning {"family": "gaussian", "params": [mu, sigma]}, called once each (T = number of components), so
   Predictive gives the mixture NLL (logmeanexp over components), moments and CRPS from mixture samples. Writes
   eval.npz/summary.json to <out_dir>/<dataset>/template_mixture[_fluct]/fold<k>/seed0 like baseline_eeg_free.py.

The model of each run is rebuilt from its config.json (all experiment.add_model_args options except the I/O ones),
exactly as train_cv.py/evaluate_ensemble.py did, and its tf checkpoint is restored. Supported: configurations whose
stochasticity is in the saved weights or the architecture (point, hetero, evidential, quantile, MC/spectral dropout,
stochastic spectral decoder, Flipout/legacy Bayesian layer) and the deep ensemble <base>_ensemble (members read as in
evaluate_ensemble.py). Not supported: SWAG and last-layer Laplace (zerofill_swag, zerofill_laplace): their samplers
(SWAG moments, Laplace Fisher and selected prior) are not saved, rebuilding them means re-running the post-hoc fit.
All conditions use the same number of samples T (train_cv used T_null=20 for the nulls; equal T keeps sigma comparable).

python reevaluate_uq.py 05 -name zerofill_hetero -out_dir RUNS -cache_dir CACHE [-folds 0,1] [-seeds 2,3,5] [-T 50]
python reevaluate_uq.py 05 -name zerofill_hetero_ensemble -out_dir RUNS -cache_dir CACHE -members 2,3,5,7,11
python reevaluate_uq.py 05 -name zerofill_hetero -out_dir RUNS -combine
python reevaluate_uq.py 05 -name template_mixture -out_dir RUNS -cache_dir CACHE -mode template_mixture
python reevaluate_uq.py 05 -name masks -out_dir RUNS -cache_dir CACHE -mode masks   (precompute the fold masks)
"""
import argparse
import copy
import gc
import glob
import json
import os

import numpy as np

import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
parser.add_argument('-mode', default="runs", choices=["runs", "combine", "template_mixture", "masks"])
parser.add_argument('-combine', action="store_true", help="Same as -mode combine (cross-conformal pass 2)")
parser.add_argument('-folds', default=None, type=str, help="Comma separated folds (default: all)")
parser.add_argument('-seeds', default=None, type=str, help="Comma separated seeds (default: every seed dir with weights)")
parser.add_argument('-members', default="2,3,5,7,11", type=str, help="Ensemble members (configurations named <base>_ensemble)")
parser.add_argument('-T', default=50, type=int, help="Monte Carlo samples, for the real EEG and the nulls")
parser.add_argument('-nulls', default="shift,swap", type=str)
parser.add_argument('-batch', default=8, type=int)
parser.add_argument('-overwrite', action="store_true", help="Recompute runs that already have both output files")

#options that are I/O or per-run, never taken from a run's config.json
IO_KEYS = {"help", "dataset", "name", "out_dir", "cache_dir", "gpu_mem", "verbose", "fold", "seed"}
MODEL_KEYS = [a.dest for a in experiment.add_model_args(argparse.ArgumentParser())._actions if a.dest not in IO_KEYS]
DATA_KEYS = ["n_folds", "n_calib", "interval_eeg", "channels", "eeg_norm", "eeg_fmax", "target", "debug_subset"]
F16_MAX = float(np.finfo(np.float16).max)
SUFFIX = {"run_zscore": "", "fluct": "_fluct", "fluct_resid": "_fluct_resid"}


def _py(x):
	"""numpy -> JSON-serializable."""
	if(isinstance(x, dict)):
		return {str(k): _py(v) for k, v in x.items()}
	if(isinstance(x, (list, tuple))):
		return [_py(v) for v in x]
	if(isinstance(x, np.ndarray)):
		return _py(x.tolist())
	if(isinstance(x, (np.floating, float))):
		return float(x)
	if(isinstance(x, (np.integer,))):
		return int(x)
	if(isinstance(x, (np.bool_,))):
		return bool(x)
	return x


def _dump(obj, path):
	tmp = path+".tmp"
	with open(tmp, "w") as f:
		json.dump(_py(obj), f, indent=1)
	os.replace(tmp, path)


def _list(s):
	return None if s is None else [int(v) for v in s.split(",") if v != ""]


# ---------------------------------------------------------------------------------------------------------------- masks

def fold_mask(splits):
	"""The fold mask exactly as evaluate_uq.evaluate (Otsu of the training mean, or the data's mask with -target fluct)."""
	import evaluate_uq
	if(splits.get("mask") is not None):
		return np.asarray(splits["mask"], bool)
	return evaluate_uq.otsu_mask(splits["train"][1][..., 0].mean(0))


def masks_path(opt):
	tag = opt.target+"_k"+str(opt.n_folds)+"_c"+str(opt.n_calib)+("_debug"+str(opt.debug_subset) if opt.debug_subset else "")
	return os.path.join(opt.out_dir, opt.dataset, "_masks", tag+".npz")


def fold_masks(opt):
	"""(n_folds, X, Y, Z) fold masks and their intersection; loads every fold's split once, cached under <out_dir>/<dataset>/_masks."""
	path = masks_path(opt)
	if(os.path.exists(path)):
		with np.load(path) as f:
			return f["folds"], f["common"]
	masks = []
	for k in range(opt.n_folds):
		o = copy.copy(opt); o.fold = k
		splits, _, _ = experiment.load_split(o)
		masks.append(fold_mask(splits))
		del splits; gc.collect()
		print("I: fold", k, "mask fraction", round(float(masks[-1].mean()), 4), flush=True)
	masks = np.stack(masks); common = np.all(masks, 0)
	print("I: common mask fraction", round(float(common.mean()), 4), "voxels", int(common.sum()), flush=True)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	tmp = path+".tmp"
	with open(tmp, "wb") as f:
		np.savez(f, folds=masks, common=common)
	os.replace(tmp, path)
	return masks, common


# ------------------------------------------------------------------------------------------------------------ run config

def run_opt(cli, cfg):
	"""CLI options with every model/data option replaced by the run's config.json."""
	o = copy.copy(cli)
	for k in MODEL_KEYS:
		if(k in cfg):
			setattr(o, k, cfg[k])
	for k in ["na_path_eeg", "na_path_fmri"]:
		if(not os.path.exists(getattr(o, k))):
			print("W:", k, getattr(o, k), "missing, using", getattr(cli, k), flush=True)
			setattr(o, k, getattr(cli, k))
	if(cli.cache_dir is None):
		o.cache_dir = cfg.get("cache_dir")
	return o


def read_config(opt, name, fold, seed):
	o = copy.copy(opt); o.fold, o.seed = fold, seed
	d = experiment.run_dir(o, name)
	path = os.path.join(d, "config.json")
	if(not os.path.exists(path)):
		raise SystemExit("no config.json in "+d)
	with open(path) as f:
		cfg = json.load(f)
	if(cfg.get("swag_epochs", 0) > 0 or cfg.get("laplace") or cfg.get("init_from")):
		raise SystemExit(name+": post-hoc sampler (SWAG/Laplace, -init_from) is not saved; rebuilding it needs a re-run of "
						"the post-hoc fit (train_cv.py -init_from ...). Not supported here.")
	if(not os.path.exists(os.path.join(d, "weights.index"))):
		raise SystemExit("no weights in "+d)
	return cfg, d


def runs_of(opt):
	"""[(fold, out_seed, null_seed, run_dir, [(member_cfg, member_dir)])] for the requested folds and seeds."""
	folds = _list(opt.folds) or list(range(opt.n_folds))
	ensemble = opt.name.endswith("_ensemble")
	base = opt.name[:-len("_ensemble")] if ensemble else opt.name
	runs = []
	for k in folds:
		if(ensemble):
			members = _list(opt.members)
			o = copy.copy(opt); o.fold, o.seed = k, 0
			runs.append((k, 0, members[0], experiment.run_dir(o), [read_config(opt, base, k, m) for m in members]))
			continue
		seeds = _list(opt.seeds)
		if(seeds is None):
			o = copy.copy(opt); o.fold, o.seed = k, 0
			fold_dir = os.path.dirname(experiment.run_dir(o))
			seeds = sorted(int(os.path.basename(d)[4:]) for d in glob.glob(os.path.join(fold_dir, "seed*"))
							if os.path.exists(os.path.join(d, "weights.index")))
		for s in seeds:
			cfg, d = read_config(opt, base, k, s)
			runs.append((k, s, s, d, [(cfg, d)]))
	return runs


def build_predictors(cli, members, eeg_shape, fmri_shape, n_train, seed):
	"""Models rebuilt as train_cv.py/evaluate_ensemble.py and restored from their checkpoints."""
	import tensorflow as tf
	from utils import tf_config
	tf.keras.backend.clear_session(); gc.collect()
	tf_config.set_seed(seed)
	models, opts = [], []
	for cfg, d in members:
		o = run_opt(cli, cfg)
		with tf.device('/CPU:0'):
			model, _ = experiment.build_model(o, eeg_shape, fmri_shape, n_train=n_train)
		model.load_weights(os.path.join(d, "weights")).expect_partial()
		models.append(model); opts.append(o)
	stochastic = experiment.is_stochastic(opts[0])
	if(len(models) > 1):
		#deep ensemble as evaluate_ensemble.py: mixture over members (T split across stochastic members)
		return models, True, (cli.T if stochastic else len(models))
	return models, stochastic, cli.T


# ------------------------------------------------------------------------------------------------------ GPU statistics

def _tf():
	import tensorflow as tf
	return tf


def _corr(a, b):
	"""Pearson along the last axis; nan where either side is constant."""
	tf = _tf()
	a = a-tf.reduce_mean(a, -1, keepdims=True); b = b-tf.reduce_mean(b, -1, keepdims=True)
	den = tf.sqrt(tf.reduce_sum(a*a, -1)*tf.reduce_sum(b*b, -1))
	return tf.where(den > 0, tf.reduce_sum(a*b, -1)/tf.where(den > 0, den, tf.ones_like(den)), tf.fill(tf.shape(den), np.float32(np.nan)))


def _mid_ranks(x):
	"""Ranks along the last axis with ties at their average rank (scipy.stats.rankdata), 0-based."""
	tf = _tf()
	s = tf.sort(x, axis=-1)
	lo = tf.searchsorted(s, x, side="left"); hi = tf.searchsorted(s, x, side="right")
	return (tf.cast(lo+hi, tf.float32)-1.)/2.


def spread_stats(sig):
	"""sig: (T, V). Does the predictive SD move over time at all, relative to its spatial pattern."""
	tf = _tf()
	tv = float(tf.reduce_mean(tf.math.reduce_variance(sig, 0)))
	sv = float(tf.math.reduce_variance(tf.reduce_mean(sig, 0)))
	return {"sigma_mean": float(tf.reduce_mean(sig)), "sigma_temporal_var": tv, "sigma_spatial_var": sv,
			"sigma_temporal_to_spatial_var": tv/sv if sv > 0 else np.nan}


def error_tracking(y, mu, sig):
	"""Spearman of sigma with the anatomical (time-mean) and temporal (residual) parts of the error; (T, V) inputs."""
	tf = _tf()
	e = y-mu; a = tf.reduce_mean(e, 0); d = tf.abs(e-a[None]); a = tf.abs(a)
	sbar = tf.reduce_mean(sig, 0)
	ok = tf.math.is_finite(sbar) & tf.math.is_finite(a)
	rho_anat = float(_corr(_mid_ranks(tf.boolean_mask(sbar, ok)), _mid_ranks(tf.boolean_mask(a, ok))))
	S, D = tf.transpose(sig), tf.transpose(d)  #(V, T)
	finite = tf.reduce_all(tf.math.is_finite(S), -1) & tf.reduce_all(tf.math.is_finite(D), -1)
	S, D = tf.where(finite[:, None], S, tf.zeros_like(S)), tf.where(finite[:, None], D, tf.zeros_like(D))
	moving = lambda X: tf.reduce_max(X, -1)-tf.reduce_min(X, -1) > 0
	valid = (finite & moving(S) & moving(D)).numpy()
	rho = _corr(_mid_ranks(S), _mid_ranks(D)).numpy()
	rho[~valid] = np.nan
	n = int(valid.sum())
	return {"spearman_sigma_abs_anatomical_error": rho_anat,
			"spearman_sigma_abs_temporal_error_mean": float(np.nanmean(rho)) if n else np.nan,
			"spearman_sigma_abs_temporal_error_median": float(np.nanmedian(rho)) if n else np.nan,
			"spearman_sigma_abs_temporal_error_frac_pos": float(np.mean(rho[valid] > 0)) if n else np.nan,
			"n_voxels_temporal": n, "n_voxels": int(valid.shape[0])}


def versus_real(mu_r, sig_r, mu_n, sig_n, point):
	"""Real vs null predictive distributions of one subject, (T, V) each."""
	tf = _tf()
	mean_term = float(tf.reduce_mean((mu_r-mu_n)**2))
	avg_var = float(tf.reduce_mean(sig_r**2))
	out = {"w2_mean_term": mean_term, "avg_pred_var_real": avg_var, "w2_mean_term_rel": mean_term/avg_var}
	if(point):
		return out
	spread_term = float(tf.reduce_mean((sig_r-sig_n)**2))
	out.update({"w2_spread_term": spread_term, "w2": mean_term+spread_term, "w2_spread_term_rel": spread_term/avg_var,
				"w2_rel": (mean_term+spread_term)/avg_var,
				"sigma_map_corr": float(_corr(tf.reduce_mean(sig_r, 0), tf.reduce_mean(sig_n, 0))),
				"sigma_volume_corr_mean": float(np.nanmean(_corr(sig_r, sig_n).numpy()))})
	return out


def subject_pass(predictors, eeg, fmri, rows, T, stochastic, idx, pos_common, split, batch, moments=True, scores=False):
	"""
	Predictive moments in the fold mask for the given rows (one subject, in time order), batched as evaluate_uq.
	Returns dict with family, point, mu/sig/y (T_s, V_fold) GPU tensors (moments=True) and, with scores=True, the
	conformity scores on the common mask (float16 host) and the summed per-voxel width terms.
	"""
	tf = _tf()
	import evaluate_uq
	flat = lambda t: tf.reshape(t, tf.concat([tf.shape(t)[:-3], [-1]], 0))
	out = {"mu": [], "sig": [], "y": [], "scores": [], "base": 0., "slope": 0.}
	for b0 in range(0, len(rows), batch):
		r = rows[b0:b0+batch]
		family, params = evaluate_uq.sample_predictions(predictors, eeg[r], T, stochastic, split, int(r[0]))
		pred = evaluate_uq.Predictive(family, [tf.gather(flat(p), idx, axis=-1) for p in params])
		del params
		y = tf.gather(tf.reshape(tf.constant(fmri[r, ..., 0], tf.float32), (len(r), -1)), idx, axis=-1)
		out["family"], out["point"] = family, pred.point
		if(moments):
			out["mu"].append(pred.mu); out["sig"].append(pred.std()); out["y"].append(y)
		if(scores):
			sc = tf.gather(pred.conformity(y), pos_common, axis=-1).numpy()
			out["scores"].append(np.minimum(sc, F16_MAX).astype(np.float16))
			if(family == "quantile"):
				out["base"] += tf.reduce_sum(tf.gather(tf.reduce_mean(pred.p[2], 0)-tf.reduce_mean(pred.p[1], 0), pos_common, axis=-1), 0).numpy()
				out["slope"] += 2.*len(r)
			else:
				out["slope"] += tf.reduce_sum(2.*tf.gather(pred.std(), pos_common, axis=-1), 0).numpy()
	for k in ["mu", "sig", "y"]:
		out[k] = tf.concat(out[k], 0) if out[k] else None
	if(scores):
		out["scores"] = np.concatenate(out["scores"], 0)
		n = len(rows)
		out["base"] = (np.zeros(int(pos_common.shape[0])) if np.isscalar(out["base"]) else out["base"]/n).astype(np.float32)
		out["slope"] = (np.full(int(pos_common.shape[0]), out["slope"]/n) if np.isscalar(out["slope"]) else out["slope"]/n).astype(np.float32)
	return out


def mask_fractions(fmri, subjects, target, fold_m, common_m):
	"""C: each test subject's own brain mask vs the fold (and common) mask."""
	import evaluate_uq
	out = {}
	for s in np.unique(subjects):
		vol = fmri[subjects == s][..., 0]
		if(target == "run_zscore"):
			own, source = evaluate_uq.otsu_mask(vol.mean(0)), "otsu_own_mean"
		elif(target == "fluct"):
			own, source = vol.std(0) > 1e-6, "own_temporal_variance"
		else:
			out[int(s)] = {"own_mask_source": None}
			continue
		inter = (own & fold_m).sum()
		out[int(s)] = {"own_mask_source": source, "own_mask_fraction": float(own.mean()),
						"frac_own_in_fold_mask": float(inter/max(own.sum(), 1)),
						"frac_fold_mask_in_own": float(inter/max(fold_m.sum(), 1)),
						"frac_own_in_common_mask": float((own & common_m).sum()/max(own.sum(), 1))}
	return out


def _summary(per_subject):
	"""Mean over subjects of every numeric leaf (each subject weighs the same)."""
	leaves = {}
	def walk(d, prefix):
		for k, v in d.items():
			if(isinstance(v, dict)):
				walk(v, prefix+k+".")
			elif(isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)):
				leaves.setdefault(prefix+k, []).append(float(v))
	for v in per_subject.values():
		walk(v, "")
	return {k: float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan for k, v in leaves.items()}


# -------------------------------------------------------------------------------------------------------------- runs

def mode_runs(cli):
	from utils import process_utils
	process_utils.process_setup_tensorflow(cli.gpu_mem, seed=0)
	tf = _tf()
	import evaluate_uq
	runs = [r for r in runs_of(cli) if cli.overwrite or not all(os.path.exists(os.path.join(r[3], f)) for f in ["uq_dependence.json", "oof_scores.npz"])]
	if(not runs):
		print("I: nothing to do", flush=True); return
	data_opt = run_opt(cli, runs[0][4][0][0])
	for _, _, _, _, members in runs:
		for cfg, d in members:
			assert all(cfg.get(k) == getattr(data_opt, k) for k in DATA_KEYS if k in cfg), "data options differ across runs: "+d
	masks, common = fold_masks(data_opt)
	common_idx = np.where(common.reshape(-1))[0]
	nulls = [n for n in cli.nulls.split(",") if n]

	for fold in sorted({r[0] for r in runs}):
		o = copy.copy(data_opt); o.fold = fold
		splits, test, calib = experiment.load_split(o)
		m = fold_mask(splits)
		assert np.array_equal(m, masks[fold]), "fold mask differs from the cached one: delete "+masks_path(data_opt)
		fold_idx = np.where(m.reshape(-1))[0]
		pos_common = np.searchsorted(fold_idx, common_idx)
		assert np.array_equal(fold_idx[pos_common], common_idx), "common mask is not inside the fold mask"
		idx, pos_common_tf = tf.constant(fold_idx, tf.int32), tf.constant(pos_common, tf.int32)
		eeg_train, fmri_train, _ = splits["train"]
		eeg_shape, fmri_shape = (None,)+eeg_train.shape[1:], (None,)+fmri_train.shape[1:]
		eeg_test, fmri_test, sub_test = splits["test"]
		eeg_cal, fmri_cal, sub_cal = splits["calib"]
		C = mask_fractions(fmri_test, sub_test, data_opt.target, m, common)
		for (_, seed, null_seed, out_dir, members) in [r for r in runs if r[0] == fold]:
			for cfg, _ in members:
				assert sorted(cfg["test_subjects"]) == sorted(test) and sorted(cfg["calib_subjects"]) == sorted(calib), "fold split differs from the run's config"
			print("I: fold", fold, "seed", seed, "->", out_dir, flush=True)
			predictors, stochastic, T = build_predictors(cli, members, eeg_shape, fmri_shape, eeg_train.shape[0], null_seed)
			null_eeg, null_info = {}, {}
			for kind in nulls:
				null_eeg[kind], null_info[kind] = evaluate_uq.eeg_null(eeg_test, sub_test, kind, seed=null_seed)
			per_subject, scores, widths = {}, {}, {}
			family, point = None, None
			for s in np.unique(sub_test):
				rows = np.where(sub_test == s)[0]
				real = subject_pass(predictors, eeg_test, fmri_test, rows, T, stochastic, idx, pos_common_tf, "test", cli.batch, scores=True)
				family, point = real["family"], real["point"]
				scores[int(s)] = real["scores"]; widths[int(s)] = (real["base"], real["slope"])
				A = {"real": {} if point else dict(spread_stats(real["sig"]), **error_tracking(real["y"], real["mu"], real["sig"]))}
				for kind in nulls:
					nul = subject_pass(predictors, null_eeg[kind], fmri_test, rows, T, stochastic, idx, pos_common_tf, "test", cli.batch)
					A[kind] = {} if point else dict(spread_stats(nul["sig"]), **error_tracking(nul["y"], nul["mu"], nul["sig"]))
					A[kind]["vs_real"] = versus_real(real["mu"], real["sig"], nul["mu"], nul["sig"], point)
					del nul
				per_subject[int(s)] = {"A": A, "C": C[int(s)]}
				del real; gc.collect()
				print("I:   subject", int(s), json.dumps(_py({k: v.get("vs_real", {}).get("w2_spread_term_rel") for k, v in A.items()})), flush=True)
			#calibration subjects under the same model: the fold's 2-subject split conformal, for comparison in -combine
			cal_scores = {}
			for s in np.unique(sub_cal):
				cal = subject_pass(predictors, eeg_cal, fmri_cal, np.where(sub_cal == s)[0], T, stochastic, idx, pos_common_tf, "calib", cli.batch, moments=False, scores=True)
				cal_scores[int(s)] = cal["scores"]
			arrays = {"subjects": np.array(sorted(scores)), "calib_subjects": np.array(sorted(cal_scores)),
						"common_index": common_idx.astype(np.int32), "fold": np.array(fold), "seed": np.array(seed),
						"family": np.array(family), "levels": np.array(evaluate_uq.conformal_levels(family))}
			for s, v in scores.items():
				arrays["scores_"+str(s)] = v
				arrays["width_base_"+str(s)], arrays["width_slope_"+str(s)] = widths[s]
			for s, v in cal_scores.items():
				arrays["calib_scores_"+str(s)] = v
			os.makedirs(out_dir, exist_ok=True)
			tmp = os.path.join(out_dir, "oof_scores.tmp.npz")
			np.savez_compressed(tmp, **arrays)
			os.replace(tmp, os.path.join(out_dir, "oof_scores.npz"))
			_dump({"config": cli.name, "fold": fold, "seed": seed, "null_seed": null_seed, "T": T, "stochastic": stochastic,
					"family": family, "point": point, "null_info": null_info,
					"n_mask_fold": int(len(fold_idx)), "n_mask_common": int(len(common_idx)),
					"fold_mask_fraction": float(m.mean()), "common_mask_fraction": float(common.mean()),
					"summary": _summary(per_subject), "subjects": per_subject}, os.path.join(out_dir, "uq_dependence.json"))
			del predictors, null_eeg, scores, cal_scores; gc.collect()
			print("I: GPU peak memory (MB)", tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**20, flush=True)
		del splits; gc.collect()


# ----------------------------------------------------------------------------------------------- cross-conformal (B2)

def _coverage(cal, test, base, slope, levels):
	"""cal: (n_volumes, V) scores pooled over the calibration subjects; test: (n_test_volumes, V)."""
	from evaluate_uq import conformal_level
	n_glob, n_vox = cal.size, cal.shape[0]
	q_glob = np.quantile(cal.reshape(-1), [conformal_level(n_glob, l) for l in levels])
	q_vox = np.quantile(cal, [conformal_level(n_vox, l) for l in levels], axis=0)  #(L, V)
	return {"global_cov": [float(np.mean(test <= q)) for q in q_glob], "global_q": q_glob.tolist(),
			"global_width": [float(np.mean(base+slope*q)) for q in q_glob],
			"voxel_cov": [float(np.mean(test <= q[None])) for q in q_vox],
			"voxel_width": [float(np.mean(base+slope*q)) for q in q_vox],
			"n_calib_volumes": int(n_vox)}


def mode_combine(cli):
	root = os.path.join(cli.out_dir, cli.dataset, cli.name)
	files = sorted(glob.glob(os.path.join(root, "fold*", "seed*", "oof_scores.npz")))
	if(not files):
		raise SystemExit("no oof_scores.npz under "+root)
	by_seed = {}
	for path in files:
		with np.load(path) as f:
			by_seed.setdefault(int(f["seed"]), []).append(path)
	methods = ["cross_all_others", "fold_calib_same_model", "fold_calib_oof"]
	per_seed, levels, family, common = {}, None, None, None
	for seed, paths in sorted(by_seed.items()):
		scores, widths, fold_of, calib_same, calib_ids = {}, {}, {}, {}, {}
		for path in paths:
			with np.load(path) as f:
				if(common is None):
					common, levels, family = f["common_index"], f["levels"].tolist(), str(f["family"])
				assert np.array_equal(common, f["common_index"]), "different common masks: "+path
				k = int(f["fold"])
				for s in f["subjects"]:
					s = int(s)
					assert s not in scores, "subject %d is a test subject in two folds" % s
					scores[s] = f["scores_"+str(s)].astype(np.float32)
					widths[s] = (f["width_base_"+str(s)], f["width_slope_"+str(s)]); fold_of[s] = k
				calib_ids[k] = [int(c) for c in f["calib_subjects"]]
				calib_same[k] = np.concatenate([f["calib_scores_"+str(c)] for c in calib_ids[k]], 0).astype(np.float32)
		subjects = sorted(scores)
		print("I: seed", seed, "subjects", len(subjects), "folds", sorted(calib_ids), flush=True)
		stack = np.stack([scores[s] for s in subjects])  #(n_subjects, n_volumes, V)
		res = {}
		for i, s in enumerate(subjects):
			others = np.delete(stack, i, axis=0)
			others = others.reshape(-1, others.shape[-1])
			r = {"cross_all_others": dict(_coverage(others, scores[s], *widths[s], levels), n_calib_subjects=len(subjects)-1)}
			del others
			k = fold_of[s]
			r["fold_calib_same_model"] = dict(_coverage(calib_same[k], scores[s], *widths[s], levels), n_calib_subjects=len(calib_ids[k]))
			if(all(c in scores for c in calib_ids[k])):
				oof = np.concatenate([scores[c] for c in calib_ids[k]], 0)
				r["fold_calib_oof"] = dict(_coverage(oof, scores[s], *widths[s], levels), n_calib_subjects=len(calib_ids[k]))
			res[s] = r
		per_seed[seed] = res
		del stack; gc.collect()

	#average over seeds per subject, then mean/SD over subjects
	subjects = sorted({s for r in per_seed.values() for s in r})
	keys = ["global_cov", "voxel_cov", "global_width", "voxel_width"]
	per_subject = {}
	for s in subjects:
		per_subject[s] = {}
		for m in methods:
			rows = [per_seed[seed][s][m] for seed in per_seed if s in per_seed[seed] and m in per_seed[seed][s]]
			if(rows):
				per_subject[s][m] = {k: np.mean([r[k] for r in rows], 0).tolist() for k in keys}
				per_subject[s][m]["n_seeds"] = len(rows)
	summary = {}
	for m in methods:
		rows = [per_subject[s][m] for s in subjects if m in per_subject[s]]
		if(rows):
			summary[m] = {"n_subjects": len(rows)}
			for k in keys:
				v = np.array([r[k] for r in rows])
				summary[m][k+"_mean"], summary[m][k+"_sd"] = v.mean(0).tolist(), v.std(0, ddof=1).tolist() if len(rows) > 1 else None
				if(k.endswith("_cov")):
					summary[m][k+"_min"] = v.min(0).tolist()
	out = os.path.join(cli.out_dir, cli.name+"_cross_conformal.json")
	_dump({"config": cli.name, "dataset": cli.dataset, "family": family, "levels": levels, "seeds": sorted(per_seed),
			"n_common_voxels": int(len(common)), "summary": summary, "per_subject": per_subject, "per_seed": per_seed}, out)
	for m in methods:
		if(m in summary):
			print("==", m, "global", np.round(summary[m]["global_cov_mean"], 3).tolist(), "voxel", np.round(summary[m]["voxel_cov_mean"], 3).tolist(),
				"min voxel", np.round(summary[m]["voxel_cov_min"], 3).tolist())
	print("I: wrote", out)


# ------------------------------------------------------------------------------------------------ template mixture (D)

class SubjectNormal:
	"""One component of the template mixture: Normal(mean, SD) of one training subject's volumes, for every input."""
	def __init__(self, mu, sigma):
		tf = _tf()
		self.mu, self.sigma = tf.constant(mu[None, ..., None], tf.float32), tf.constant(sigma[None, ..., None], tf.float32)

	def __call__(self, eeg):
		tf = _tf()
		shape = tf.concat([[eeg.shape[0]], tf.shape(self.mu)[1:]], 0)
		return {"family": "gaussian", "params": [tf.broadcast_to(self.mu, shape), tf.broadcast_to(self.sigma, shape)]}


def template_mixture(splits):
	"""Equal-weight mixture over the training subjects, as a list of component predictors (one call each)."""
	fmri, subjects = splits["train"][1][..., 0], splits["train"][2]
	return [SubjectNormal(fmri[subjects == s].mean(0), fmri[subjects == s].std(0)+1e-4) for s in np.unique(subjects)]


def mode_template_mixture(cli):
	from utils import process_utils
	process_utils.process_setup_tensorflow(cli.gpu_mem, seed=0)
	import evaluate_uq
	for fold in _list(cli.folds) or list(range(cli.n_folds)):
		o = copy.copy(cli); o.fold = fold
		splits, _, _ = experiment.load_split(o)
		components = template_mixture(splits)
		o.name, o.seed = "template_mixture"+SUFFIX[o.target], 0
		out = experiment.run_dir(o)
		os.makedirs(out, exist_ok=True)
		res = evaluate_uq.evaluate(components, splits, T=len(components), stochastic=True, seed=0)
		evaluate_uq.save(res, out)
		print("== template_mixture fold", fold, "components", len(components)); evaluate_uq.print_summary(res)
		del splits, components, res; gc.collect()


def main():
	cli = parser.parse_args()
	mode = "combine" if cli.combine else cli.mode
	if(mode == "combine"):
		mode_combine(cli)
	elif(mode == "template_mixture"):
		mode_template_mixture(cli)
	elif(mode == "masks"):
		fold_masks(cli)
	else:
		mode_runs(cli)


if __name__ == "__main__":
	main()
