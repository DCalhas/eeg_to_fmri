"""RQ1b: can EEG predict *preprocessed* BOLD fluctuations on NODDI? (fair target; CPU job)

Target: e2f.data.noddi_bold parcels (Harvard-Oxford cortical + subcortical GM + global; motion-regressed,
high-passed, z-scored) and, secondarily, masked voxels (PCA-reduced). Features per volume k, all from
trigger-aligned EEG covering the 20 TRs (43 s) before volume k (spans the HRF):
  bandpower  log power in delta/theta/alpha/beta/gamma per channel and TR frame (57 x 5 x 20), the classic
             EEG-informed fMRI regressor set with a full lag window
  spectral   PCA(256) of the full log|FFT| frames (57 x 97 x 20)
Controls: ecg (ECG band power, same lags), motion (6 params + FD, lags 0..5 TRs), and the circular-shift null.
Model: ridge (alpha on the 2 val subjects); LOSO folds from data/noddi/folds.json; training-only standardization.
Metric: Pearson r per parcel on the held-out subject; summaries over parcels; shift null by circularly
rolling the feature rows (>= 30 TRs) with the fitted model fixed.
"""
import os, json, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from e2f.data import noddi, noddi_bold
from e2f.synth.benchmark_noddi import make_folds

RUNS = os.path.expanduser("~/eeg_to_fmri/runs/synth_bold_v4")  # v4: + detrended scoring, 100 shifts, swap null (A20); v3: Friston-24, alpha by val R^2
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]


_COMMON = None


def common_parcels():
	"""Parcel names present in every subject (parcels with < 5 in-mask voxels are dropped per subject)."""
	global _COMMON
	if _COMMON is None:
		sets = [list(np.load(noddi_bold.build(s))["parcel_names"]) for s in noddi.subjects()]
		_COMMON = [n for n in sets[0] if all(n in t for t in sets)]
	return _COMMON


def features(sub, kind, n_frames=20):
	if kind.startswith("motion+"):
		a = features(sub, "motion", n_frames); b2 = features(sub, kind.split("+", 1)[1], n_frames)
		common, ia, ib = np.intersect1d(a[5], b2[5], return_indices=True)  # same volumes in both blocks
		return np.c_[a[0][ia], b2[0][ib]], a[1][ia], a[2][ia], a[3], a[4], common
	d = noddi.load(sub)
	b = np.load(noddi_bold.build(sub))
	if kind == "motion":
		m = np.c_[b["motion"], b["fd"]]
		idx = np.arange(n_frames, 300)
		X = np.stack([np.concatenate([m[k - l] for l in range(6)]) for k in idx])
	else:
		sig = d["ecg"] if kind == "ecg" else d["eeg"]
		F, idx, freqs = noddi.tr_frames(sig, d["onsets"], n_frames=n_frames)  # (V, C, F, T) log|FFT|
		keep = idx >= n_frames  # same volume set for every feature kind (motion lags need k >= 20 too)
		F, idx = F[keep], idx[keep]
		if kind in ("bandpower", "ecg"):
			pw = np.exp(2 * F)
			X = np.stack([np.log(pw[:, :, (freqs >= lo) & (freqs < hi)].mean(2)) for lo, hi in BANDS], 2)  # (V, C, 5, T)
			X = X.reshape(len(X), -1)
		else:
			X = F.reshape(len(F), -1)
	names = list(b["parcel_names"]); cols = [names.index(n) for n in common_parcels()]
	return X.astype(np.float32), b["parcels"][idx][:, cols], b["bold"][idx], b["mask"], common_parcels(), np.asarray(idx)


def corr_cols(P, Y):
	P = P - P.mean(0); Y = Y - Y.mean(0)
	den = np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0))
	return np.where(den > 0, (P * Y).sum(0) / np.where(den > 0, den, 1), np.nan)


def detrend_cols(A):
	"""Remove a per-column linear trend over time (shared slow drift can make real > shifted without neural content)."""
	from scipy.signal import detrend
	return detrend(np.asarray(A, np.float64), axis=0, type="linear")


def r2_cols(P, Y):
	"""Out-of-sample R^2 per column: 1 - SSE/SST (SST around the held-out subject's own mean; can be < 0)."""
	sse = ((Y - P) ** 2).sum(0); sst = ((Y - Y.mean(0)) ** 2).sum(0)
	return np.where(sst > 0, 1 - sse / np.where(sst > 0, sst, 1), np.nan), float(1 - sse.sum() / max(sst.sum(), 1e-12))


def run_fold(fold, kind, n_shifts, seed, target):
	load = lambda ss: [features(s, kind) for s in ss]
	tr, va, te = load(fold["train"]), load(fold["val"]), load(fold["test"])
	Xtr = np.concatenate([t[0] for t in tr]); Xva = np.concatenate([t[0] for t in va]); Xte = te[0][0]
	if target == "parcels":
		Ytr = np.concatenate([t[1] for t in tr]); Yva = np.concatenate([t[1] for t in va]); Yte = te[0][1]
	else:  # voxels: intersection of training masks is not needed; z-scored voxels, 0 outside each subject's mask
		Ytr = np.concatenate([t[2].reshape(len(t[2]), -1).astype(np.float32) for t in tr])
		Yva = np.concatenate([t[2].reshape(len(t[2]), -1).astype(np.float32) for t in va])
		Yte = te[0][2].reshape(len(te[0][2]), -1).astype(np.float32)
		mte = te[0][3].reshape(-1)
	sc = StandardScaler().fit(Xtr)
	Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
	pca_f = None
	if kind.endswith("spectral"):
		k0 = 42 if kind.startswith("motion+") else 0  # motion block (7 x 6 lags) kept as is, EEG block PCA-reduced
		pca = PCA(256, random_state=seed).fit(Xtr[:, k0:])
		pca_f = lambda X: np.c_[X[:, :k0], pca.transform(X[:, k0:])]
		Xtr, Xva, Xte = pca_f(Xtr), pca_f(Xva), pca_f(Xte)
	ypca = None
	if target == "voxels":
		ypca = PCA(256, random_state=seed).fit(Ytr)
		Ytr_, Yva_ = ypca.transform(Ytr), ypca.transform(Yva)
	else:
		Ytr_, Yva_ = Ytr, Yva
	best = None
	for a in ALPHAS:
		r = Ridge(alpha=a).fit(Xtr, Ytr_)
		pv = r.predict(Xva)
		pv = ypca.inverse_transform(pv) if ypca is not None else pv
		s = r2_cols(pv, Yva)[1]  # select on validation pooled R^2 (scale-aware); r is scale-free and under-shrinks
		if best is None or s > best[0]:
			best = (s, a, r)
	model = best[2]
	predict = lambda X: (ypca.inverse_transform(model.predict(X)) if ypca is not None else model.predict(X))
	Yte_d = detrend_cols(Yte)
	def score(X):
		P = detrend_cols(predict(X))  # targets and predictions linearly detrended before scoring
		rr = corr_cols(P, Yte_d); r2, r2g = r2_cols(P, Yte_d)
		if target == "voxels":
			rr, r2 = rr[mte], r2[mte]
			sse = ((Yte_d[:, mte] - P[:, mte]) ** 2).sum(); sst = ((Yte_d[:, mte] - Yte_d[:, mte].mean(0)) ** 2).sum()
			r2g = float(1 - sse / max(sst, 1e-12))
		return rr, r2, r2g
	real, r2, r2g = score(Xte)
	rng = np.random.default_rng(seed)
	T = len(Xte)
	def shifted():
		k = int(rng.integers(30, T - 30)); Xs = np.roll(Xte, k, axis=0)
		if kind.startswith("motion+"):  # null for the EEG/ECG block only: motion stays aligned
			Xs[:, :42] = Xte[:, :42]
		return Xs
	nulls = [score(shifted()) for _ in range(n_shifts)]
	null_r = [float(np.nanmean(n[0])) for n in nulls]; null_r2g = [n[2] for n in nulls]
	# swap null: features of the other held-out-fold subjects (val), truncated/tiled to the test length; keeps the
	# EEG's own slow dynamics but from a different person, so shared drift alone cannot beat it
	swaps = []
	for t in va:
		Xs = sc.transform(t[0]); Xs = pca_f(Xs) if pca_f is not None else Xs
		Xs = np.concatenate([Xs] * int(np.ceil(T / len(Xs))))[:T]
		if kind.startswith("motion+"):
			Xs[:, :42] = Xte[:, :42]
		swaps.append(score(Xs))
	swap_r = [float(np.nanmean(n[0])) for n in swaps]; swap_r2g = [n[2] for n in swaps]
	return {"alpha": best[1], "val_R2": float(best[0]), "r_mean": float(np.nanmean(real)), "r_median": float(np.nanmedian(real)),
			"r2corr_mean": float(np.nanmean(real ** 2)), "R2_mean": float(np.nanmean(r2)), "R2_median": float(np.nanmedian(r2)),
			"R2_frac_pos": float(np.nanmean(r2 > 0)), "R2_global": r2g, "R2_max": float(np.nanmax(r2)),
			"r_per_parcel": real.tolist() if target == "parcels" else None, "R2_per_parcel": r2.tolist() if target == "parcels" else None,
			"r_p95": float(np.nanpercentile(real, 95)), "shift_null_r": null_r, "shift_null_R2_global": null_r2g,
			"p_shift": float((1 + sum(v >= np.nanmean(real) for v in null_r)) / (1 + n_shifts)),
			"p_shift_R2_descriptive_only": float((1 + sum(v >= r2g for v in null_r2g)) / (1 + n_shifts)),
			"swap_null_r": swap_r, "swap_null_R2_global": swap_r2g, "detrended": True}


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--kinds", nargs="+", default=["bandpower", "spectral", "ecg", "motion", "motion+spectral", "motion+ecg"])
	ap.add_argument("--target", default="parcels", choices=["parcels", "voxels"])
	ap.add_argument("--n_shifts", type=int, default=100)
	ap.add_argument("--seed", type=int, default=0)
	a = ap.parse_args()
	folds = make_folds()["fold_sets"]["loso"]["folds"]
	os.makedirs(RUNS, exist_ok=True)
	names = features(noddi.subjects()[0], "motion")[4]
	for kind in a.kinds:
		out = os.path.join(RUNS, f"{a.target}_{kind}.json")
		if os.path.exists(out):
			continue
		res = []
		for i, f in enumerate(folds):
			r = run_fold(f, kind, a.n_shifts, a.seed, a.target)
			res.append({"test": f["test"], **r})
			print(a.target, kind, f["test"], {k: v for k, v in r.items() if not isinstance(v, list)}, flush=True)
		json.dump({"target": a.target, "kind": kind, "parcel_names": names, "folds": res}, open(out, "w"), indent=1)
