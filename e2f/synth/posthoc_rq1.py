"""Post-hoc null scoring of saved RQ1 image-protocol models (ledger post-hoc v1 requests; GPU job).

For each saved fold model (benchmark_noddi --save_models), re-predict the held-out subject under:
  real                 its own aligned EEG
  shift (N=100)        EEG rows circularly shifted >= 30 TRs (breaks time locking, keeps the prediction set)
  swap                 another subject's EEG (the two val subjects), truncated/tiled; keeps volume-index alignment
and score the dynamics on
  steady-state volumes only (volume index >= 10),
  linearly detrended target and prediction,
  optionally Friston-24 motion traces (from the RQ1b flirt estimates of the same run) regressed out of both,
in three voxel sets: all in-brain voxels, edge voxels (brain mask minus its 2-voxel erosion), interior voxels.
Statistics: pooled R2c (= 2 cov / SST difference vs a shift), variance-free pooled r (mean of per-voxel r) and
median voxel r; permutation p vs shifts with the +1 correction; swap reported alongside.
"""
import os, sys, json, argparse, glob
import numpy as np
import torch
from scipy import ndimage

from e2f.data import noddi, noddi_bold
from e2f.eval import fidelity as F
from e2f.eval.surrogates import perm_p
from e2f.synth import benchmark_noddi as BN

RUNS = BN.RUNS
OUT = os.path.join(RUNS, "posthoc")


def friston24(sub, idx):
	b = np.load(noddi_bold.build(sub))
	m = b["motion"].astype(np.float64)
	dm = np.r_[np.zeros((1, 6)), np.diff(m, axis=0)]
	C = np.c_[m, dm, m ** 2, dm ** 2][idx]
	return np.c_[np.ones(len(C)), C - C.mean(0)]


def projector(C, device):
	"""Residual-forming matrix I - C (C'C)^-1 C' for nuisance regression along time (on the GPU)."""
	C = torch.as_tensor(C, dtype=torch.float64, device=device)
	return torch.eye(len(C), dtype=torch.float64, device=device) - C @ torch.linalg.pinv(C)


def dyn_stats(P, Y, sets):
	"""P, Y: (T, V) torch tensors, already prepared. Per voxel set: pooled R2c, mean r, median r (all on the GPU)."""
	out = {}
	for name, sel in sets.items():
		p, y = P[:, sel], Y[:, sel]
		pc, yc = p - p.mean(0), y - y.mean(0)
		sst = (yc ** 2).sum()
		r2c = 1 - ((yc - pc) ** 2).sum() / torch.clamp(sst, min=1e-12)
		den = torch.sqrt((pc ** 2).sum(0) * (yc ** 2).sum(0))
		r = torch.where(den > 0, (pc * yc).sum(0) / torch.where(den > 0, den, torch.ones_like(den)), torch.full_like(den, float("nan")))
		out[name] = {"R2c": float(r2c), "r_mean": float(torch.nanmean(r)), "r_median": float(torch.nanmedian(r))}
	return out


def score_fold(arm, name, fold_i, seed, n_shifts, n_skip, motion, device="cuda"):
	folds = BN.make_folds()["fold_sets"]["loso"]["folds"]; fold = folds[fold_i]
	st = np.load(os.path.join(RUNS, "models", f"{arm}_fold{fold_i:02d}_seed{seed}_featstats.npz"))
	z = lambda X: ((X - st["mu"]) / st["sd"]).astype(np.float32)
	sub = fold["test"][0]
	Xte, Yte = BN.subject_arrays(sub, arm)
	_, idx, _ = noddi.tr_frames(noddi.load(sub)["eeg"], noddi.load(sub)["onsets"] if arm != "filestart" else np.arange(300) * noddi.TR, n_frames=20)
	keep = idx >= n_skip
	Xte, Yte, idx = z(Xte)[keep], Yte[keep], idx[keep]
	ck = torch.load(os.path.join(RUNS, "models", f"{arm}_{name}_fold{fold_i:02d}_seed{seed}.pt"), map_location="cpu", weights_only=False)
	model = BN.build_model(name, tuple(ck["x_shape"])); model.load_state_dict(ck["state_dict"]); model.to(device).eval()
	predict = lambda X: torch.as_tensor(np.clip(BN.predict(model, X, device), 0, 1).reshape(len(X), -1), dtype=torch.float64, device=device)
	Yn = Yte.reshape(len(Yte), -1)
	brain = Yn.std(0) > 0
	interior = ndimage.binary_erosion(brain.reshape(Yte.shape[1:]), iterations=2).reshape(-1)
	sets = {k: torch.as_tensor(v, device=device) for k, v in {"all": brain, "edge": brain & ~interior, "interior": interior & brain}.items()}
	T = len(Xte)
	t = np.arange(T, dtype=np.float64)
	base = np.c_[np.ones(T), (t - t.mean()) / t.std()]  # linear detrend
	Cfull = np.c_[base, friston24(sub, idx)[:, 1:]] if motion else base
	R = projector(Cfull, device)  # detrend (+ Friston-24) residualization, applied to target and prediction alike
	prep = lambda A: R @ A
	Yp = prep(torch.as_tensor(Yn, dtype=torch.float64, device=device))
	real = dyn_stats(prep(predict(Xte)), Yp, sets)
	rng = np.random.default_rng(seed)
	shifts = [dyn_stats(prep(predict(np.roll(Xte, int(rng.integers(30, T - 30)), axis=0))), Yp, sets) for _ in range(n_shifts)]
	swaps = []
	for v in fold["val"]:
		Xv = z(BN.subject_arrays(v, arm)[0]); Xv = np.concatenate([Xv] * int(np.ceil(T / len(Xv))))[:T]
		swaps.append(dyn_stats(prep(predict(Xv)), Yp, sets))
	res = {"test": sub, "model": name, "arm": arm, "seed": seed, "n_skip": n_skip, "motion_regressed": motion, "real": real, "swap": swaps}
	for s_ in sets:
		for k in ("R2c", "r_mean", "r_median"):
			nul = [x[s_][k] for x in shifts]
			res.setdefault("shift_mean", {}).setdefault(s_, {})[k] = float(np.mean(nul))
			res.setdefault("p_shift", {}).setdefault(s_, {})[k] = perm_p(real[s_][k], nul)
	return res


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--arm", default="aligned")
	ap.add_argument("--models", nargs="+", default=["cnn_tag", "e2fnet"])
	ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
	ap.add_argument("--n_shifts", type=int, default=100)
	ap.add_argument("--n_skip", type=int, default=10)
	a = ap.parse_args()
	assert torch.cuda.is_available(), "GPU-only rule"
	os.makedirs(OUT, exist_ok=True)
	for seed in a.seeds:
		for name in a.models:
			for motion in (False, True):
				out = os.path.join(OUT, f"{a.arm}_{name}_seed{seed}_motion{int(motion)}.json")
				if os.path.exists(out):
					continue
				res = [score_fold(a.arm, name, i, seed, a.n_shifts, a.n_skip, motion) for i in range(16)]
				json.dump(res, open(out, "w"), indent=1)
				print("posthoc", a.arm, name, seed, "motion", motion, flush=True)
