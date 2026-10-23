"""Shared EEG-to-fMRI fidelity metrics and EEG-free baselines (paper S RQ1 and paper U; ledger item A3).

Arrays are (T, X, Y, Z) per subject: T volumes of one held-out subject. All summaries are per subject;
CIs/tests must be over subjects, never over volumes (ledger A5).

Metrics
  rmse            sqrt(mean squared error) over voxels and volumes
  ssim_2d_e2fnet  torchmetrics SSIM, data_range=1, volume as Z-channel 2D image of X*Y (E2fNet convention; inputs in [0,1])
  ssim_3d         3D SSIM (gaussian 7^3) via skimage on each volume, data_range from target
  voxel_r         Pearson r over time per voxel (inside mask), summarised by mean and median: the only metric that a
                  static predictor (template / own mean) cannot score on (it gives r = nan/0)
Baselines (no EEG)
  template(train)        mean volume over training subjects (population template)
  own_mean(target)       the held-out subject's temporal mean (static upper bound; uses test anatomy)
  gaussian_template      voxelwise mean/var over training volumes, for NLL/CRPS scoring of predictive distributions
"""
import numpy as np

# Range convention (ledger C10): ssim_2d_e2fnet / psnr assume volumes min-max scaled to [0,1] per volume
# (E2fNet convention, data_range=1). No transductive statistics: any normalisation uses the volume itself or
# training subjects only. Metrics run on CUDA when available (GPU-only rule for GPU jobs).


def _device():
	import torch
	return "cuda" if torch.cuda.is_available() else "cpu"


def _t(x, dev):
	import torch
	return torch.as_tensor(np.array(x, dtype=np.float32), device=dev)  # copy: broadcast views are read-only


def rmse(pred, target):
	import torch
	dev = _device()
	p, t = _t(np.broadcast_to(pred, target.shape), dev), _t(target, dev)
	return float(torch.sqrt(((p - t) ** 2).mean()))


def ssim_2d_e2fnet(pred, target, batch=64):
	from torchmetrics.image import StructuralSimilarityIndexMeasure
	dev = _device()
	m = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
	tot, n = 0.0, 0
	pred = np.broadcast_to(pred, target.shape)
	for i in range(0, len(target), batch):
		t = _t(np.moveaxis(target[i:i + batch], -1, 1), dev)
		p = _t(np.moveaxis(pred[i:i + batch], -1, 1), dev)
		tot += m(p, t).item() * len(t); n += len(t)
	return tot / n


def ssim_3d(pred, target):
	from skimage.metrics import structural_similarity
	pred = np.broadcast_to(pred, target.shape)
	dr = float(target.max() - target.min())
	return float(np.mean([structural_similarity(p, t, data_range=dr, gaussian_weights=True, sigma=1.5) for p, t in zip(pred, target)]))


def voxel_r(pred, target, mask=None):
	"""Per-voxel Pearson r over time. Static predictions return nan (no temporal variance)."""
	import torch
	dev = _device()
	pred = np.broadcast_to(pred, target.shape)
	if mask is None:
		mask = target.std(0) > 0
	p, t = _t(pred[:, mask], dev), _t(target[:, mask], dev)
	p = p - p.mean(0); t = t - t.mean(0)
	den = torch.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
	r = torch.where(den > 0, (p * t).sum(0) / torch.where(den > 0, den, torch.ones_like(den)), torch.full_like(den, float("nan")))
	r = r.cpu().numpy()
	return {"mean": float(np.nanmean(r)) if np.isfinite(r).any() else float("nan"),
			"median": float(np.nanmedian(r)) if np.isfinite(r).any() else float("nan"), "r": r}


def template(train_subjects):
	"""train_subjects: list of (T_i, X, Y, Z) arrays -> mean of per-subject temporal means."""
	return np.mean([s.mean(0) for s in train_subjects], axis=0)


def own_mean(target):
	return target.mean(0)


def gaussian_template(train_subjects):
	"""Voxelwise predictive mean/variance pooled over all training volumes (EEG-free predictive distribution)."""
	v = np.concatenate(train_subjects, 0)
	return v.mean(0), v.var(0) + 1e-8


def gaussian_nll(mu, var, target):
	return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (target - mu) ** 2 / var)))


def gaussian_crps(mu, var, target):
	from scipy.stats import norm
	s = np.sqrt(var)
	z = (target - mu) / s
	return float(np.mean(s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def score(pred, target, conventions=("rmse", "ssim_2d_e2fnet", "voxel_r")):
	out = {}
	if "rmse" in conventions:
		out["rmse"] = rmse(np.broadcast_to(pred, target.shape), target)
	if "ssim_2d_e2fnet" in conventions:
		out["ssim_2d_e2fnet"] = ssim_2d_e2fnet(pred, target)
	if "ssim_3d" in conventions:
		out["ssim_3d"] = ssim_3d(pred, target)
	if "voxel_r" in conventions:
		vr = voxel_r(pred, target); out["voxel_r_mean"], out["voxel_r_median"] = vr["mean"], vr["median"]
	return out


# ---- Evaluation-time nulls (ledger A9): no retraining, applied to the EEG input of a trained synthesizer ----

def circular_shift(eeg, min_shift=30, rng=None):
	"""Circularly shift a subject's per-TR EEG sequence (T, ...) by k ~ U[min_shift, T - min_shift] TRs.
	Keeps EEG statistics and subject identity, breaks EEG-BOLD temporal correspondence."""
	rng = np.random.default_rng(rng)
	T = len(eeg)
	assert T > 2 * min_shift, (T, min_shift)
	k = int(rng.integers(min_shift, T - min_shift + 1))
	return np.roll(eeg, k, axis=0), k


def cross_subject_swap(eegs, rng=None):
	"""eegs: dict subject -> (T_s, ...). Returns dict subject -> EEG of a different subject (derangement),
	truncated/tiled to T_s. Breaks subject identity as well as timing."""
	rng = np.random.default_rng(rng)
	subs = list(eegs)
	while True:
		perm = rng.permutation(len(subs))
		if all(perm[i] != i for i in range(len(subs))):
			break
	out = {}
	for i, s in enumerate(subs):
		src = eegs[subs[perm[i]]]
		reps = int(np.ceil(len(eegs[s]) / len(src)))
		out[s] = np.concatenate([src] * reps, 0)[:len(eegs[s])]
	return out, {s: subs[perm[i]] for i, s in enumerate(subs)}


def null_report(predict, eeg, target, n_shifts=20, min_shift=30, rng=0, conventions=("rmse", "voxel_r")):
	"""predict: callable EEG (T, ...) -> fMRI (T, X, Y, Z). Scores real EEG vs circularly shifted EEG.
	The model 'uses EEG timing' only if real beats the shift distribution on voxel_r (and RMSE)."""
	real = score(predict(eeg), target, conventions)
	g = np.random.default_rng(rng)
	nulls = [score(predict(circular_shift(eeg, min_shift, g)[0]), target, conventions) for _ in range(n_shifts)]
	return {"real": real, "shift_null": {k: [n[k] for n in nulls] for k in real},
			"p_voxel_r": float((1 + sum(n["voxel_r_mean"] >= real["voxel_r_mean"] for n in nulls)) / (1 + n_shifts))
			if "voxel_r_mean" in real else None}
