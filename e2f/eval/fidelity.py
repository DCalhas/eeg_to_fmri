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


def rmse(pred, target):
	return float(np.sqrt(np.mean((pred - target) ** 2)))


def ssim_2d_e2fnet(pred, target, batch=32):
	import torch
	from torchmetrics.image import StructuralSimilarityIndexMeasure
	m = StructuralSimilarityIndexMeasure(data_range=1.0)
	tot, n = 0.0, 0
	for i in range(0, len(target), batch):
		t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(target[i:i + batch], -1, 1))).float()
		p = torch.from_numpy(np.ascontiguousarray(np.moveaxis(np.broadcast_to(pred, target.shape)[i:i + batch], -1, 1))).float()
		tot += m(p, t).item() * len(t); n += len(t)
	return tot / n


def ssim_3d(pred, target):
	from skimage.metrics import structural_similarity
	pred = np.broadcast_to(pred, target.shape)
	dr = float(target.max() - target.min())
	return float(np.mean([structural_similarity(p, t, data_range=dr, gaussian_weights=True, sigma=1.5) for p, t in zip(pred, target)]))


def voxel_r(pred, target, mask=None):
	"""Per-voxel Pearson r over time. Static predictions return nan (no temporal variance)."""
	pred = np.broadcast_to(pred, target.shape)
	if mask is None:
		mask = target.std(0) > 0
	p, t = pred[:, mask], target[:, mask]
	p = p - p.mean(0); t = t - t.mean(0)
	den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
	r = np.where(den > 0, (p * t).sum(0) / np.where(den > 0, den, 1), np.nan)
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
