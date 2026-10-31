"""
Voxelwise significance maps in the style of viz_utils.comparison_plot_3D_representation_projected_slices, from the saved
evaluations (no recomputation): for every test subject, the per-voxel temporal correlation r between predicted and
observed time courses (eval*.npz temporal_r_<subject>, stored for the fold's brain mask) is put back on the
64x64x30 grid, and two conditions are compared voxel by voxel with a paired Wilcoxon test over subjects.

Comparisons (registered volumes, so voxels correspond across subjects):
	real vs swap   - the model with the subject's own EEG against the same model with another subject's EEG
	real vs shift  - against the EEG circularly shifted in time
	model vs locked - the model against the EEG-free stimulus-locked template
Output: <out>/<name>.npz with res1, res2 (mean over subjects of 1 - r, lower is better, as residues), pvalues,
n (subjects per voxel) and anatomy (mean training volume of the native run z-score), all (64, 64, 30, 1).

python region_maps.py 05 -config mni_hetero_lr1e-4_b32_fluct -fmri_space mni -target fluct -out RUNS/region_maps
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
parser.add_argument('-config', required=True)
parser.add_argument('-locked', default=None, help="stimulus-locked template configuration (default: <prefix>stimulus_locked<suffix>)")
parser.add_argument('-seed_glob', default="seed*")
parser.add_argument('-out', required=True)
opt, _ = parser.parse_known_args()


def subject_maps(config, tag, masks):
	"""{subject: (64,64,30) r map averaged over seeds}, NaN outside the fold mask."""
	acc = {}
	for path in sorted(glob.glob(os.path.join(opt.out_dir, opt.dataset, config, "fold*", opt.seed_glob, "eval"+tag+".npz"))):
		fold = int(path.split("/fold")[1].split("/")[0])
		with np.load(path) as f:
			for k in f.files:
				if(not k.startswith("temporal_r_")):
					continue
				s = int(k.split("_")[-1]); m = np.full(masks[fold].size, np.nan, np.float32)
				m[np.where(masks[fold].reshape(-1))[0]] = f[k]
				acc.setdefault(s, []).append(m.reshape(masks[fold].shape))
	return {s: np.nanmean(v, 0) for s, v in acc.items()}


def compare(a, b, name, anatomy):
	common = sorted(set(a) & set(b))
	A = np.stack([a[s] for s in common]); B = np.stack([b[s] for s in common])
	ok = np.isfinite(A) & np.isfinite(B); n = ok.sum(0)
	p = np.ones(A.shape[1:], np.float32)
	for idx in zip(*np.where(n >= 6)):
		x, y = A[(slice(None),)+idx][ok[(slice(None),)+idx]], B[(slice(None),)+idx][ok[(slice(None),)+idx]]
		if(np.any(x != y)):
			p[idx] = wilcoxon(x, y).pvalue  #two-sided: the map colours whichever condition is better
	res1, res2 = 1-np.nanmean(np.where(ok, A, np.nan), 0), 1-np.nanmean(np.where(ok, B, np.nan), 0)
	res1, res2 = np.nan_to_num(res1, nan=1.), np.nan_to_num(res2, nan=1.)
	os.makedirs(opt.out, exist_ok=True)
	np.savez_compressed(os.path.join(opt.out, name+".npz"), res1=res1[..., None], res2=res2[..., None], pvalues=p[..., None], n=n[..., None],
						anatomy=anatomy[..., None], subjects=np.array(common))
	sig = (p < 0.05) & (n >= 6)
	print("I:", name, "subjects", len(common), "voxels tested", int((n >= 6).sum()), "p<0.05", int(sig.sum()),
		  "of which first better", int((sig & (res1 < res2)).sum()), flush=True)


masks, anatomy = {}, []
for fold in range(opt.n_folds):
	o = opt; o.fold = fold
	splits, test, calib = experiment.load_split(o)
	mean_train = splits["train"][1][..., 0].mean(0)
	if(splits.get("mask") is not None):
		masks[fold] = splits["mask"].astype(bool)
	else:
		from evaluate_uq import otsu_mask
		masks[fold] = otsu_mask(mean_train).astype(bool)
	anatomy.append(mean_train)
anatomy = np.mean(anatomy, 0)
if(opt.target != "run_zscore"):  #the fluctuation targets have no anatomy: use the run z-score mean for the background
	o = opt; o.target = "run_zscore"; o.fold = 0
	anatomy = experiment.load_split(o)[0]["train"][1][..., 0].mean(0)
	opt.target = "fluct"

suffix = {"run_zscore": "", "fluct": "_fluct", "fluct_resid": "_fluct_resid"}[opt.target]
prefix = "mni_" if opt.fmri_space == "mni" else ""
real, shift, swap = (subject_maps(opt.config, t, masks) for t in ["", "_null_shift", "_null_swap"])
compare(real, swap, opt.config+"_real_vs_swap", anatomy)
compare(real, shift, opt.config+"_real_vs_shift", anatomy)
locked = subject_maps(opt.locked or prefix+"stimulus_locked"+suffix, "", masks)
if(locked):
	compare(real, locked, opt.config+"_vs_locked", anatomy)
