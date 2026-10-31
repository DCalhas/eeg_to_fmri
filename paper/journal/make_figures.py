"""
Figures of the journal draft from the saved summaries (no recomputation). python make_figures.py -> figures/*.pdf
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_tables import R, config_mean

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "figures")
os.makedirs(F, exist_ok=True)
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
	"ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "serif"})
GREY, BLUE, ORANGE, GREEN, RED, PURPLE = "#7f7f7f", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"


def oracle():
	with open(os.path.join(R, "oracle_05.json")) as f:
		o = json.load(f)
	fac = np.array(o["factors"])
	get = lambda key: np.array([[s[str(k)][key] for s in o["subjects"]] for k in fac])
	spatial, temporal, r = get("spatial_kept"), get("temporal_kept_median"), get("temporal_r_median")
	fig, ax = plt.subplots(figsize=(3.4, 2.9))
	for y, c, lab in [(spatial, BLUE, "volume energy kept"), (temporal, ORANGE, "voxel temporal variance kept (median)"), (r, GREEN, "voxel temporal $r$ of the projection (median)")]:
		m = y.mean(1)
		ax.plot(fac, m, "o-", color=c, ms=3, lw=1.2, label=lab)
		ax.fill_between(fac, y.min(1), y.max(1), color=c, alpha=0.15, lw=0)
	ax.axvline(2, color=GREY, lw=0.8, ls=":")
	ax.text(2.05, 0.05, "$\\kappa=2$", color=GREY, fontsize=6.5)
	ax.set_xscale("log", base=2); ax.set_xticks(fac); ax.set_xticklabels(["%g" % f for f in fac])
	ax.set_xlabel("relaxation factor (DCT cutoff per axis)"); ax.set_ylim(0, 1.02); ax.set_ylabel("fraction")
	ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.25))
	fig.tight_layout(); fig.savefig(os.path.join(F, "oracle_bound.pdf"), format="pdf"); plt.close(fig)


def reliability():
	rows = [("Population template", "eeg_free_template", GREY, "--"), ("Heteroscedastic", "zerofill_hetero", BLUE, "-"),
		("MC dropout", "zerofill_mcdropout", ORANGE, "-"), ("Stochastic spectral", "spectral", GREEN, "-"),
		("SWAG-diag", "zerofill_swag", PURPLE, "-"), ("Deep ensemble", "zerofill_hetero_ensemble", RED, "-"),
		("Spectral, voxel head", "spectral_voxelhead", GREEN, ":")]
	fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
	for label, c, col, ls in rows:
		picp = np.array([config_mean(c, "picp", i)[0] for i in range(len(LEVELS))], float)
		conf = np.array([config_mean(c, "conf_voxel_cov", i)[0] for i in range(len(LEVELS))], float)
		axes[0].plot(LEVELS, picp, ls, color=col, lw=1.2, marker="o", ms=2.5, label=label)
		axes[1].plot(LEVELS, conf, ls, color=col, lw=1.2, marker="o", ms=2.5, label=label)
	for ax, t in zip(axes, ["model predictive intervals (PICP)", "per-voxel split conformal on calibration subjects"]):
		ax.plot([0, 1], [0, 1], color="k", lw=0.6); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
		ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage (test subjects)"); ax.set_title(t)
	axes[1].legend(frameon=False, loc="lower right")
	fig.tight_layout(); fig.savefig(os.path.join(F, "reliability.pdf"), format="pdf"); plt.close(fig)


def nulls():
	with open(os.path.join(R, "uq_grid_report.json")) as f:
		rep = json.load(f)
	rows = [("Heteroscedastic", "zerofill_hetero"), ("Evidential", "zerofill_evidential"), ("CQR", "zerofill_cqr"),
		("MC dropout", "zerofill_mcdropout"), ("Stoch. spectral", "spectral"), ("SWAG-diag", "zerofill_swag"),
		("LL Laplace", "zerofill_laplace"), ("Ensemble", "zerofill_hetero_ensemble"), ("Spectral, voxel head", "spectral_voxelhead"),
		("ECG only", "zerofill_hetero_ecg_only")]
	rows = [(l, c) for l, c in rows if c in rep]
	x = np.arange(len(rows)); w = 0.26
	real = [rep[c]["temporal_r_vs_shift"]["model"] for _, c in rows]
	shift = [rep[c]["temporal_r_vs_shift"]["reference"] for _, c in rows]
	swap = [rep[c]["temporal_r_vs_swap"]["reference"] for _, c in rows]
	locked = rep[rows[0][1]]["vs_stimulus_locked"]["temporal_r"]["reference"]
	fig, ax = plt.subplots(figsize=(6.8, 2.2))
	ax.bar(x-w, real, w, color=BLUE, label="real EEG")
	ax.bar(x, shift, w, color=ORANGE, label="EEG circularly shifted")
	ax.bar(x+w, swap, w, color=GREY, label="EEG of another subject")
	ax.axhline(locked, color=RED, lw=1, ls="--")
	ax.text(len(rows)-0.5, locked-0.006, "stimulus-locked template (no EEG), $r=%.2f$" % locked, color=RED, ha="right", va="top", fontsize=6.5)
	ax.axhline(0, color="k", lw=0.5)
	ax.set_xticks(x); ax.set_xticklabels([l for l, _ in rows], rotation=25, ha="right")
	ax.set_ylabel("median voxel temporal $r$"); ax.set_ylim(-0.02, 0.19)
	ax.legend(frameon=False, ncol=3, loc="upper left")
	fig.tight_layout(); fig.savefig(os.path.join(F, "nulls.pdf"), format="pdf"); plt.close(fig)


def ridge():
	def load(name):
		p = os.path.join(R, "ridge_mni", "ridge_"+name+".json")
		with open(p) as f:
			return json.load(f)["subjects"]
	fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
	for ax, design, title in [(axes[0], "within", "within subject"), (axes[1], "cross", "leave one subject out")]:
		cols = [("EEG", load(design+"_resid_eeg_composite"), BLUE), ("heart\nrate", load(design+"_resid_hr_composite"), RED),
			("ECG\nband power", load(design+"_resid_ecg_composite"), PURPLE), ("motion", load(design+"_resid_motion_composite"), ORANGE)]
		subs = sorted(cols[0][1])
		for i, (lab, d, c) in enumerate(cols):
			r = np.array([d[s]["r"] for s in subs]); sig = np.array([d[s]["p_r"] < 0.05 for s in subs])
			jit = (np.random.RandomState(i).rand(len(r))-0.5)*0.25
			ax.scatter(i+jit[~sig], r[~sig], s=10, facecolors="none", edgecolors=c, lw=0.8)
			ax.scatter(i+jit[sig], r[sig], s=12, color=c)
			ax.plot([i-0.25, i+0.25], [r.mean()]*2, color="k", lw=1.2)
		labels = [c[0] for c in cols]; ticks = list(range(len(cols)))
		if(np.isfinite(cols[0][1][subs[0]].get("swap_r", np.nan))):
			sw = np.array([cols[0][1][s]["swap_r"] for s in subs]); x0 = len(cols)
			ax.scatter(x0+(np.random.RandomState(8).rand(len(sw))-0.5)*0.25, sw, s=8, color=GREY, marker="+", lw=0.7)
			ax.plot([x0-0.2, x0+0.2], [sw.mean()]*2, color=GREY, lw=1.2); labels.append("EEG of\nanother subj."); ticks.append(x0)
		ax.set_xticks(ticks)
		ax.set_xticklabels(labels); ax.axhline(0, color="k", lw=0.5); ax.set_title(title)
	axes[0].set_ylabel("motor-ROI BOLD $r$ (test)")
	fig.tight_layout(); fig.savefig(os.path.join(F, "ridge.pdf"), format="pdf"); plt.close(fig)


def curves():
	"""Training loss and held-out (calibration subjects) RMSE per epoch, mean +- SD over folds and seeds (review #2c)."""
	rows = [("Heteroscedastic", "zerofill_hetero", BLUE), ("MC dropout", "zerofill_mcdropout", ORANGE), ("Stochastic spectral", "spectral", GREEN),
		("Evidential", "zerofill_evidential", PURPLE), ("Point (L1)", "zerofill", GREY)]
	load = lambda c: [json.load(open(f)) for f in glob.glob(os.path.join(R, "summaries", c, "fold*", "seed*", "curves.json"))]
	def band(ax, cs, key, col, label=None):
		Y = np.array([x[key] for x in cs if len(x.get(key, [])) == len(cs[0][key])]); e = np.arange(1, Y.shape[1]+1)
		ax.plot(e, Y.mean(0), color=col, lw=1.2, label=label); ax.fill_between(e, Y.mean(0)-Y.std(0), Y.mean(0)+Y.std(0), color=col, alpha=0.12, lw=0)
	fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
	for label, c, col in rows:
		cs = load(c)
		if(cs):
			band(axes[0], cs, "train", col, label); band(axes[1], cs, "val_rmse", col)
	axes[0].set_title("run z-score: training loss"); axes[1].set_title("run z-score: RMSE, calibration subj.")
	cs = load("zerofill_hetero_fluct")
	if(cs):
		band(axes[2], cs, "train", RED, "training loss"); tw = axes[2].twinx(); band(tw, cs, "val_rmse", BLUE, "RMSE, calibration subj.")
		tw.spines["right"].set_visible(True); tw.set_ylabel("RMSE", color=BLUE); axes[2].set_ylabel("loss", color=RED)
	axes[2].set_title("fluctuation target")
	for ax in axes:
		ax.set_xlabel("epoch")
	axes[0].legend(frameon=False, fontsize=6.5)
	fig.tight_layout(); fig.savefig(os.path.join(F, "curves.pdf"), format="pdf"); plt.close(fig)


def coverage_strip():
	"""Per-subject coverage of the per-voxel split-conformal 90% interval (review #7)."""
	path = os.path.join(R, "effect_report.json")
	if(not os.path.exists(path)):
		return
	e = json.load(open(path))
	rows = [("Template", None), ("Heteroscedastic", "zerofill_hetero"), ("MC dropout", "zerofill_mcdropout"), ("Stoch. spectral", "spectral"),
		("SWAG-diag", "zerofill_swag"), ("Ensemble", "zerofill_hetero_ensemble")]
	fig, ax = plt.subplots(figsize=(4.6, 2.3))
	for i, (label, c) in enumerate(rows):
		v = np.array(list((e["coverage90_template_per_subject"] if c is None else e["configs"][c]["coverage90_per_subject"]).values()), float)
		ax.scatter(i+(np.random.RandomState(i).rand(len(v))-0.5)*0.3, v, s=9, color=GREY if c is None else BLUE)
		ax.plot([i-0.25, i+0.25], [np.nanmean(v)]*2, color="k", lw=1.2)
	ax.axhline(0.9, color=RED, lw=0.8, ls="--"); ax.set_ylim(0, 1)
	ax.set_xticks(range(len(rows))); ax.set_xticklabels([l for l, _ in rows], rotation=25, ha="right")
	ax.set_ylabel("coverage of the 90% interval")
	fig.tight_layout(); fig.savefig(os.path.join(F, "coverage_subjects.pdf"), format="pdf"); plt.close(fig)


def volume_panel():
	"""Synthesized volumes in the style of the thesis figures (viz_utils, nipy_spectral): rows are axial slices of one test
	subject's volume; columns are the observed volume, the EEG-free template, the heteroscedastic and stochastic spectral
	predictions (with the subject's EEG and with another subject's), and the spectral decoder's epistemic and aleatoric SD."""
	V = os.path.join(R, "volmaps")
	load = lambda p, i: np.load(os.path.join(V, p))["map_0"][i].astype(np.float32)
	cols = [("Observed", load("eeg_free_template/fold0/seed0/eval.npz", 0), "v"),
		("EEG-free\ntemplate", load("eeg_free_template/fold0/seed0/eval.npz", 1), "v"),
		("Heteroscedastic", load("zerofill_hetero/fold0/seed2/eval.npz", 1), "v"),
		("Heteroscedastic,\nswapped EEG", load("zerofill_hetero/fold0/seed2/eval_null_swap.npz", 1), "v"),
		("Spectral\ndecoder", load("spectral/fold0/seed2/eval.npz", 1), "v"),
		("Spectral:\nepistemic SD", load("spectral/fold0/seed2/eval.npz", 2), "e"),
		("Spectral:\naleatoric SD", load("spectral/fold0/seed2/eval.npz", 3), "a")]
	vmin, vmax = np.percentile(cols[0][1], [1, 99.5])
	amax = np.percentile(cols[6][1], 99.5)
	slices = [25, 20, 15, 10, 5]
	fig, axes = plt.subplots(len(slices), len(cols), figsize=(7.2, 5.4), gridspec_kw={"wspace": 0.03, "hspace": 0.03})
	for j, (title, vol, kind) in enumerate(cols):
		for i, z in enumerate(slices):
			ax = axes[i, j]
			emax = max(float(cols[5][1].max()), 1e-6)
			lim = (vmin, vmax) if kind == "v" else ((0, amax) if kind == "a" else (0, emax))
			ax.imshow(np.rot90(vol[:, :, z]), cmap=plt.cm.nipy_spectral, vmin=lim[0], vmax=lim[1], interpolation="nearest")
			ax.set_xticks([]); ax.set_yticks([])
			for sp in ax.spines.values():
				sp.set_visible(False)
			if(i == 0):
				ax.set_title(title+("\n(max %.1g)" % max(float(cols[5][1].max()), 0.) if kind == "e" else ""), fontsize=6)
			if(j == 0):
				ax.set_ylabel("slice %d/30" % z, fontsize=6.5)
	fig.savefig(os.path.join(F, "volumes.pdf"), format="pdf", bbox_inches="tight", dpi=200); plt.close(fig)


def significance_maps():
	"""The thesis-style voxelwise significance maps, drawn with the authors' own viz_utils function (loaded from source
	without its TensorFlow imports): magenta/cyan where one condition has a significantly better voxel temporal r."""
	import ast
	from matplotlib.gridspec import GridSpec
	from matplotlib.colors import ListedColormap
	from scipy.ndimage import rotate
	src = open(os.path.join(HERE, "..", "..", "src", "utils", "viz_utils.py")).read()
	fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "comparison_plot_3D_representation_projected_slices"][0]
	ns = {"np": np, "plt": plt, "GridSpec": GridSpec, "ListedColormap": ListedColormap, "rotate": rotate}
	exec(compile(ast.Module(body=[fn], type_ignores=[]), "viz_utils", "exec"), ns)
	plot = ns["comparison_plot_3D_representation_projected_slices"]
	for name, m1, m2 in [("mni_hetero_lr1e-4_b32_fluct_real_vs_swap", "own EEG", "other subject's EEG"),
		("mni_hetero_lr1e-4_b32_fluct_real_vs_shift", "own EEG", "shifted EEG"), ("mni_hetero_fluct_vs_locked", "network", "stimulus-locked")]:
		path = os.path.join(R, "region_maps", name+".npz")
		if(not os.path.exists(path)):
			continue
		z = dict(np.load(path))
		#brain voxels of the acquired slab only: the analysis mask (Otsu of the EPI) also contains the scalp above the brain,
		#and the oblique 16-slice slab misses the ventral brain; background = MNI brain mask on the model grid AND tested voxels
		brain = np.load(os.path.join(R, "region_maps", "mni_brain_grid.npy")) > 0.5
		shown = brain & (z["n"][..., 0] >= 6)
		zs = np.where(shown.any((0, 1)))[0]; sl = slice(zs[0], zs[-1]+1)
		crop = lambda a: a[:, :, sl].copy()
		#display scaling only: the function colours a winning voxel with intensity 1 - residue; with residue = 1 - r that is
		#r itself (~0.05, near black), so both conditions' r are rescaled jointly to [0, 1] before drawing (p-values unchanged)
		r1, r2 = 1-z["res1"], 1-z["res2"]
		lo, hi = np.percentile(np.r_[r1[shown[..., None]], r2[shown[..., None]]], [1, 99])
		disp = lambda r: 1-0.95*np.clip((r-lo)/(hi-lo+1e-9), 0, 1)
		background = shown[..., None].astype(np.float32)
		p = np.where(shown[..., None], z["pvalues"], 1.)
		sig = (p[..., 0] < 0.05) & shown
		print("I:", name, "brain voxels tested", int(shown.sum()), "p<0.05", int(sig.sum()), "first better", int((sig & (z["res1"][..., 0] < z["res2"][..., 0])).sum()), "slices", zs[0], zs[-1])
		#in the function, a voxel is cyan when res1 < res2 (the first condition has the lower 1 - r) and magenta otherwise,
		#while the legend prints model1 at the magenta end: labels are passed so each colour names the condition that wins
		fig = plot(crop(disp(r1)), crop(disp(r2)), crop(p), crop(background), model1=m2, model2=m1, threshold=0.5, factor=1,
				   save=True, save_path=os.path.join(F, "sig_"+name+".pdf"), save_format="pdf")
		plt.close(fig)


if __name__ == "__main__":
	oracle(); reliability(); nulls(); ridge(); curves(); coverage_strip(); volume_panel(); significance_maps()
	print(sorted(os.listdir(F)))
