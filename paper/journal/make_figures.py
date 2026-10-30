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
	fig.tight_layout(); fig.savefig(os.path.join(F, "oracle_bound.pdf")); plt.close(fig)


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
	fig.tight_layout(); fig.savefig(os.path.join(F, "reliability.pdf")); plt.close(fig)


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
	fig.tight_layout(); fig.savefig(os.path.join(F, "nulls.pdf")); plt.close(fig)


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
	fig.tight_layout(); fig.savefig(os.path.join(F, "ridge.pdf")); plt.close(fig)


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
	fig.tight_layout(); fig.savefig(os.path.join(F, "curves.pdf")); plt.close(fig)


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
	fig.tight_layout(); fig.savefig(os.path.join(F, "coverage_subjects.pdf")); plt.close(fig)


if __name__ == "__main__":
	oracle(); reliability(); nulls(); ridge(); curves(); coverage_strip()
	print(sorted(os.listdir(F)))
