"""Results figure for the Brain Informatics manuscript: AUC (95% bootstrap CI) per cohort for EEG band power and the
synthesized fMRI view of each synthesizer, trained and untrained. Reads the synth_clf result files copied from Frida
(~/eeg_to_fmri/runs/synth_clf/*_clf*.json) into data/synth_clf/. Writes submission/fig4.pdf."""
import os, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
COHORTS = [("fep1", "FEP1"), ("fep2", "FEP2"), ("semmelweis", "Semmelweis"), ("pd", "PD"), ("fribourg", "Fribourg")]
MODELS = [("cnn_tag", "CNN-TAG"), ("e2fnet", "E2fNet"), ("spec2vol", "Spec2VolCAMU-Net"), ("ntvit", "NT-ViT")]
BAND, SYN = "#3a3a3a", "#2b6cb0"  # band power: near-black; synthesized views: one blue (trained filled, untrained open)

res = {}
for f in glob.glob(os.path.join(HERE, "data", "synth_clf", "*.json")):
	c = os.path.basename(f).split("_clf")[0]
	for k, v in json.load(open(f)).items():
		if "+" not in k:
			res.setdefault(c, {})[k] = v

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
					 "ytick.major.width": 0.6, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, len(COHORTS), figsize=(6.3, 2.6), sharey=True)
rows = [("bandpower", "Band power")] + MODELS
for ax, (c, title) in zip(axes, COHORTS):
	r = res[c]
	ax.axvline(0.5, color="#999999", lw=0.6, ls=(0, (3, 2)), zorder=0)
	bp = r["bandpower"]["auc"]
	ax.axvline(bp, color=BAND, lw=0.5, alpha=0.35, zorder=0)
	for i, (key, _) in enumerate(rows):
		y = len(rows) - 1 - i
		if key == "bandpower":
			v = r[key]
			ax.errorbar(v["auc"], y, xerr=[[v["auc"] - v["auc_ci95"][0]], [v["auc_ci95"][1] - v["auc"]]], fmt="s", ms=4,
						color=BAND, mfc=BAND, elinewidth=0.9, capsize=0)
			continue
		for dy, suffix, mfc in [(0.17, "", SYN), (-0.17, "_untrained", "white")]:
			v = r[f"synth_{key}{suffix}"]
			ax.errorbar(v["auc"], y + dy, xerr=[[v["auc"] - v["auc_ci95"][0]], [v["auc_ci95"][1] - v["auc"]]], fmt="o",
						ms=3.6, color=SYN, mfc=mfc, mew=0.9, elinewidth=0.8, capsize=0)
	ax.set_title(title, fontsize=8)
	ax.set_xlim(0.12, 1.0)
	ax.set_xticks([0.3, 0.5, 0.7, 0.9])
	ax.tick_params(length=2.5)
	for s in ("top", "right"):
		ax.spines[s].set_visible(False)
	ax.set_xlabel("AUC")
axes[0].set_yticks(range(len(rows)))
axes[0].set_yticklabels([lab for _, lab in rows][::-1])
axes[0].set_ylim(-0.6, len(rows) - 0.4)
handles = [Line2D([], [], marker="s", ls="", color=BAND, ms=4, label="EEG band power"),
		   Line2D([], [], marker="o", ls="", color=SYN, ms=4, label="Synthesized fMRI, trained synthesizer"),
		   Line2D([], [], marker="o", ls="", color=SYN, mfc="white", ms=4, label="Synthesized fMRI, untrained (random weights)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
out = os.path.join(HERE, "submission", "fig4.pdf")
fig.savefig(out, bbox_inches="tight")

print(out)
