"""Example-volume figure for the Brain Informatics revision (review point 5): held-out NODDI subject 49, final
synthesizers (the models used for the clinical cohorts) against the real volumes and the EEG-free template.
Input: runs/revision/examples.npz (revision_gpu.py --examples; copied locally, not versioned, 334 MB).
Output: submission/fig3.pdf."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "examples.npz")
d = {k: v.astype(np.float32) for k, v in np.load(SRC).items()}
Z = 15  # axial slice (of 30)
T = d["real"].shape[0] // 2
cols = [("real", "Real fMRI"), ("template", "EEG-free template"), ("cnn_tag", "CNN-TAG"), ("e2fnet", "E2fNet"),
		("spec2vol", "Spec2VolCAMU-Net"), ("ntvit", "NT-ViT")]
img = lambda V: np.rot90(V[Z])  # (64, 64) slice, anterior up

plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "pdf.fonttype": 42, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(6.3, 4.6))
gs = fig.add_gridspec(3, 7, width_ratios=[1] * 6 + [0.06], height_ratios=[1, 1, 0.95], hspace=0.28, wspace=0.06)
mask = d["real"].mean(0) > 0.15  # brain voxels, for the SD scale and the mean signal
sd_max = float(np.percentile(d["real"].std(0)[mask], 99))
for j, (k, name) in enumerate(cols):
	vol = d[k] if k == "template" else d[k][min(T, len(d[k]) - 1)]
	ax = fig.add_subplot(gs[0, j]); im0 = ax.imshow(img(vol), cmap="gray", vmin=0, vmax=1); ax.set_axis_off()
	ax.set_title(name, fontsize=7.5)
	ax = fig.add_subplot(gs[1, j]); ax.set_axis_off()
	if k == "template":
		ax.text(0.5, 0.5, "constant\n(SD = 0)", ha="center", va="center", transform=ax.transAxes, fontsize=7)
	else:
		im1 = ax.imshow(img(d[k].std(0)), cmap="magma", vmin=0, vmax=sd_max)
cax = fig.add_subplot(gs[0, 6]); fig.colorbar(im0, cax=cax).ax.tick_params(labelsize=6)
cax = fig.add_subplot(gs[1, 6]); fig.colorbar(im1, cax=cax).ax.tick_params(labelsize=6)
fig.text(0.005, 0.80, "A", fontweight="bold", fontsize=9); fig.text(0.005, 0.53, "B", fontweight="bold", fontsize=9)
fig.text(0.005, 0.27, "C", fontweight="bold", fontsize=9)
ax = fig.add_subplot(gs[2, :6])
z = lambda x: x - x.mean()  # demeaned, same intensity scale for all: near-constant outputs stay flat
t = np.arange(d["real"].shape[0]) * 2.16
ax.plot(t, z(d["real"][:, mask].mean(1)), color="black", lw=1.0, label="Real fMRI")
for k, name, c in [("cnn_tag", "CNN-TAG", "#d62728"), ("e2fnet", "E2fNet", "#1f77b4"), ("spec2vol", "Spec2VolCAMU-Net", "#2ca02c"), ("ntvit", "NT-ViT", "#9467bd")]:
	x = d[k][:, mask].mean(1)
	ax.plot(t[:len(x)], z(x), color=c, lw=0.7, alpha=0.85, label=name)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Brain-mean intensity\n(demeaned)")
for s in ("top", "right"):
	ax.spines[s].set_visible(False)
ax.legend(ncol=5, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
out = os.path.join(HERE, "submission", "fig3.pdf")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
r = {k: float(np.corrcoef(d["real"][:len(d[k]), mask].mean(1), d[k][:, mask].mean(1))[0, 1]) for k, _ in cols[2:]}
print(out, "brain-mean r with real:", {k: round(v, 3) for k, v in r.items()}, "real SD p99", round(sd_max, 4),
	  "model SD medians", {k: float(np.median(d[k].std(0)[mask])) for k, _ in cols[2:]}, "real", round(float(np.median(d['real'].std(0)[mask])), 4))
