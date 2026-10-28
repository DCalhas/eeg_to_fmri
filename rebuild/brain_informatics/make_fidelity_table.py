"""Fidelity table for the Brain Informatics revision (review point 5): NODDI leave-one-subject-out, per arm
(trigger-aligned EEG, EEG paired from the file start, ECG only) and model, mean +/- SD over the 16 test subjects
(seeds averaged within a subject). Reads the per-fold results copied from Frida runs/synth_noddi into data/synth_noddi.
PSNR is derived per subject from the RMSE of [0, 1]-scaled volumes (20 log10(1/RMSE))."""
import os, json, glob, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "synth_noddi")
NAMES = {"template": "EEG-free template", "ridge": "Ridge (spectra)", "cnn_tag": "CNN-TAG", "e2fnet": "E2fNet", "spec2vol": "Spec2VolCAMU-Net"}

def collect(arm, suffix=""):
	per = collections.defaultdict(lambda: collections.defaultdict(list))  # model -> subject -> list of dicts
	for f in sorted(glob.glob(os.path.join(D, f"{arm}_loso_fold*_seed*{suffix}.json"))):
		base = os.path.basename(f)
		if "smoke" in base or (suffix == "" and ("_s2v" in base or "_saved" in base)):
			continue
		d = json.load(open(f)); sub = d["fold"]["test"][0] if isinstance(d["fold"], dict) else str(d["fold"])
		for m, v in d["results"].items():
			if m in NAMES and isinstance(v, dict):
				per[m][sub].append(v)
	return per

def summarise(per, m):
	subs = sorted(per[m])
	g = lambda k: np.array([np.mean([v[k] for v in per[m][s] if v.get(k) is not None]) for s in subs])
	out = {"n_subjects": len(subs), "n_runs": int(sum(len(per[m][s]) for s in subs))}
	ssim, rmse = g("ssim_2d_e2fnet"), g("rmse")
	out.update(ssim=(ssim.mean(), ssim.std()), rmse=(rmse.mean(), rmse.std()), psnr=((20 * np.log10(1 / rmse)).mean(), (20 * np.log10(1 / rmse)).std()))
	if all("voxel_r_mean" in v for s in subs for v in per[m][s]):
		r = g("voxel_r_mean"); out["r"] = (r.mean(), r.std())
		if m != "template":
			nul = np.array([np.mean([v["shift_null_mean"]["voxel_r_mean"] for v in per[m][s]]) for s in subs])
			out["r_null"] = (nul.mean(), nul.std())
			out["frac_p05"] = float(np.mean([np.mean([v["p_voxel_r"] < 0.05 for v in per[m][s]]) for s in subs]))
			ep = [v.get("best_epoch") for s in subs for v in per[m][s] if v.get("best_epoch") is not None]
			out["epoch_median"] = float(np.median(ep)) if ep else None
	return out

if __name__ == "__main__":
	rows = []
	res = {}
	for arm, label in [("aligned", "Aligned EEG"), ("filestart", "File-start EEG"), ("ecg_only", "ECG only")]:
		per = collect(arm)
		if arm == "aligned":
			per2 = collect(arm, "_s2v")
			per["spec2vol"] = per2["spec2vol"]
		for m in ["template", "ridge", "cnn_tag", "e2fnet", "spec2vol"]:
			if m in per and per[m]:
				s = summarise(per, m); res[f"{arm}/{m}"] = s
				rows.append((label, NAMES[m], s))
	json.dump(res, open(os.path.join(HERE, "data", "fidelity_summary.json"), "w"), indent=1, default=float)
	f2 = lambda t, d=3: f"${t[0]:.{d}f}\\pm{t[1]:.{d}f}$"
	lines = []
	last = None
	for label, name, s in rows:
		if name == "EEG-free template" and label != "Aligned EEG":
			continue
		arm = label if label != last else ""
		last = label
		r = f"${s['r'][0]:+.3f}$" if "r" in s else "--"
		rn = f"${s['r_null'][0]:+.3f}$" if "r_null" in s else "--"
		ep = f"{s['epoch_median']:.0f}" if s.get("epoch_median") is not None else "--"
		lines.append(f"        {arm} & {name} & {f2(s['ssim'])} & {f2(s['psnr'], 1)} & {r} & {rn} & {ep} \\\\")
	open(os.path.join(HERE, "submission", "table_fidelity_rows.tex"), "w").write("\n".join(lines) + "\n")
	for l in lines:
		print(l)
