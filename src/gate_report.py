"""
Gate decision for the full grid (audit A11), from saved per-volume evaluations (no recomputation).

Per test subject (seed-averaged), for each model configuration:
	* vs the EEG-free template: rmse_brain, nll, crps, conf_voxel_cov@0.9 (and vs the template mixture of
	  reevaluate_uq.py -mode template_mixture when it exists; not part of the gate)
	* vs its own evaluation-time nulls (EEG circularly shifted / swapped across subjects): median voxel temporal r, nll
Paired Wilcoxon signed-rank tests over subjects. The gate passes for a configuration if it beats the template on NLL
and CRPS and its temporal r exceeds the shift null (both p < 0.05, one-sided).

python gate_report.py RUNS/cv/05 -configs zerofill,zerofill_hetero,zerofill_hetero_e30
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon


def per_subject(root, config, tag=""):
	"""-> {subject: {metric: value}} averaged over seeds."""
	values = {}
	for path in glob.glob(os.path.join(root, config, "fold*", "seed*", "eval"+tag+".npz")):
		with np.load(path) as f:
			sub = f["subject"]
			for s in np.unique(sub):
				row = values.setdefault(int(s), {})
				for k in ["rmse_brain", "nll", "crps", "ssim"]:
					row.setdefault(k, []).append(np.nanmean(f[k][sub == s]))
				row.setdefault("conf90", []).append(np.nanmean(f["conf_voxel_cov"][sub == s, 8]))
				row.setdefault("temporal_r", []).append(float(np.nanmedian(f["temporal_r_"+str(s)])))
				if("R2_values" in f.files):
					k = list(f["R2_keys"]); v = f["R2_values"][list(f["R2_subjects"]).index(s)]
					for name in ["R2_global", "voxel_R2_median", "R2_global_brain", "R2c_global", "R2c_global_brain", "pooled_r_dt_brain"]:
						if(name not in k):
							continue
						row.setdefault(name, []).append(float(v[k.index(name)]))
	return {s: {k: float(np.mean(v)) for k, v in row.items()} for s, row in values.items()}


def paired(a, b, key, alternative):
	common = sorted(set(a) & set(b))
	x = np.array([a[s][key] for s in common]); y = np.array([b[s][key] for s in common])
	ok = np.isfinite(x) & np.isfinite(y)
	if(ok.sum() < 5):
		return {"n": int(ok.sum()), "p": None}
	return {"n": int(ok.sum()), "model": float(np.mean(x[ok])), "reference": float(np.mean(y[ok])),
			"p": float(wilcoxon(x[ok], y[ok], alternative=alternative).pvalue)}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('root')
	parser.add_argument('-configs', required=True)
	parser.add_argument('-out', default=None)
	parser.add_argument('-suffix', default="", help='"_fluct" for the fluctuation-target baselines')
	parser.add_argument('-prefix', default="", help='"mni_" for the baselines on registered volumes')
	opt = parser.parse_args()

	template = per_subject(opt.root, opt.prefix+"eeg_free_template"+opt.suffix)
	oracle = per_subject(opt.root, opt.prefix+"oracle_own_mean"+opt.suffix)
	locked = per_subject(opt.root, opt.prefix+"stimulus_locked"+opt.suffix)
	#leave-one-subject-out mixture of the training subjects' voxelwise Normals (reevaluate_uq.py -mode template_mixture)
	mixture = per_subject(opt.root, "template_mixture"+opt.suffix)
	report = {"template_subjects": len(template)}
	for config in opt.configs.split(","):
		real = per_subject(opt.root, config)
		if(not real):
			continue
		shift, swap = per_subject(opt.root, config, "_null_shift"), per_subject(opt.root, config, "_null_swap")
		r = {"n_subjects": len(real),
			"vs_template": {k: paired(real, template, k, "less") for k in ["rmse_brain", "nll", "crps"]},
			"vs_oracle_own_mean": {k: paired(real, oracle, k, "less") for k in ["rmse_brain", "nll"]},
			#audit A17: the task-locked EEG-free prediction (identical block timing across subjects)
			"vs_stimulus_locked": {k: paired(real, locked, k, "greater" if k == "temporal_r" else "less") for k in ["rmse_brain", "nll", "temporal_r"]},
			"conf90_coverage": float(np.mean([v["conf90"] for v in real.values()])),
			"temporal_r_vs_shift": paired(real, shift, "temporal_r", "greater"),
			"temporal_r_vs_swap": paired(real, swap, "temporal_r", "greater"),
			"nll_vs_shift": paired(real, shift, "nll", "less"),
			#variance of the BOLD time courses explained (shared definition with paper S)
			"R2_global_vs_template": paired(real, template, "R2_global", "greater"),
			"R2_global_vs_shift": paired(real, shift, "R2_global", "greater"),
			"R2_global_brain_vs_shift": paired(real, shift, "R2_global_brain", "greater"),
			"R2c_global_brain_vs_shift": paired(real, shift, "R2c_global_brain", "greater"),
			#swap keeps shared run trends and task timing; report it next to the shift (audit post-hoc v2). U's R2/R2c are
			#not detrended, so a shift only permutes the centred predictions and real-shift is a covariance difference
			"R2c_global_brain_vs_swap": paired(real, swap, "R2c_global_brain", "greater"),
			"R2_global_vs_swap": paired(real, swap, "R2_global", "greater"),
			"pooled_r_dt_brain_vs_shift": paired(real, shift, "pooled_r_dt_brain", "greater"),
			"pooled_r_dt_brain_vs_swap": paired(real, swap, "pooled_r_dt_brain", "greater")}
		if(mixture):
			r["vs_template_mixture"] = {k: paired(real, mixture, k, "less") for k in ["rmse_brain", "nll", "crps"]}
		beats = lambda d: d["p"] is not None and d["p"] < 0.05
		r["gate_pass"] = bool(all(beats(r["vs_template"][k]) for k in ["nll", "crps"]) and beats(r["temporal_r_vs_shift"]))
		#beyond task timing: must also beat the swap null and the stimulus-locked baseline on r
		r["eeg_specific"] = bool(beats(r["temporal_r_vs_swap"]) and (not locked or beats(r["vs_stimulus_locked"]["temporal_r"])))
		report[config] = r
		vt = r["vs_template"]
		fmt = lambda d: "n/a" if d["p"] is None else "%.3f vs %.3f (p=%.3g)" % (d["model"], d["reference"], d["p"])
		print("==", config, "subjects", r["n_subjects"], "GATE", "PASS" if r["gate_pass"] else "FAIL")
		for k in ["rmse_brain", "nll", "crps"]:
			print("   %-10s model vs template %s" % (k, fmt(vt[k])))
		print("   temporal r  real vs shift %s | vs swap %s" % (fmt(r["temporal_r_vs_shift"]), fmt(r["temporal_r_vs_swap"])))
		print("   nll real vs shift %s | conformal voxel coverage@0.9 %.3f" % (fmt(r["nll_vs_shift"]), r["conf90_coverage"]))
		print("   R2_global vs template %s | vs shift %s | in brain vs shift %s" % (fmt(r["R2_global_vs_template"]), fmt(r["R2_global_vs_shift"]), fmt(r["R2_global_brain_vs_shift"])))
		print("   centred R2c_global in brain vs shift %s | vs swap %s" % (fmt(r["R2c_global_brain_vs_shift"]), fmt(r["R2c_global_brain_vs_swap"])))
		print("   detrended pooled r in brain vs shift %s | vs swap %s  (real > shift but ~ swap = run-locked, not EEG-specific)" % (fmt(r["pooled_r_dt_brain_vs_shift"]), fmt(r["pooled_r_dt_brain_vs_swap"])))
		if(mixture):
			print("   vs template mixture: nll %s | crps %s | rmse %s" % tuple(fmt(r["vs_template_mixture"][k]) for k in ["nll", "crps", "rmse_brain"]))
		if(locked):
			vl = r["vs_stimulus_locked"]
			print("   vs stimulus-locked baseline: r %s | nll %s | rmse %s" % (fmt(vl["temporal_r"]), fmt(vl["nll"]), fmt(vl["rmse_brain"])))
			print("   EEG-specific beyond task timing requires real > swap (swap keeps the shared block timing)")
	if(opt.out):
		with open(opt.out, "w") as f:
			json.dump(report, f, indent=1)


if __name__ == "__main__":
	main()
