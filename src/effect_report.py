"""
Effect sizes and non-superiority tests for the negative claims (review #3), from the saved per-volume evaluations.

For every paired comparison over test subjects (seed-averaged, as gate_report.per_subject):
	* Hodges-Lehmann estimate of the paired difference (median of the Walsh averages) with a subject bootstrap 95% CI;
	* one-sided Wilcoxon p for the model being better (as gate_report);
	* a non-superiority test against a smallest effect size of interest (SESOI) fixed before looking at the
	  differences: H0 "the model is better by at least delta", rejected by a one-sided Wilcoxon on the shifted
	  differences. Rejecting it supports "no meaningful improvement".
SESOI: temporal r, 0.05 (absolute); NLL, CRPS, RMSE: 5% of the distance between the EEG-free population template and
the own-subject oracle on that metric (a model recovering less than 5% of what knowing the subject's own image buys).

Also: the deep ensemble against its own members (the heteroscedastic runs, seed-averaged = mean single member),
ECG-only against EEG input, and per-subject conformal coverage (for the strip plot).

python effect_report.py RUNS/cv/05 -out RUNS/effect_report.json
"""
import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

from gate_report import per_subject

parser = argparse.ArgumentParser()
parser.add_argument('root')
parser.add_argument('-out', required=True)
parser.add_argument('-n_boot', default=10000, type=int)
parser.add_argument('-configs', default=None, help="comma separated (default: the CHUR-XP2 grid)")
parser.add_argument('-prefix', default="", help='"mni_" for the baselines on registered volumes')
parser.add_argument('-members', default="zerofill_hetero", help="base configuration of the deep ensemble")
opt = parser.parse_args()

CONFIGS = ["zerofill", "zerofill_hetero", "zerofill_evidential", "zerofill_cqr", "zerofill_mcdropout", "zerofill_specdropout", "spectral",
		   "zerofill_flipout", "flipout", "zerofill_swag", "zerofill_laplace", "zerofill_hetero_ensemble", "spectral_voxelhead",
		   "zerofill_hetero_ecg_only", "zerofill_hetero_minprep"]
if(opt.configs):
	CONFIGS = opt.configs.split(",")
rng = np.random.RandomState(0)


def hodges_lehmann(d):
	i, j = np.triu_indices(len(d))
	return float(np.median((d[i]+d[j])/2))


def paired(a, b, key, better, delta):
	"""better: "lower" or "higher" for the model (a) being better than the reference (b)."""
	common = sorted(set(a) & set(b))
	x = np.array([a[s][key] for s in common]); y = np.array([b[s][key] for s in common])
	ok = np.isfinite(x) & np.isfinite(y); d = (x-y)[ok]
	if(len(d) < 5):
		return None
	boot = [hodges_lehmann(d[rng.randint(0, len(d), len(d))]) for _ in range(opt.n_boot)]
	sign = -1. if better == "lower" else 1.  #improvement = sign*d > 0
	return {"n": int(len(d)), "model": float(np.mean(x[ok])), "reference": float(np.mean(y[ok])),
			"hl": hodges_lehmann(d), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
			"p_better": float(wilcoxon(sign*d, alternative="greater").pvalue),
			#non-superiority: H0 improvement >= delta  <=>  sign*d - delta >= 0; reject if the shifted differences are < 0
			"delta": float(delta), "p_nonsuperiority": float(wilcoxon(sign*d-delta, alternative="less").pvalue),
			"per_subject": {str(s): float(v) for s, v in zip(np.array(common)[ok], d)}}


template = per_subject(opt.root, opt.prefix+"eeg_free_template")
oracle = per_subject(opt.root, opt.prefix+"oracle_own_mean")
locked = per_subject(opt.root, opt.prefix+"stimulus_locked")
mean = lambda d, k: float(np.nanmean([v[k] for v in d.values()]))
DELTA = {k: 0.05*(mean(template, k)-mean(oracle, k)) for k in ["nll", "crps", "rmse_brain"]}
DELTA["temporal_r"] = 0.05
report = {"sesoi": DELTA, "configs": {}}
for c in CONFIGS:
	real = per_subject(opt.root, c)
	if(not real):
		continue
	shift, swap = per_subject(opt.root, c, "_null_shift"), per_subject(opt.root, c, "_null_swap")
	r = {"vs_template": {k: paired(real, template, k, "lower", DELTA[k]) for k in ["nll", "crps", "rmse_brain"]},
		 "vs_stimulus_locked_r": paired(real, locked, "temporal_r", "higher", DELTA["temporal_r"]),
		 "r_vs_shift": paired(real, shift, "temporal_r", "higher", DELTA["temporal_r"]),
		 "r_vs_swap": paired(real, swap, "temporal_r", "higher", DELTA["temporal_r"]),
		 "nll_vs_shift": paired(real, shift, "nll", "lower", DELTA["nll"]),
		 "nll_vs_swap": paired(real, swap, "nll", "lower", DELTA["nll"]),
		 "coverage90_per_subject": {str(s): v["conf90"] for s, v in real.items()},
		 "nll_per_subject": {str(s): v.get("nll") for s, v in real.items()}}
	report["configs"][c] = r
	brief = lambda v: None if v is None else [round(v["hl"], 4), [round(x, 4) for x in v["ci95"]], round(v["p_better"], 4), round(v["p_nonsuperiority"], 4)]
	print("==", c, json.dumps({k: brief(v) for k, v in r.items() if k in ("vs_stimulus_locked_r", "r_vs_shift", "r_vs_swap", "nll_vs_shift", "nll_vs_swap")}),
		  json.dumps({k: brief(v) for k, v in r["vs_template"].items()}), flush=True)

members = per_subject(opt.root, opt.members)
ens = per_subject(opt.root, opt.members+"_ensemble")
report["ensemble_vs_members"] = {k: paired(ens, members, k, "lower", DELTA[k]) for k in ["nll", "crps", "rmse_brain"]}
ecg, eeg = per_subject(opt.root, "zerofill_hetero_ecg_only"), members
report["eeg_vs_ecg"] = {"nll": paired(eeg, ecg, "nll", "lower", DELTA["nll"]), "temporal_r": paired(eeg, ecg, "temporal_r", "higher", DELTA["temporal_r"])} if ecg else None  #no ECG-only arm on CHUR-XP1
report["coverage90_template_per_subject"] = {str(s): v["conf90"] for s, v in template.items()}
report["template_per_subject"] = {str(s): {k: v.get(k) for k in ["nll", "crps", "rmse_brain"]} for s, v in template.items()}
print("ensemble vs members", json.dumps({k: [v["hl"], v["ci95"], v["p_better"]] for k, v in report["ensemble_vs_members"].items()}))
if(report["eeg_vs_ecg"]):
	print("EEG vs ECG", json.dumps({k: [v["hl"], v["ci95"], v["p_better"], v["p_nonsuperiority"]] for k, v in report["eeg_vs_ecg"].items()}))
with open(opt.out, "w") as f:
	json.dump(report, f, indent=1)
