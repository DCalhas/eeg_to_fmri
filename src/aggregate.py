"""
Collect eval.npz files of the cross-validated runs into subject-level tables.

Each subject is a test subject exactly once (in its fold). A subject's value is the mean over its volumes,
averaged over seeds; tables report mean +- std across subjects, the std across seeds of the dataset mean,
and paired Wilcoxon signed-rank tests across subjects (Holm-corrected) against a reference configuration.

python aggregate.py RUNS/05 -reference calhas_spectral -out table_05.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["rmse", "rmse_brain", "ssim", "nll", "crps", "spearman", "ause"]
LOWER_IS_BETTER = {"rmse": True, "rmse_brain": True, "ssim": False, "nll": True, "crps": True, "spearman": False, "ause": True}
BASELINE = {"base_rmse": "rmse", "base_rmse_brain": "rmse_brain", "base_ssim": "ssim"}


TAG = ""  #"_null_shift" / "_null_swap" to aggregate the evaluation-time nulls


def load(root):
	"""-> values[config][metric][seed][subject] and coverage curves"""
	values = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
	for path in glob.glob(os.path.join(root, "*", "fold*", "seed*", "eval"+TAG+".npz")):
		config, seed = path.split(os.sep)[-4], path.split(os.sep)[-2]
		with np.load(path) as f:
			subjects = f["subject"]
			for metric in METRICS+list(BASELINE)+["picp", "conf_global_cov", "conf_voxel_cov", "conf_global_width", "conf_voxel_width", "width"]:
				for s in np.unique(subjects):
					values[config][metric][seed][int(s)] = np.nanmean(f[metric][subjects == s], axis=0)
			for name in f.files:
				if(name.startswith("temporal_r_")):
					values[config]["temporal_r"][seed][int(name.split("_")[-1])] = float(np.nanmedian(f[name]))
	return values


def per_subject(values, config, metric):
	"""Seed-averaged value per subject (only subjects present in every seed)."""
	seeds = values[config][metric]
	subjects = sorted(set.intersection(*[set(v) for v in seeds.values()]))
	missing = sorted(set.union(*[set(v) for v in seeds.values()])-set(subjects))
	if(len(missing)):
		print("W:", config, metric, "subjects missing in some seed (dropped):", missing)
	return subjects, np.array([np.mean([seeds[seed][s] for seed in seeds], axis=0) for s in subjects]), \
		np.array([np.mean([seeds[seed][s] for s in subjects], axis=0) for seed in seeds])


def holm(pvalues):
	order = np.argsort(pvalues); adjusted = np.empty(len(pvalues)); running = 0.
	for rank, i in enumerate(order):
		running = max(running, (len(pvalues)-rank)*pvalues[i]); adjusted[i] = min(1., running)
	return adjusted


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('root')
	parser.add_argument('-reference', default=None)
	parser.add_argument('-out', default=None)
	parser.add_argument('-tag', default="", help='"_null_shift" or "_null_swap" for the evaluation-time nulls')
	opt = parser.parse_args()
	global TAG
	TAG = opt.tag

	values = load(opt.root)
	table = {}
	for config in sorted(values):
		row = {}
		for metric in METRICS+["temporal_r"]+list(BASELINE):
			if(metric not in values[config]):
				continue
			subjects, v, by_seed = per_subject(values, config, metric)
			if(np.all(np.isnan(v))):
				continue
			row[metric] = {"mean": float(np.nanmean(v)), "std_subjects": float(np.nanstd(v, ddof=1)),
						"std_seeds": float(np.nanstd(by_seed, ddof=1)) if len(by_seed) > 1 else None,
						"n_subjects": len(subjects), "n_seeds": len(by_seed)}
		for metric in ["picp", "conf_global_cov", "conf_voxel_cov", "width", "conf_global_width", "conf_voxel_width"]:
			if(metric in values[config]):
				_, v, _ = per_subject(values, config, metric)
				if(not np.all(np.isnan(v))):
					#coverage of a new subject is only loosely guaranteed (2 calibration subjects): report the spread over subjects
					row[metric] = {"mean": np.nanmean(v, 0).tolist(), "std_subjects": np.nanstd(v, 0, ddof=1).tolist(),
								"min_subject": np.nanmin(v, 0).tolist(), "max_subject": np.nanmax(v, 0).tolist()}
		table[config] = row

	#the EEG-free baselines are the eeg_free_template / oracle_own_mean configurations (baseline_eeg_free.py)

	if(opt.reference is not None):
		tests = []
		for config in values:
			if(config == opt.reference):
				continue
			for metric in METRICS+["temporal_r"]:
				if(metric not in values[config] or metric not in values[opt.reference]):
					continue
				s1, v1, _ = per_subject(values, opt.reference, metric)
				s2, v2, _ = per_subject(values, config, metric)
				common = sorted(set(s1) & set(s2))
				a = np.array([v1[s1.index(s)] for s in common]); b = np.array([v2[s2.index(s)] for s in common])
				if(np.any(np.isnan(a)) or np.any(np.isnan(b)) or len(common) < 5):
					continue
				tests.append([config, metric, float(wilcoxon(a, b).pvalue), float(np.mean(a-b))])
		if(len(tests)):
			for t, p in zip(tests, holm(np.array([t[2] for t in tests]))):
				table[t[0]].setdefault("vs_reference", {})[t[1]] = {"p": t[2], "p_holm": float(p), "mean_diff_ref_minus_config": t[3]}

	for config, row in table.items():
		print(config.ljust(28), "  ".join(m+"="+"{:.3f}±{:.3f}".format(row[m]["mean"], row[m]["std_subjects"]) for m in METRICS+["temporal_r"] if m in row and "std_subjects" in row[m]))
	if(opt.out is not None):
		with open(opt.out, "w") as f:
			json.dump(table, f, indent=1)


if __name__ == "__main__":
	main()
