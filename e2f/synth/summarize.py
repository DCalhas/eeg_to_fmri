"""Summarize RQ1 fold results: per model and arm, mean metrics over test subjects, paired Wilcoxon vs the
EEG-free template (subject level), wins, and the shift-null evidence (per-subject p and a group sign test of
real minus shift-null voxel r)."""
import os, sys, json, glob
import numpy as np
from scipy import stats

RUNS = os.path.expanduser("~/eeg_to_fmri/runs/synth_noddi")


def load(arm, tag=""):
	pat = os.path.join(RUNS, f"{arm}_loso_fold*_seed0{('_' + tag) if tag else ''}.json")
	files = sorted(f for f in glob.glob(pat) if (tag or not any(f.endswith(f"_{t}.json") for t in ("smoke", "smoke_s2v", "s2v"))))
	return [json.load(open(f)) for f in files]


def summarize(arm, tag=""):
	rs = load(arm, tag)
	if not rs:
		return None
	models = [m for m in rs[0]["results"] if m not in ("template", "own_mean")]
	out = {"n_folds": len(rs)}
	T = {k: np.array([r["results"]["template"][k] for r in rs]) for k in ("ssim_2d_e2fnet", "rmse")}
	out["template"] = {k: float(v.mean()) for k, v in T.items()}
	out["own_mean"] = {k: float(np.mean([r["results"]["own_mean"][k] for r in rs])) for k in ("ssim_2d_e2fnet", "rmse")}
	for m in models:
		rows = [r["results"][m] for r in rs if m in r["results"]]
		if len(rows) != len(rs):
			continue
		d = {}
		for k in ("ssim_2d_e2fnet", "rmse"):
			v = np.array([x[k] for x in rows])
			better = (v > T[k]) if k == "ssim_2d_e2fnet" else (v < T[k])
			d[k] = {"mean": float(v.mean()), "wins_vs_template": int(better.sum()),
					"wilcoxon_p": float(stats.wilcoxon(v, T[k]).pvalue) if np.any(v != T[k]) else 1.0}
		r = np.array([x["voxel_r_mean"] for x in rows]); rn = np.array([x["shift_null_mean"]["voxel_r_mean"] for x in rows])
		d["voxel_r"] = {"mean": float(r.mean()), "shift_null_mean": float(rn.mean()), "delta_mean": float((r - rn).mean()),
						"sign_test_p": float(stats.binomtest(int((r > rn).sum()), len(r)).pvalue),
						"wilcoxon_real_vs_null_p": float(stats.wilcoxon(r, rn).pvalue),
						"n_subjects_p_lt_05": int(sum(x["p_voxel_r"] < 0.05 for x in rows))}
		d["best_epoch_median"] = float(np.median([x.get("best_epoch", np.nan) for x in rows])) if "best_epoch" in rows[0] else None
		out[m] = d
	return out


if __name__ == "__main__":
	res = {}
	for arm, tag in [("aligned", ""), ("filestart", ""), ("ecg_only", ""), ("aligned", "s2v")]:
		s = summarize(arm, tag)
		if s:
			res[f"{arm}{'_' + tag if tag else ''}"] = s
	json.dump(res, open(os.path.join(RUNS, "summary.json"), "w"), indent=1)
	print(json.dumps(res, indent=1))
