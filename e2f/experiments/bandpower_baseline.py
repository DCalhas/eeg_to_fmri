"""Harmonize every clinical cohort to the NODDI montage (cached) and run the EEG band-power baseline.

Per subject: preprocess.harmonized (61 ch, 250 Hz, 0.5-45 Hz, avg ref), fixed crop per cohort, saved to
runs/harmonized/<cohort>/<sub>.npz. Features: Welch log relative band power (5 bands x 61 ch).
Models (nested LOSO, subject level): band power; covariates only (age, sex); band power + covariates.
Inference: bootstrap CI and a subject-level label-permutation null (full nested LOSO per permutation).
"""
import os, sys, json, argparse
import numpy as np
import mne
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from e2f.data import cohorts, preprocess as P
from e2f.eval import stats

OUT = os.path.expanduser("~/eeg_to_fmri/runs/harmonized")
LINE = {"fribourg": 50, "fep1": 60, "fep2": 60, "semmelweis": 50, "pd": 60}
CROP = {"fribourg": (0, 400), "fep1": (0, 300), "fep2": (0, 300), "semmelweis": (0, 118), "pd": (0, 400)}
CS = [1e-3, 1e-2, 1e-1, 1, 10]


def harmonize_one(cohort, rec):
	f = os.path.join(OUT, cohort, f"{rec['sub']}.npz")
	if os.path.exists(f):
		return f
	mne.set_log_level("ERROR")
	raw = cohorts.load_raw(rec)
	el = rec["path"].replace("_task-responder_run-1_eeg.vhdr", "_electrodes.tsv") if cohort == "fribourg" else None
	tmax = min(CROP[cohort][1], raw.n_times / raw.info["sfreq"] - 1 / raw.info["sfreq"])
	h = P.harmonized(raw, LINE[cohort], electrodes_tsv=el, crop=(CROP[cohort][0], tmax))
	np.savez(f, eeg=h.get_data().astype(np.float32), sfreq=h.info["sfreq"], ch=np.array(h.ch_names), label=rec["label"],
			 duration=h.n_times / h.info["sfreq"])
	return f


def covariates(recs):
	age = np.array([float(r["meta"].get("age", np.nan)) for r in recs])
	sex = np.array([1.0 if str(r["meta"].get("sex", "")).lower().startswith(("m",)) else 0.0 for r in recs])
	age = np.where(np.isnan(age), np.nanmean(age) if np.isfinite(age).any() else 0, age)
	return np.c_[age, sex]


def nested_loso(X, y, n_jobs=8):
	def fold(i):
		tr = np.arange(len(y)) != i
		inner = StratifiedKFold(5, shuffle=True, random_state=0)
		best, bestC = -1, CS[0]
		for C in CS:
			p = np.zeros(tr.sum())
			for a, b in inner.split(X[tr], y[tr]):
				p[b] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[tr][a], y[tr][a]).predict_proba(X[tr][b])[:, 1]
			auc = roc_auc_score(y[tr], p)
			if auc > best:
				best, bestC = auc, C
		return make_pipeline(StandardScaler(), LogisticRegression(C=bestC, max_iter=5000)).fit(X[tr], y[tr]).predict_proba(X[~tr])[0, 1]
	return np.array(Parallel(n_jobs=n_jobs)(delayed(fold)(i) for i in range(len(y))))


def evaluate(X, y, n_perm, n_jobs, seed=0):
	p = nested_loso(X, y, n_jobs)
	out = {**stats.classification_metrics(p, y), "auc_ci95": list(stats.bootstrap_ci(p, y, n_boot=5000))}
	if n_perm:
		rng = np.random.default_rng(seed)
		null = [roc_auc_score(yp, nested_loso(X, yp, n_jobs)) for yp in (rng.permutation(y) for _ in range(n_perm))]
		out["perm_p_greater"] = float((1 + sum(v >= out["auc"] for v in null)) / (1 + n_perm))
		out["perm_null_mean"] = float(np.mean(null))
	return out, p


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("cohorts", nargs="*", default=["fep1", "fep2", "semmelweis", "pd", "fribourg"])
	ap.add_argument("--n_jobs", type=int, default=16)
	ap.add_argument("--perm", type=int, default=200)
	a = ap.parse_args()
	report = {}
	for c in a.cohorts:
		os.makedirs(os.path.join(OUT, c), exist_ok=True)
		recs = cohorts.COHORTS[c]()
		files = Parallel(n_jobs=a.n_jobs)(delayed(harmonize_one)(c, r) for r in recs)
		bp = np.stack([P.log_bandpower(np.load(f)["eeg"]).ravel() for f in files])
		y = np.array([r["label"] for r in recs])
		cov = covariates(recs)
		res = {}
		for name, X in {"bandpower": bp, "covariates": cov, "bandpower+cov": np.c_[bp, cov]}.items():
			res[name], _ = evaluate(X, y, a.perm if name != "bandpower+cov" else 0, a.n_jobs)
			print(c, name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()}), flush=True)
		report[c] = res
		json.dump(report, open(os.path.join(OUT, "bandpower_report.json"), "w"), indent=1, default=float)
