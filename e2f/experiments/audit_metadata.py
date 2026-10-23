"""Leakage audit: does any recording metadata predict the diagnostic label?

For every cohort: header summary per recording (cohorts.header_summary) and, for each field,
an association test with the label (Fisher/chi2 for categorical, Mann-Whitney AUC for continuous).
A field with |AUC-0.5| large or p small is a candidate shortcut and must be removed or controlled.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from e2f.data import cohorts

OUT = os.path.expanduser("~/eeg_to_fmri/runs/audit_metadata")
CATEGORICAL = ["n_channels", "n_eeg", "ch_set", "eeg_set", "sfreq", "highpass", "lowpass"]
CONTINUOUS = ["duration", "n_annot", "year", "month", "hour", "file_mb"]


def audit(name):
	recs = cohorts.COHORTS[name]()
	rows = []
	for r in recs:
		try:
			h = cohorts.header_summary(r)
		except Exception as e:
			h = {"error": repr(e)[:200]}
		rows.append({"sub": r["sub"], "label": r["label"], **r["meta"], **h})
	df = pd.DataFrame(rows)
	df.to_csv(os.path.join(OUT, f"{name}_headers.csv"), index=False)
	tests = {}
	y = df["label"].values
	for c in CATEGORICAL:
		if c not in df or df[c].nunique() < 2:
			tests[c] = {"levels": int(df[c].nunique()) if c in df else 0}
			continue
		ct = pd.crosstab(df[c], df["label"])
		p = stats.fisher_exact(ct.values)[1] if ct.shape == (2, 2) else stats.chi2_contingency(ct.values)[1]
		tests[c] = {"levels": int(df[c].nunique()), "p": float(p), "crosstab": {str(k): v for k, v in ct.to_dict("index").items()}}
	for c in CONTINUOUS:
		if c not in df:
			continue
		x = pd.to_numeric(df[c], errors="coerce")
		ok = x.notna().values
		if ok.sum() < 4 or len(set(y[ok])) < 2 or x[ok].nunique() < 2:
			continue
		tests[c] = {"auc": float(roc_auc_score(y[ok], x[ok])), "p": float(stats.mannwhitneyu(x[ok][y[ok] == 1], x[ok][y[ok] == 0]).pvalue),
					"median_pt": float(x[ok][y[ok] == 1].median()), "median_hc": float(x[ok][y[ok] == 0].median())}
	return df, tests


if __name__ == "__main__":
	os.makedirs(OUT, exist_ok=True)
	names = sys.argv[1:] or list(cohorts.COHORTS)
	report = {}
	for n in names:
		df, t = audit(n)
		report[n] = {"n": len(df), "n_pt": int(df.label.sum()), "errors": int(df.get("error", pd.Series(dtype=str)).notna().sum()), "tests": t}
		flag = {k: v for k, v in t.items() if v.get("p", 1) < 0.05 or abs(v.get("auc", 0.5) - 0.5) > 0.25}
		print(n, len(df), "pt", int(df.label.sum()), "FLAGGED:", json.dumps(flag, default=str)[:1500], flush=True)
	json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)
