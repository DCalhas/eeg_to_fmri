"""Table 1: per-cohort, per-group demographics and recording facts used in the analysis (CPU job)."""
import os, json
import numpy as np
import pandas as pd
from e2f.data import cohorts

OUT = os.path.expanduser("~/eeg_to_fmri/runs/table1.json")
HARM = os.path.expanduser("~/eeg_to_fmri/runs/harmonized_v2")

res = {}
for c, f in cohorts.COHORTS.items():
	recs = f()
	g = {}
	for lab in (1, 0):
		R = [r for r in recs if r["label"] == lab]
		age = np.array([float(r["meta"]["age"]) for r in R if "age" in r["meta"] and pd.notna(r["meta"]["age"])], float)
		sex = [str(r["meta"].get("sex", "")).lower()[:1] for r in R]
		dur = [float(np.load(os.path.join(HARM, c, f"{r['sub']}.npz"))["duration"]) for r in R]
		g["patients" if lab else "controls"] = {"n": len(R), "age_mean": float(age.mean()) if len(age) else None,
			"age_sd": float(age.std(ddof=1)) if len(age) > 1 else None, "n_age": int(len(age)),
			"male": sex.count("m"), "female": sex.count("f"), "analysed_s_min": float(min(dur)), "analysed_s_max": float(max(dur))}
	res[c] = g
pd_events = pd.read_csv(os.path.expanduser("~/eeg_to_fmri/data/ds003490/sub-001/ses-01/eeg/sub-001_ses-01_task-Rest_events.tsv"), sep="\t")
res["pd_events_columns"] = list(pd_events.columns)
res["pd_events_head"] = pd_events.head(12).astype(str).values.tolist()
med = pd.read_csv(os.path.expanduser("~/eeg_to_fmri/data/ds003944/phenotype/medication.tsv"), sep="\t")
res["fep1_medication_columns"] = list(med.columns)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
