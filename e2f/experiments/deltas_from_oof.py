"""Recompute the added-value bootstraps of a revision_clf output from its saved out-of-fold probabilities (used when a
job was stopped after its views but before the delta step). CPU only."""
import os, sys, json
import numpy as np
from joblib import Parallel, delayed
from e2f.experiments.revision_clf import OUT, boot_delta

cohort, tag = sys.argv[1], sys.argv[2]
P = np.load(os.path.join(OUT, cohort, f"oof_probs{tag}.npz")); y = P["y"]
f = os.path.join(OUT, f"{cohort}{tag}.json"); d = json.load(open(f))
keys = [k for k in d["views"] if "+" in k and k in P.files and k.split("+")[0] in P.files]
res = Parallel(n_jobs=len(keys))(delayed(boot_delta)(P[k].mean(0), P[k.split("+")[0]].mean(0), y) for k in keys)
for k, r in zip(keys, res):
	d["views"][k]["delta"] = r
d["note"] = "deltas recomputed from oof_probs with revision_clf.boot_delta (job stopped after the views)"
json.dump(d, open(f, "w"), indent=1, default=float)
print("deltas", len(keys))
