"""Diagnostic for below-chance cross-validated AUCs (review point 8: CNN-TAG in FEP2, AUC 0.33). CPU only.

For a cohort and synthesized-feature file:
  1. PCA of the full sample (unsupervised): variance explained and the association of each of the first 20 components
     with the label and with recording metadata (sampling rate, duration, age, sex, CPZ);
  2. leverage: share of the first components carried by single subjects;
  3. per outer fold of the published CV (repeat 0): training AUC, test AUC, chosen C, and the difference between the
     training-fold class means projected on the fitted weight vector for training and test subjects (sign flips =
     the classifier learns a direction that reverses out of sample).
Writes runs/revision/<cohort>_<tag>_diag.json.
"""
import os, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

from e2f.experiments.revision_clf import OUT, SYN, Reducer, CS


def main(cohort, tag):
	from sklearn.metrics import roc_auc_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
	d = np.load(os.path.join(SYN, cohort, f"{tag}.npz")); X, y, subs = d["X"].astype(np.float32), d["y"], list(d["sub"])
	meta = pd.read_csv(os.path.join(OUT, cohort, "metadata.csv")).set_index("sub").loc[subs]
	cov = pd.read_csv(os.path.join(OUT, cohort, "covariates.csv")).set_index("sub").loc[subs]
	red = Reducer([(slice(0, X.shape[1]), 20)]).fit(X)
	Z = red._tr(X)  # unscaled component scores
	ev = (Z ** 2).sum(0); ev = ev / ((X - X.mean(0)) / (X.std(0) + 1e-12)).var(0).sum() / len(X)
	out = {"cohort": cohort, "tag": tag, "explained_variance_ratio": [float(v) for v in ev]}
	assoc = {}
	cols = {"label": y, **{c: cov[c].to_numpy(float) for c in cov.columns}}
	for c in ("sfreq", "duration", "lowpass"):
		if c in meta and meta[c].nunique() > 1:
			cols[c] = meta[c].to_numpy(float)
	for k in range(Z.shape[1]):
		assoc[f"pc{k + 1}"] = {c: float(stats.spearmanr(Z[:, k], v).correlation) for c, v in cols.items()}
		assoc[f"pc{k + 1}"]["label_auc"] = float(roc_auc_score(y, Z[:, k]))
	out["pc_associations"] = assoc
	lev = (Z[:, :5] ** 2) / (Z[:, :5] ** 2).sum(0)
	out["max_subject_share_pc1_5"] = [float(v) for v in lev.max(0)]
	out["max_share_subject"] = [subs[i] for i in lev.argmax(0)]
	folds = []
	for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=0).split(X, y)):
		inner = StratifiedKFold(5, shuffle=True, random_state=0)
		P = {C_: np.zeros(len(tr)) for C_ in CS}
		for a, b in inner.split(X[tr], y[tr]):
			r = Reducer([(slice(0, X.shape[1]), 20)]).fit(X[tr][a])
			for C_ in CS:
				P[C_][b] = LogisticRegression(C=C_, max_iter=5000).fit(r.transform(X[tr][a]), y[tr][a]).predict_proba(r.transform(X[tr][b]))[:, 1]
		best = max(CS, key=lambda C_: (roc_auc_score(y[tr], P[C_]), -CS.index(C_)))
		r = Reducer([(slice(0, X.shape[1]), 20)]).fit(X[tr])
		clf = LogisticRegression(C=best, max_iter=5000).fit(r.transform(X[tr]), y[tr])
		s_tr, s_te = clf.decision_function(r.transform(X[tr])), clf.decision_function(r.transform(X[te]))
		folds.append({"fold": i, "C": best, "inner_auc": float(roc_auc_score(y[tr], P[best])), "train_auc": float(roc_auc_score(y[tr], s_tr)),
					  "test_auc": float(roc_auc_score(y[te], s_te)) if len(set(y[te])) == 2 else None,
					  "test_mean_score_pt_minus_hc": float(s_te[y[te] == 1].mean() - s_te[y[te] == 0].mean())})
	out["folds_repeat0"] = folds
	out["n_folds_test_below_half"] = int(sum(f["test_auc"] is not None and f["test_auc"] < 0.5 for f in folds))
	json.dump(out, open(os.path.join(OUT, f"{cohort}_{tag}_diag.json"), "w"), indent=1)
	print(json.dumps({k: v for k, v in out.items() if k != "pc_associations"}, indent=1))
	top = sorted(((abs(v["label"]), k, v) for k, v in assoc.items()), reverse=True)[:5]
	for _, k, v in top:
		print(k, {c: round(x, 3) for c, x in v.items()})


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--cohort", default="fep2")
	ap.add_argument("--tags", nargs="+", default=["cnn_tag", "cnn_tag_untrained"])
	a = ap.parse_args()
	for t in a.tags:
		main(a.cohort, t)
