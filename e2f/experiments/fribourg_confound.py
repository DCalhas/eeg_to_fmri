"""Does the 2023 STFT result depend on the group-dependent EXG channels?

Features: the 2023 STFT pipeline (fribourg.legacy_instances) with three channel selections:
  legacy      first 132 channels in file order (as in 2023; positions 130-131 are EXG7/8 in HC, EXG3/4 in patients)
  eeg         the 128 EEG channels only (z-scored across those 128)
  exg_legacy  only positions 128-131 of the legacy selection
Classifier: L2 logistic regression, C tuned by inner subject-grouped 5-fold CV; outer LOSO.
Reported: window-pooled AUC (2023 protocol) and subject-level AUC with bootstrap CI.
"""
import os, sys, json, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from e2f.data import fribourg
from e2f.eval import stats

CACHE = os.path.expanduser("~/eeg_to_fmri/runs/fribourg_confound")
CS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def features(mode):
	f = os.path.join(CACHE, f"X_{mode}.npz")
	if os.path.exists(f):
		d = np.load(f); return d["X"], d["y"], d["g"]
	subs, labels = fribourg.subjects()
	X, y, g = [], [], []
	for i, s in enumerate(subs):
		Xi = fribourg.legacy_instances(s, channel_mode=mode)
		X.append(Xi); y += [labels[i]] * len(Xi); g += [i] * len(Xi)
		print(mode, s, Xi.shape, flush=True)
	X, y, g = np.concatenate(X), np.array(y), np.array(g)
	np.savez(f, X=X, y=y, g=g)
	return X, y, g


def model(C):
	return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def fold(X, y, g, s):
	tr, te = g != s, g == s
	inner = GroupKFold(5)
	scores = []
	for C in CS:
		p = np.zeros(tr.sum())
		Xt, yt, gt = X[tr], y[tr], g[tr]
		for a, b in inner.split(Xt, yt, gt):
			p[b] = model(C).fit(Xt[a], yt[a]).predict_proba(Xt[b])[:, 1]
		ps, ys, _ = stats.aggregate(p, yt, gt)
		scores.append(roc_auc_score(ys, ps))
	C = CS[int(np.argmax(scores))]
	return s, C, model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


def run(X, y, g, n_jobs):
	res = Parallel(n_jobs=n_jobs)(delayed(fold)(X, y, g, s) for s in np.unique(g))
	p = np.zeros(len(y))
	for s, C, ps in res:
		p[g == s] = ps
	sp, sy, _ = stats.aggregate(p, y, g)
	out = {"auc_window_pooled": roc_auc_score(y, p), **{f"subject_{k}": v for k, v in stats.classification_metrics(sp, sy).items()},
		   "subject_auc_ci95": list(stats.bootstrap_ci(sp, sy, n_boot=5000)), "C_chosen": [r[1] for r in res]}
	return out, sp, sy


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--n_jobs", type=int, default=8)
	ap.add_argument("--time_average", action="store_true", help="average the 10 STFT frames (band-power-like features)")
	a = ap.parse_args()
	os.makedirs(CACHE, exist_ok=True)
	results, subj_scores = {}, {}
	Xl, y, g = features("legacy")
	Xe, ye, ge = features("eeg")
	assert (y == ye).all() and (g == ge).all()
	sets = {"legacy_132": Xl, "eeg_128": Xe, "exg_only_legacy_pos128_131": Xl[:, 128:132]}
	for name, X in sets.items():
		if a.time_average:
			X = X.mean(-1)
		out, sp, sy = run(X.reshape(len(X), -1), y, g, a.n_jobs)
		results[name] = out; subj_scores[name] = sp.tolist()
		print(name, json.dumps({k: v for k, v in out.items() if k != "C_chosen"}), flush=True)
	names = list(sets)
	for i in range(len(names)):
		for j in range(i + 1, len(names)):
			a1, a2, z, p = stats.delong(np.array(subj_scores[names[i]]), np.array(subj_scores[names[j]]), sy)
			results[f"delong_{names[i]}_vs_{names[j]}"] = {"auc1": a1, "auc2": a2, "z": z, "p": p}
	tag = "timeavg" if a.time_average else "full"
	json.dump({"results": results, "subject_scores": subj_scores, "y": sy.tolist()}, open(os.path.join(CACHE, f"results_{tag}.json"), "w"), indent=1, default=float)
	print(json.dumps(results, indent=1, default=float))
