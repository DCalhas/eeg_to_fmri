"""RQ1b significance with the IAAFT-surrogate refit null (Mehta et al. 2504.10752, with the audit fixes D1):
hyperparameters re-tuned per surrogate; p = (1+count)/(1+N), N >= 100; detrended scoring; binomial group test.

Two designs (CPU job):
  cross    cross-subject LOSO ridge (as benchmark_bold): surrogates replace the training and validation BOLD
           (each training subject's parcels independently), the refit model is scored on the real test subject.
  within   per-subject, Mehta-style: first 60% of volumes train (alpha by blocked 5-fold CV inside), 15 TR gap,
           last 30% test; surrogates replace the training BOLD. NODDI has one session, so this is intra-session.
Features: bandpower (57 ch x 5 bands x 20 TR lags) or spectral (PCA 256), from benchmark_bold.features.
Statistic: mean over parcels of the detrended test r (and pooled detrended R^2).
"""
import os, json, argparse
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from e2f.data import noddi
from e2f.eval.surrogates import iaaft, perm_p, binomial_group
from e2f.synth import benchmark_bold as BB
from e2f.synth.benchmark_noddi import make_folds

RUNS = os.path.expanduser("~/eeg_to_fmri/runs/bold_surrogate")
ALPHAS = BB.ALPHAS


def fit_predict(Xtr, Ytr, Xva, Yva, Xte):
	best = None
	for a in ALPHAS:
		m = Ridge(alpha=a).fit(Xtr, Ytr)
		s = BB.r2_cols(m.predict(Xva), Yva)[1]
		if best is None or s > best[0]:
			best = (s, a, m)
	return best[2].predict(Xte), best[1]


def stat(P, Y):
	Pd, Yd = BB.detrend_cols(P), BB.detrend_cols(Y)
	return float(np.nanmean(BB.corr_cols(Pd, Yd))), BB.r2_cols(Pd, Yd)[1]


def prep(Xtr, Xs, kind, seed):
	sc = StandardScaler().fit(Xtr)
	Xtr, Xs = sc.transform(Xtr), [sc.transform(x) for x in Xs]
	if kind == "spectral":
		p = PCA(256, random_state=seed).fit(Xtr); Xtr, Xs = p.transform(Xtr), [p.transform(x) for x in Xs]
	return Xtr, Xs


def cross_fold(fold, kind, n_surr, seed):
	tr = [BB.features(s, kind)[:2] for s in fold["train"]]; va = [BB.features(s, kind)[:2] for s in fold["val"]]
	Xte, Yte = BB.features(fold["test"][0], kind)[:2]
	Xtr = np.concatenate([t[0] for t in tr]); Xva = np.concatenate([t[0] for t in va])
	Xtr, (Xva, Xte) = prep(Xtr, [Xva, Xte], kind, seed)
	Ytr = np.concatenate([t[1] for t in tr]); Yva = np.concatenate([t[1] for t in va])
	P, a = fit_predict(Xtr, Ytr, Xva, Yva, Xte)
	real_r, real_R2 = stat(P, Yte)
	def one(b):
		rng = np.random.default_rng(seed * 100003 + b)
		Ytr_s = np.concatenate([iaaft(t[1], rng=rng) for t in tr]); Yva_s = np.concatenate([iaaft(t[1], rng=rng) for t in va])
		Ps, _ = fit_predict(Xtr, Ytr_s, Xva, Yva_s, Xte)  # alpha re-tuned on surrogate validation targets
		return stat(Ps, Yte)
	null = [one(b) for b in range(n_surr)]
	return {"test": fold["test"][0], "alpha": a, "r": real_r, "R2": real_R2, "null_r": [n[0] for n in null], "null_R2": [n[1] for n in null],
			"p_r": perm_p(real_r, [n[0] for n in null]), "p_R2": perm_p(real_R2, [n[1] for n in null])}


def within_subject(sub, kind, n_surr, seed, train_frac=0.6, gap=15):
	X, Y = BB.features(sub, kind)[:2]
	T = len(X); n_tr = int(train_frac * T); te0 = n_tr + gap
	Xtr, Ytr, Xte, Yte = X[:n_tr], Y[:n_tr], X[te0:], Y[te0:]
	Xtr, (Xte,) = prep(Xtr, [Xte], kind, seed)
	blocks = np.array_split(np.arange(n_tr), 5)
	def tune_fit(Ytrain):
		best = None
		for a in ALPHAS:
			sc = []
			for b in blocks:  # blocked CV (contiguous folds) inside the training part
				trn = np.setdiff1d(np.arange(n_tr), b)
				sc.append(BB.r2_cols(Ridge(alpha=a).fit(Xtr[trn], Ytrain[trn]).predict(Xtr[b]), Ytrain[b])[1])
			s = float(np.mean(sc))
			if best is None or s > best[0]:
				best = (s, a)
		return Ridge(alpha=best[1]).fit(Xtr, Ytrain).predict(Xte), best[1]
	P, a = tune_fit(Ytr)
	real_r, real_R2 = stat(P, Yte)
	null = []
	for b in range(n_surr):
		Ps, _ = tune_fit(iaaft(Ytr, rng=np.random.default_rng(seed * 100003 + b)))
		null.append(stat(Ps, Yte))
	return {"test": sub, "alpha": a, "r": real_r, "R2": real_R2, "null_r": [n[0] for n in null], "null_R2": [n[1] for n in null],
			"p_r": perm_p(real_r, [n[0] for n in null]), "p_R2": perm_p(real_R2, [n[1] for n in null])}


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--design", default="within", choices=["within", "cross"])
	ap.add_argument("--kinds", nargs="+", default=["bandpower", "spectral"])
	ap.add_argument("--n_surr", type=int, default=100)
	ap.add_argument("--n_jobs", type=int, default=16)
	ap.add_argument("--seed", type=int, default=0)
	a = ap.parse_args()
	os.makedirs(RUNS, exist_ok=True)
	for kind in a.kinds:
		out = os.path.join(RUNS, f"{a.design}_{kind}.json")
		if os.path.exists(out):
			continue
		if a.design == "within":
			res = Parallel(n_jobs=a.n_jobs)(delayed(within_subject)(s, kind, a.n_surr, a.seed) for s in noddi.subjects())
		else:
			folds = make_folds()["fold_sets"]["loso"]["folds"]
			res = Parallel(n_jobs=a.n_jobs)(delayed(cross_fold)(f, kind, a.n_surr, a.seed) for f in folds)
		summ = {"mean_r": float(np.mean([r["r"] for r in res])), "mean_R2": float(np.mean([r["R2"] for r in res])),
				"group_r": binomial_group([r["p_r"] for r in res]), "group_R2": binomial_group([r["p_R2"] for r in res])}
		print(a.design, kind, json.dumps(summ), flush=True)
		json.dump({"design": a.design, "kind": kind, "summary": summ, "subjects": res}, open(out, "w"), indent=1)
