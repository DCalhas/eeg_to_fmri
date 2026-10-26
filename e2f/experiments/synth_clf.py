"""RQ3/RQ4: does EEG-synthesized fMRI carry diagnostic information beyond the EEG it is computed from?

Per clinical subject: harmonized EEG (57 NODDI channels, 250 Hz; runs/harmonized_v2) -> pseudo-TR frames
(2.16 s, 20-frame history, same featurization and training-subject standardization as the NODDI synthesizer)
-> final synthesizer (runs/synth_noddi/final) -> synthesized volumes (V, 30, 64, 64) -> subject features:
temporal mean and temporal SD of the synthesized volumes.
Views compared (same CV, same permutation null):
  bandpower          EEG log relative band power (57 x 5)
  synth_<model>      PCA of [mean, sd] synthesized volume (PCA fit on training folds only)
  bandpower+synth    concatenation
  synth_untrained    same pipeline with a randomly initialised synthesizer (architecture-only control)
Step 1 (GPU job, --synthesize): write runs/synth_clf/<cohort>/<model>.npz with per-subject features.
Step 2 (CPU job, --classify): classification on CPU nodes.
"""
import os, sys, json, argparse
import numpy as np

OUT = os.path.expanduser("~/eeg_to_fmri/runs/synth_clf")
HARM = os.path.expanduser("~/eeg_to_fmri/runs/harmonized_v2")
FINAL = os.path.expanduser("~/eeg_to_fmri/runs/synth_noddi/final")
COHORTS = ["fep1", "fep2", "semmelweis", "pd", "fribourg"]


def synthesize(cohort, model_name, arm="aligned", untrained=False, seed=0):
	import torch
	from e2f.data import noddi, cohorts as C
	from e2f.synth.benchmark_noddi import build_model
	assert torch.cuda.is_available(), "GPU-only rule"
	st = np.load(os.path.join(FINAL, f"{arm}_featstats.npz"))
	ck = torch.load(os.path.join(FINAL, f"{arm}_{model_name}_seed{seed}.pt"), map_location="cpu", weights_only=False)
	if model_name == "ntvit":
		from e2f.synth import ntvit_final as NT
		model = NT.build(ck["hparams"])
	else:
		model = build_model(model_name, tuple(ck["x_shape"]))
	if not untrained:
		model.load_state_dict(ck["state_dict"])
	model = model.cuda().eval()
	feats, labels, subs = [], [], []
	for rec in C.COHORTS[cohort]():
		d = np.load(os.path.join(HARM, cohort, f"{rec['sub']}.npz"))
		eeg = d["eeg"]
		onsets = np.arange(20, int(eeg.shape[1] / 250 / noddi.TR)) * noddi.TR  # pseudo-TRs after a 20-frame history
		if model_name == "ntvit":  # raw waveforms, same 20-TR window ending at each pseudo-onset
			W, _ = NT.raw_windows(eeg, onsets)
			Y = torch.from_numpy(NT.predict(model, W)).cuda()
		else:
			X, idx, _ = noddi.tr_frames(eeg, onsets, n_frames=20)
			X = ((np.transpose(X, (0, 3, 1, 2)) - st["mu"]) / st["sd"]).astype(np.float32)
			with torch.no_grad():
				Y = torch.cat([model(torch.from_numpy(X[i:i + 64]).cuda()).clamp(0, 1) for i in range(0, len(X), 64)])
		feats.append(torch.cat([Y.mean(0).flatten(), Y.std(0).flatten()]).cpu().numpy().astype(np.float32))
		labels.append(rec["label"]); subs.append(rec["sub"])
	os.makedirs(os.path.join(OUT, cohort), exist_ok=True)
	tag = f"{model_name}{'_untrained' if untrained else ''}"
	np.savez(os.path.join(OUT, cohort, f"{tag}.npz"), X=np.stack(feats), y=np.array(labels), sub=np.array(subs))
	print("synthesized", cohort, tag, np.stack(feats).shape, flush=True)


def classify(cohort, models, n_perm, n_jobs):
	from sklearn.decomposition import PCA
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.metrics import roc_auc_score
	from joblib import Parallel, delayed
	from e2f.data import cohorts as C, preprocess as P
	recs = C.COHORTS[cohort]()
	y = np.array([r["label"] for r in recs])
	bp = np.stack([P.log_bandpower(np.load(os.path.join(HARM, cohort, f"{r['sub']}.npz"))["eeg"]).ravel() for r in recs])
	views = {"bandpower": (bp, None)}
	for m in models:
		f = os.path.join(OUT, cohort, f"{m}.npz")
		if os.path.exists(f):
			d = np.load(f); assert list(d["sub"]) == [r["sub"] for r in recs]
			views[f"synth_{m}"] = (d["X"], 20)
			views[f"bandpower+synth_{m}"] = (np.c_[bp, d["X"]], ("split", bp.shape[1], 20))
	CS = [1e-3, 1e-2, 1e-1, 1, 10]

	def pipe(C_, n_pca, n_feat):
		from sklearn.compose import ColumnTransformer
		if n_pca is None:
			return make_pipeline(StandardScaler(), LogisticRegression(C=C_, max_iter=5000))
		if isinstance(n_pca, tuple):  # bandpower columns kept, synth columns PCA-reduced (fit on training folds)
			_, k, nc = n_pca
			ct = ColumnTransformer([("bp", StandardScaler(), list(range(k))),
									("syn", make_pipeline(StandardScaler(), PCA(nc, random_state=0)), list(range(k, n_feat)))])
			return make_pipeline(ct, StandardScaler(), LogisticRegression(C=C_, max_iter=5000))
		return make_pipeline(StandardScaler(), PCA(n_pca, random_state=0), StandardScaler(), LogisticRegression(C=C_, max_iter=5000))

	def cv_auc(X, yy, n_pca, repeats=20, seed=0):
		splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed).split(X, yy))
		def fold(tr, te):
			inner = StratifiedKFold(5, shuffle=True, random_state=0)
			best, bc = -1, CS[0]
			for C_ in CS:
				p = np.zeros(len(tr))
				for a, b in inner.split(X[tr], yy[tr]):
					p[b] = pipe(C_, n_pca, X.shape[1]).fit(X[tr][a], yy[tr][a]).predict_proba(X[tr][b])[:, 1]
				s = roc_auc_score(yy[tr], p)
				if s > best: best, bc = s, C_
			return te, pipe(bc, n_pca, X.shape[1]).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
		res = Parallel(n_jobs=n_jobs)(delayed(fold)(tr, te) for tr, te in splits)
		Pm = np.zeros((repeats, len(yy)))
		for i, (te, p) in enumerate(res):
			Pm[i // 10, te] = p
		return float(np.mean([roc_auc_score(yy, p) for p in Pm])), Pm.mean(0)

	from e2f.eval import stats
	out, scores = {}, {}
	for name, (X, n_pca) in views.items():
		auc, p = cv_auc(X, y, n_pca)
		rng = np.random.default_rng(0)
		null = [cv_auc(X, rng.permutation(y), n_pca, repeats=2)[0] for _ in range(n_perm)] if n_perm else []
		out[name] = {"auc": auc, "auc_ci95": list(stats.bootstrap_ci(p, y, n_boot=2000)),
					 "perm_p_greater": float((1 + sum(v >= auc for v in null)) / (1 + len(null))) if null else None}
		scores[name] = p
		print(cohort, name, json.dumps(out[name]), flush=True)
	for name in scores:
		if name.startswith("bandpower+synth_"):
			d, ci = stats.paired_bootstrap_delta(scores[name], scores["bandpower"], y, n_boot=2000)
			out[name]["delta_vs_bandpower"] = {"delta": d, "ci95": list(ci)}
	tag = "-".join(sorted({m.replace("_untrained", "") for m in models}))  # never overwrite results of other model sets
	json.dump(out, open(os.path.join(OUT, f"{cohort}_clf_{tag}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--synthesize", action="store_true")
	ap.add_argument("--classify", action="store_true")
	ap.add_argument("--cohorts", nargs="+", default=COHORTS)
	ap.add_argument("--models", nargs="+", default=["cnn_tag", "e2fnet", "spec2vol"])
	ap.add_argument("--perm", type=int, default=100)
	ap.add_argument("--n_jobs", type=int, default=32)
	a = ap.parse_args()
	if a.synthesize:
		for c in a.cohorts:
			for m in a.models:
				synthesize(c, m)
				synthesize(c, m, untrained=True)
	if a.classify:
		ms = [m for m in a.models] + [f"{m}_untrained" for m in a.models]
		for c in a.cohorts:
			classify(c, ms, a.perm, a.n_jobs)
