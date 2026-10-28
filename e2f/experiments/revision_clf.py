"""Revision analyses for the Brain Informatics manuscript (mock review 2026-10-28). CPU only (Slurm -p amd).

Same subjects, splits, classifier and nested CV as synth_clf (RepeatedStratifiedKFold 10 x 20, seed 0; inner
StratifiedKFold 5; l2 logistic regression, C in {1e-3..10}); the dimensionality reduction of a training fold is
fitted once and reused for every C (identical results, ~4x faster). Out-of-fold probabilities are saved for every view.

Views (per cohort):
  bp_rel            log relative band power (57 x 5), the published EEG baseline
  bp_abs            log absolute band power (57 x 5)                                    [review 1]
  bp_rel_abs        both
  bp_nogamma        relative power of 1-30 Hz only (excludes the EMG-sensitive band)     [review 6]
  stft_ms           temporal mean and SD over pseudo-TRs of the synthesizers' own input spectra
                    (57 channels x 97 bins of 2.16 s segments), PCA 20: the information-matched EEG view [review 1]
  stft_rp           sparse random projection (8,192) of each pseudo-TR's full input (57 x 97 x 20), then temporal
                    mean and SD, PCA 20: an untrained linear projection of the same input                 [review 1]
  synth_<m>[_untrained][_k<k>]   synthesized fMRI [mean, SD] volume, PCA k (5, 10, 20, 40)    [minor: PCA]
  synth_<m>_block   [mean, SD] volumes average-pooled in 3x4x4 voxel blocks (2 x 10 x 16 x 16), PCA 20
  <a>+<b>           concatenations; each high-dimensional block reduced by its own PCA 20
  <view>_cov        covariates regressed out of every feature within training folds (age, sex; CPZ for
                    psychosis; recording hour for Semmelweis and Fribourg; recording year for Fribourg) [review 9]
  <view>_clean      pseudo-TRs with peak-to-peak > 150 uV or 30-45 Hz power > median + 3 MAD removed      [review 6]
Inference: AUC over 20 repeats, 95% bootstrap CI over subjects (repeat-averaged OOF probabilities), label
permutation p from N permutations each re-running the nested CV with 2 repeats (as in synth_clf; 1,000 for the EEG
views, 500 for the synthesized views), PCA of high-dimensional blocks computed exactly through the subject Gram matrix, and for
added-value comparisons a paired stratified bootstrap (20,000) of AUC differences: percentile CI at 95% and at the
Bonferroni levels for 20 and 40 comparisons, a two-sided bootstrap p, Holm-adjusted p across the family.
"""
import os, json, argparse, time
import numpy as np
import pandas as pd

OUT = os.path.expanduser("~/eeg_to_fmri/runs/revision")
HARM = os.path.expanduser("~/eeg_to_fmri/runs/harmonized_v2")
SYN = os.path.expanduser("~/eeg_to_fmri/runs/synth_clf")
DATA = os.path.expanduser("~/eeg_to_fmri/data")
CROP = {"fribourg": (0, 400), "fep1": (0, 300), "fep2": (0, 300), "semmelweis": (0, 118), "pd": (0, 400)}
MODELS = ["cnn_tag", "e2fnet", "spec2vol", "ntvit"]
CS = [1e-3, 1e-2, 1e-1, 1, 10]
P2P_MAX = 150e-6


# ------------------------------------------------------------------ features
def pseudo_tr_frames(eeg):
	from e2f.data import noddi
	onsets = np.arange(20, int(eeg.shape[1] / 250 / noddi.TR)) * noddi.TR
	X, idx, _ = noddi.tr_frames(eeg, onsets, n_frames=20)
	return X, onsets[idx]


def clean_mask(eeg, onsets):
	"""Pseudo-TRs whose newest 2.16 s segment is free of large-amplitude and EMG-like artefacts."""
	n = int(round(2.16 * 250))
	ends = np.round((onsets + 2.16) * 250).astype(int)
	seg = np.stack([eeg[:, e - n:e] for e in ends])  # (V, C, n)
	p2p = (seg.max(-1) - seg.min(-1)).max(-1)
	f = np.fft.rfftfreq(n, 1 / 250)
	hf = np.log((np.abs(np.fft.rfft(seg, axis=-1)) ** 2)[..., (f >= 30) & (f <= 45)].mean((-1, -2)))
	med, mad = np.median(hf), np.median(np.abs(hf - np.median(hf))) + 1e-12
	return (p2p <= P2P_MAX) & (hf <= med + 3 * 1.4826 * mad), p2p, hf


def eeg_features(cohort, recs):
	from e2f.data import preprocess as P
	from sklearn.random_projection import SparseRandomProjection
	rp = SparseRandomProjection(n_components=8192, random_state=0).fit(np.zeros((2, 57 * 97 * 20), np.float32))
	F = {k: [] for k in ["bp_rel", "bp_abs", "bp_nogamma", "stft_ms", "stft_rp", "bp_rel_clean", "stft_ms_clean"]}
	art = []
	for r in recs:
		eeg = np.load(os.path.join(HARM, cohort, f"{r['sub']}.npz"))["eeg"]
		F["bp_rel"].append(P.log_bandpower(eeg).ravel())
		F["bp_abs"].append(P.log_bandpower(eeg, relative=False).ravel())
		F["bp_nogamma"].append(P.log_bandpower(eeg, bands=((1, 4), (4, 8), (8, 13), (13, 30))).ravel())
		X, on = pseudo_tr_frames(eeg)  # (V, 57, 97, 20)
		S = X[..., -1]
		F["stft_ms"].append(np.r_[S.mean(0).ravel(), S.std(0).ravel()])
		Z = rp.transform(X.reshape(len(X), -1))
		F["stft_rp"].append(np.r_[Z.mean(0), Z.std(0)])
		ok, p2p, hf = clean_mask(eeg, on)
		n = int(round(2.16 * 250)); ends = np.round((on + 2.16) * 250).astype(int)
		kept = np.concatenate([eeg[:, e - n:e] for e in ends[ok]], 1) if ok.sum() >= 5 else eeg
		F["bp_rel_clean"].append(P.log_bandpower(kept).ravel())
		Sk = S[ok] if ok.sum() >= 5 else S
		F["stft_ms_clean"].append(np.r_[Sk.mean(0).ravel(), Sk.std(0).ravel()])
		art.append({"sub": r["sub"], "frac_rejected": float(1 - ok.mean()), "median_p2p_uV": float(np.median(p2p) * 1e6),
					"hf_logpow_30_45": float(np.median(hf)), "n_tr": int(len(ok))})
	return {k: np.stack(v).astype(np.float32) for k, v in F.items()}, pd.DataFrame(art)


def raw_artefact_channels(cohort, recs):
	"""RMS of EOG (1-15 Hz) and accelerometer (2-12 Hz) channels over the analysed crop, from the raw files.
	Only channels present in every recording of the cohort are used (Fribourg EXG layout differs by group)."""
	import mne
	from e2f.data import cohorts as C
	mne.set_log_level("ERROR")
	rows, chans = [], None
	raws = []
	for r in recs:
		raw = C.load_raw(r, preload=False)
		names = {"eog": [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eog"],
				 "acc": [c for c in raw.ch_names if c in ("X", "Y", "Z")]}
		chans = names if chans is None else {k: [c for c in chans[k] if c in names[k]] for k in chans}
		raws.append(raw)
	for r, raw in zip(recs, raws):
		row = {"sub": r["sub"]}
		t1 = min(CROP[cohort][1], raw.times[-1])
		for kind, (lo, hi) in [("eog", (1.0, 15.0)), ("acc", (2.0, 12.0))]:
			if chans[kind]:
				x = raw.copy().pick(chans[kind]).crop(CROP[cohort][0], t1).load_data().filter(lo, hi, picks="all").get_data()
				row[f"{kind}_log_rms"] = float(np.log(np.sqrt((x ** 2).mean()) + 1e-12))
		rows.append(row)
	return pd.DataFrame(rows), chans


def pd_eye_order(recs):
	"""Order and timing of the eyes-open/closed codes inside the PD crop (events.tsv of the OFF session)."""
	out = []
	for r in recs:
		ev = r["path"].replace("_eeg.set", "_events.tsv")
		if not os.path.exists(ev):
			continue
		e = pd.read_csv(ev, sep="\t")
		col = [c for c in e.columns if c not in ("onset", "duration", "sample")]
		e = e[e.onset <= CROP["pd"][1]]
		codes = e[col[0]].astype(str).tolist() if col else []
		seq = [c for i, c in enumerate(codes) if i == 0 or c != codes[i - 1]]
		out.append({"sub": r["sub"], "label": r["label"], "col": col[0] if col else None, "first_codes": seq[:12],
					"n_events": len(e), "first_onset": float(e.onset.min()) if len(e) else None})
	return out


def covariates(cohort, recs, hdr):
	cov = pd.DataFrame({"sub": [r["sub"] for r in recs], "age": [float(r["meta"].get("age", np.nan)) for r in recs],
						"male": [float(str(r["meta"].get("sex", "")).lower() in ("m", "male")) for r in recs]})
	if cohort in ("fep1", "fep2"):
		ds = {"fep1": "ds003944", "fep2": "ds003947"}[cohort]
		med = pd.read_csv(os.path.join(DATA, ds, "phenotype", "medication.tsv"), sep="\t")
		m = dict(zip(med.participant_id, pd.to_numeric(med.CPZ_at_scan, errors="coerce")))
		cov["cpz"] = [m.get(r["sub"], np.nan) if r["label"] == 1 else 0.0 for r in recs]
		cov["cpz"] = cov["cpz"].fillna(cov["cpz"][[r["label"] == 1 for r in recs]].median())  # missing patient CPZ -> patient median
	if cohort in ("semmelweis", "fribourg"):
		h = dict(zip(hdr["sub"], hdr["hour"]))
		cov["hour"] = [h.get(r["sub"], np.nan) for r in recs]
	if cohort == "fribourg":
		yv = dict(zip(hdr["sub"], hdr["year"]))
		cov["year"] = [yv.get(r["sub"], np.nan) for r in recs]
	cov = cov.drop(columns=[c for c in cov.columns if c != "sub" and cov[c].isna().all()])
	for c in cov.columns:
		if c != "sub":
			cov[c] = cov[c].fillna(cov[c].median())
	return cov


def metadata_tests(cohort, recs):
	from e2f.data import cohorts as C
	from scipy import stats
	from sklearn.metrics import roc_auc_score
	rows = []
	for r in recs:
		try:
			h = C.header_summary(r)
		except Exception as e:
			h = {"error": repr(e)[:100]}
		rows.append({"sub": r["sub"], "label": r["label"], **{k: v for k, v in r["meta"].items()}, **h})
	df = pd.DataFrame(rows)
	tests = {}
	for k in df.columns:
		if k in ("sub", "label", "error", "ch_set", "eeg_set", "race", "session", "med", "bno"):
			continue
		x = df[k]
		if x.notna().sum() < 6 or x.nunique() < 2:
			continue
		if pd.api.types.is_numeric_dtype(x) and x.nunique() > 3:
			m = x.notna()
			tests[k] = {"auc": float(roc_auc_score(df.label[m], x[m])),
						"p": float(stats.mannwhitneyu(x[m][df.label[m] == 1], x[m][df.label[m] == 0]).pvalue),
						"mean_pt": float(x[m][df.label[m] == 1].mean()), "mean_hc": float(x[m][df.label[m] == 0].mean())}
		else:
			t = pd.crosstab(x.astype(str), df.label)
			p = stats.fisher_exact(t.values)[1] if t.shape == (2, 2) else stats.chi2_contingency(t.values)[1]
			tests[k] = {"table": {str(i): t.loc[i].tolist() for i in t.index}, "p": float(p)}
	return df, tests


# ------------------------------------------------------------------ classification
class Residualize:
	"""Regress covariates (training fold) out of every feature; the covariates ride along as the last q columns."""
	def __init__(self, q):
		self.q = q
	def fit(self, X):
		C = np.c_[np.ones(len(X)), X[:, -self.q:]]
		self.mc, self.sc = X[:, -self.q:].mean(0), X[:, -self.q:].std(0) + 1e-9
		Cz = np.c_[np.ones(len(X)), (X[:, -self.q:] - self.mc) / self.sc]
		self.B = np.linalg.lstsq(Cz, X[:, :-self.q], rcond=None)[0]
		return self
	def transform(self, X):
		Cz = np.c_[np.ones(len(X)), (X[:, -self.q:] - self.mc) / self.sc]
		return X[:, :-self.q] - Cz @ self.B


class GramPCA:
	"""Exact PCA for n << p through the n x n Gram matrix of the (already standardised, zero-mean) training data:
	identical subspace to full-SVD PCA, ~10x faster than sklearn's randomized solver for 245,760 features.
	Component signs are arbitrary; the downstream standardisation + logistic regression is sign-invariant."""
	def __init__(self, k):
		self.k = k
	def fit(self, Z):
		Z = np.asarray(Z, dtype=np.float64)
		self.mean = Z.mean(0)
		Zc = Z - self.mean
		w, U = np.linalg.eigh(Zc @ Zc.T)
		order = np.argsort(w)[::-1][:self.k]
		w, U = np.clip(w[order], 1e-12, None), U[:, order]
		self.V = (Zc.T @ U) / np.sqrt(w)  # (p, k) orthonormal principal axes
		return self
	def transform(self, Z):
		return (np.asarray(Z, dtype=np.float64) - self.mean) @ self.V


class Reducer:
	"""blocks: list of (column slice, n_pca or None). Each block is standardised and, if n_pca, PCA-reduced; a final
	standardisation follows whenever a PCA is present. This reproduces synth_clf's pipelines exactly:
	SS -> LR (band power), SS -> PCA -> SS -> LR (one block), ColumnTransformer(SS | SS -> PCA) -> SS -> LR (fusion)."""
	def __init__(self, blocks, q=0):
		self.blocks, self.q = blocks, q
	def fit(self, X):
		from sklearn.preprocessing import StandardScaler
		from sklearn.decomposition import PCA
		self.res = Residualize(self.q).fit(X) if self.q else None
		Xr = self.res.transform(X) if self.q else X
		self.parts = []
		for sl, k in self.blocks:
			s1 = StandardScaler().fit(Xr[:, sl])
			Z = s1.transform(Xr[:, sl])
			pca = (GramPCA(k) if Z.shape[1] > 4 * Z.shape[0] else PCA(k, random_state=0)).fit(Z) if k else None
			self.parts.append((sl, s1, pca))
		self.final = StandardScaler().fit(self._tr(Xr)) if any(k for _, k in self.blocks) else None
		return self
	def _tr(self, Xr):
		zs = []
		for sl, s1, pca in self.parts:
			Z = s1.transform(Xr[:, sl])
			zs.append(Z if pca is None else pca.transform(Z))
		return np.concatenate(zs, 1)
	def transform(self, X):
		Xr = self.res.transform(X) if self.q else X
		Z = self._tr(Xr)
		return Z if self.final is None else self.final.transform(Z)


def _fold(X, y, tr, te, blocks, q):
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import StratifiedKFold
	from sklearn.metrics import roc_auc_score
	inner = StratifiedKFold(5, shuffle=True, random_state=0)
	P = {C_: np.zeros(len(tr)) for C_ in CS}
	for a, b in inner.split(X[tr], y[tr]):
		red = Reducer(blocks, q).fit(X[tr][a]); Za, Zb = red.transform(X[tr][a]), red.transform(X[tr][b])
		for C_ in CS:
			P[C_][b] = LogisticRegression(C=C_, max_iter=5000).fit(Za, y[tr][a]).predict_proba(Zb)[:, 1]
	best = max(CS, key=lambda C_: (roc_auc_score(y[tr], P[C_]), -CS.index(C_)))  # first C reaching the max, as synth_clf
	red = Reducer(blocks, q).fit(X[tr])
	return LogisticRegression(C=best, max_iter=5000).fit(red.transform(X[tr]), y[tr]).predict_proba(red.transform(X[te]))[:, 1]


def cv(X, y, blocks, q=0, repeats=20, seed=0, n_jobs=1):
	from sklearn.model_selection import RepeatedStratifiedKFold
	from sklearn.metrics import roc_auc_score
	from joblib import Parallel, delayed
	splits = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed).split(X, y))
	if n_jobs == 1:
		res = [_fold(X, y, tr, te, blocks, q) for tr, te in splits]
	else:
		res = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, tr, te, blocks, q) for tr, te in splits)
	Pm = np.zeros((repeats, len(y)))
	for i, ((tr, te), p) in enumerate(zip(splits, res)):
		Pm[i // 10, te] = p
	return float(np.mean([roc_auc_score(y, p) for p in Pm])), Pm


def _perm_one(X, y, blocks, q, b):
	rng = np.random.default_rng(10_000 + b)
	return cv(X, rng.permutation(y), blocks, q, repeats=2)[0]


def boot_delta(p1, p2, y, n_boot=20000, seed=0):
	"""Paired stratified bootstrap of AUC(p1) - AUC(p2) over subjects."""
	from sklearn.metrics import roc_auc_score
	rng = np.random.default_rng(seed)
	pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
	d = np.empty(n_boot)
	for b in range(n_boot):
		i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
		d[b] = roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i])
	obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
	q = lambda a: [float(np.quantile(d, a / 2)), float(np.quantile(d, 1 - a / 2))]
	p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
	return {"delta": float(obs), "ci95": q(0.05), "ci_bonf20": q(0.05 / 20), "ci_bonf40": q(0.05 / 40), "p_boot": p,
			"se_boot": float(d.std())}


# ------------------------------------------------------------------ main
def run(cohort, n_perm, n_jobs, models, only=None, synth_tags=None, out_tag="", eeg_views=True, n_perm_synth=None):
	from e2f.data import cohorts as C
	from e2f.eval import stats
	from sklearn.metrics import roc_auc_score
	from joblib import Parallel, delayed
	t0 = time.time()
	os.makedirs(os.path.join(OUT, cohort), exist_ok=True)
	recs = C.COHORTS[cohort]()
	y = np.array([r["label"] for r in recs]); subs = [r["sub"] for r in recs]
	meta_df, meta_tests = metadata_tests(cohort, recs)
	meta_df.to_csv(os.path.join(OUT, cohort, "metadata.csv"), index=False)
	cov = covariates(cohort, recs, meta_df)
	cov.to_csv(os.path.join(OUT, cohort, "covariates.csv"), index=False)
	E, art = eeg_features(cohort, recs)
	rawart, chans = raw_artefact_channels(cohort, recs)
	art = art.merge(rawart, on="sub")
	art["label"] = y
	art.to_csv(os.path.join(OUT, cohort, "artefacts.csv"), index=False)
	from scipy import stats as ss
	art_tests = {}
	for k in [c for c in art.columns if c not in ("sub", "label", "n_tr")]:
		x = art[k].astype(float)
		art_tests[k] = {"auc": float(roc_auc_score(y, x)), "p": float(ss.mannwhitneyu(x[y == 1], x[y == 0]).pvalue),
						"mean_pt": float(x[y == 1].mean()), "mean_hc": float(x[y == 0].mean())}
	extra = {"metadata_tests": meta_tests, "artefact_tests": art_tests, "artefact_channels": chans,
			 "covariates": [c for c in cov.columns if c != "sub"]}
	if cohort == "pd":
		extra["pd_eye_order"] = pd_eye_order(recs)
	Q = cov.drop(columns="sub").to_numpy(float)
	q = Q.shape[1]

	S = {}
	tags = synth_tags if synth_tags else [t for m in models for t in (m, f"{m}_untrained")]
	for m in [None]:
		for tag in tags:
			f = os.path.join(SYN, cohort, f"{tag}.npz")
			if os.path.exists(f):
				d = np.load(f); assert list(d["sub"]) == subs, tag
				S[tag] = d["X"].astype(np.float32)
	nb = lambda X: [(slice(0, X.shape[1]), None)]
	pc = lambda X, k=20: [(slice(0, X.shape[1]), k)]
	def two(A, B, ka, kb):
		return np.c_[A, B], [(slice(0, A.shape[1]), ka), (slice(A.shape[1], A.shape[1] + B.shape[1]), kb)]
	V = {}  # name -> (X, blocks, q, n_perm, baseline for delta)
	ev = {}
	V["bp_rel"] = (E["bp_rel"], nb(E["bp_rel"]), 0, n_perm, None)
	V["bp_abs"] = (E["bp_abs"], nb(E["bp_abs"]), 0, n_perm, None)
	V["bp_rel_abs"] = (np.c_[E["bp_rel"], E["bp_abs"]], nb(np.c_[E["bp_rel"], E["bp_abs"]]), 0, 0, None)
	V["bp_nogamma"] = (E["bp_nogamma"], nb(E["bp_nogamma"]), 0, 0, None)
	V["bp_rel_clean"] = (E["bp_rel_clean"], nb(E["bp_rel_clean"]), 0, 0, None)
	V["stft_ms"] = (E["stft_ms"], pc(E["stft_ms"]), 0, n_perm, None)
	V["stft_ms_clean"] = (E["stft_ms_clean"], pc(E["stft_ms_clean"]), 0, 0, None)
	V["stft_rp"] = (E["stft_rp"], pc(E["stft_rp"]), 0, 0, None)
	X2, b2 = two(E["bp_rel"], E["stft_ms"], None, 20); V["bp_rel+stft_ms"] = (X2, b2, 0, 0, "bp_rel")
	V["bp_rel_cov"] = (np.c_[E["bp_rel"], Q], nb(E["bp_rel"]), q, 0, None)
	V["stft_ms_cov"] = (np.c_[E["stft_ms"], Q], pc(E["stft_ms"]), q, 0, None)
	for tag, X in S.items():
		main = not tag.startswith("xp1_")
		V[f"synth_{tag}"] = (X, pc(X), 0, n_perm if n_perm_synth is None else n_perm_synth, None)
		for k in (5, 10, 40):
			V[f"synth_{tag}_k{k}"] = (X, pc(X, k), 0, 0, None)
		Xb = X.reshape(len(X), 2, 10, 3, 16, 4, 16, 4).mean((3, 5, 7)).reshape(len(X), -1)
		V[f"synth_{tag}_block"] = (Xb, pc(Xb), 0, 0, None)
		X2, b2 = two(E["bp_rel"], X, None, 20); V[f"bp_rel+synth_{tag}"] = (X2, b2, 0, 0, "bp_rel")
		X2, b2 = two(E["stft_ms"], X, 20, 20); V[f"stft_ms+synth_{tag}"] = (X2, b2, 0, 0, "stft_ms")
		V[f"synth_{tag}_cov"] = (np.c_[X, Q], pc(X), q, 0, None)
	if not eeg_views:  # extra synth tags: keep the two baselines (no permutations) for the added-value comparisons
		V = {k: (v[0], v[1], v[2], 0 if k in ("bp_rel", "stft_ms") else v[3], v[4]) for k, v in V.items()
			 if k in ("bp_rel", "stft_ms") or k.startswith(("synth_", "bp_rel+synth_", "stft_ms+synth_"))}
	if only:
		V = {k: v for k, v in V.items() if k in only}

	res, probs = {}, {}
	for name, (X, blocks, qq, npm, base) in V.items():
		t1 = time.time()
		auc, Pm = cv(X, y, blocks, qq, n_jobs=n_jobs)
		p = Pm.mean(0)
		r = {"auc": auc, "auc_repeats": [float(roc_auc_score(y, pp)) for pp in Pm],
			 "auc_ci95": list(stats.bootstrap_ci(p, y, n_boot=2000)), "n_features": int(X.shape[1] - qq)}
		if npm:
			null = Parallel(n_jobs=n_jobs)(delayed(_perm_one)(X, y, blocks, qq, b) for b in range(npm))
			r.update({"perm_p_greater": float((1 + sum(v >= auc for v in null)) / (1 + npm)),
					  "perm_p_less": float((1 + sum(v <= auc for v in null)) / (1 + npm)), "n_perm": npm,
					  "null_mean": float(np.mean(null)), "null_sd": float(np.std(null))})
			np.save(os.path.join(OUT, cohort, f"null_{name}{out_tag}.npy"), np.array(null))
		res[name] = r; probs[name] = Pm
		np.savez(os.path.join(OUT, cohort, f"oof_probs{out_tag}.npz"), y=y, sub=np.array(subs), **probs)
		print(cohort, name, json.dumps({k: v for k, v in r.items() if k != "auc_repeats"}), f"{(time.time() - t1) / 60:.1f} min", flush=True)
		json.dump({"views": res, **extra}, open(os.path.join(OUT, f"{cohort}{out_tag}.json"), "w"), indent=1, default=float)
	for name, (X, blocks, qq, npm, base) in V.items():
		if base and base in probs:
			res[name]["delta"] = boot_delta(probs[name].mean(0), probs[base].mean(0), y)
	json.dump({"views": res, **extra, "minutes": (time.time() - t0) / 60}, open(os.path.join(OUT, f"{cohort}{out_tag}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--cohorts", nargs="+", default=["fep1", "fep2", "semmelweis", "pd", "fribourg"])
	ap.add_argument("--models", nargs="+", default=MODELS)
	ap.add_argument("--perm", type=int, default=1000)
	ap.add_argument("--perm_synth", type=int, default=None, help="permutations for synthesized views (default: --perm)")
	ap.add_argument("--n_jobs", type=int, default=64)
	ap.add_argument("--only", nargs="*", default=None, help="exact view names to run (smoke tests)")
	ap.add_argument("--synth_tags", nargs="*", default=None, help="synthesized feature files to use (default: <model>, <model>_untrained)")
	ap.add_argument("--out_tag", default="", help="suffix of the output files")
	ap.add_argument("--no_eeg_views", action="store_true", help="only synthesized views (+ the two baselines, no permutations)")
	a = ap.parse_args()
	for c in a.cohorts:
		run(c, a.perm, a.n_jobs, a.models, a.only, a.synth_tags, a.out_tag, not a.no_eeg_views, a.perm_synth)
