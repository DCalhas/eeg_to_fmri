"""
Replication of the pipeline and statistical test of Mehta et al. (arXiv 2504.10752 v2) on CHUR-XP2 (ds002338).

Their pipeline, as published, and what we change because of the data:
	* EEG features: complex Morlet CWT (central frequency 1 Hz, time resolution 3 s, i.e. a constant number of cycles:
	  FWHM 3/f s at frequency f) at every integer frequency 1-40 Hz, relative power (|W|^2 divided by the total over
	  1-40 Hz), downsampled to the TR with an anti-aliasing low-pass (here: the mean over each 1-s TR window),
	  standardized per run. Channels: the 63 EEG channels of ds002338 (theirs: 31).
	* Distributed-lag model: lags 0..M-1 TRs covering ~10 s (theirs M=9 at TR 1.26 s; here M=10 at TR 1 s), intercept.
	* Sparse group lasso, groups = EEG channels (all frequencies x lags of a channel):
	  1/(2n)||y - X b - b0||^2 + lambda (1-alpha) sum_g sqrt(p_g) ||b_g||_2 + lambda alpha ||b||_1, solved by FISTA.
	* Hyperparameters (alpha, lambda) by Bayesian optimization with a GP surrogate and an upper-confidence-bound
	  acquisition (kappa = 0.1) on the validation MSE of block 3-fold cross-validation inside the training data; the
	  chosen point minimizes the GP-smoothed validation-loss surface. Search ranges are not given in the paper:
	  alpha in [0, 1], lambda/lambda_max in [1e-3, 1] (log scale), 5 initial + 15 UCB evaluations.
	* Sessions: ds002338 has one session; the two motor-imagery runs MIpre and MIpost (same block design, recorded
	  before and after neurofeedback training) play the role of their two sessions.
	    inter - train on one run, test on the other, both directions (their inter-session scheme)
	    intra - train on the first half of MIpre + second half of MIpost, test on the other halves, and vice versa
	            (their intra-session scheme)
	  r is averaged over the two folds.
	* Target: their subject-specific somatomotor network (SMN) from first-level task activation masked by a group map.
	  Here, in MNI space (cleaned BOLD registered by experiments/register_xp2_bold.py): voxels of the Harvard-Oxford
	  precentral, postcentral and supplementary motor parcels covered in both runs (the group-level mask) with task z > 3.1 in a first-level GLM fitted on the TRAINING volumes of each fold only (top 50 voxels if
	  fewer pass), averaged. Conditions: TE (trial-evoked: the ROI time course) and TBT (trial-by-trial: the
	  trial-averaged response removed from the BOLD and from every EEG regressor, per run).
	* Metric: Pearson r between predicted and observed test BOLD.
	* Statistical test (theirs): 100 IAAFT surrogates of the training BOLD, the model refitted on each with the
	  hyperparameters re-tuned by the same Bayesian optimization, p = #{r_null >= r} / N; group level: Wilcoxon
	  signed-rank tests with Benjamini-Hochberg FDR, SGL vs the SMR model (C3/C4 8-30 Hz relative power shifted by
	  6.3 s) and vs the massive-univariate-correlation model (the single regressor with the highest training |r|),
	  and inter vs intra.
	* Added control (ours): -features ecg fits the same model on the ECG lead (one group).

GPU: features, FISTA (batched over the real target and all surrogates, each with its own hyperparameters), IAAFT.
CPU inside the job: the GPs of the Bayesian optimization (small).

python mehta_sgl.py -features eeg -out RUNS/mehta_sgl
python mehta_sgl.py -selftest -out /tmp/x
"""
import argparse
import json
import os
import re

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-features', default="eeg", choices=["eeg", "ecg"])
parser.add_argument('-lags', default=10, type=int)
parser.add_argument('-n_surrogates', default=100, type=int)
parser.add_argument('-n_init', default=5, type=int)
parser.add_argument('-n_bo', default=15, type=int)
parser.add_argument('-fista_iter', default=400, type=int)
parser.add_argument('-subjects', default=None, type=str)
parser.add_argument('-schemes', default="intra,inter")
parser.add_argument('-conditions', default="TE,TBT")
parser.add_argument('-gpu_mem', default=30000, type=int)
parser.add_argument('-seed', default=0, type=int)
parser.add_argument('-selftest', action="store_true")
parser.add_argument('-out', required=True)
opt = parser.parse_args()

from utils import process_utils
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=opt.seed)
import tensorflow as tf
from scipy.stats import gamma, wilcoxon, binomtest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

DATA = os.environ['EEG_FMRI_DATASETS']+"/ds002338"
CACHE = os.path.join(os.environ.get('EEG_FMRI_CACHE', opt.out), "mehta_features")
FS, TR = 200, 1.0
FREQS = np.arange(1, 41)
RUNS = ["MIpre", "MIpost"]
MOTOR = ["Precentral Gyrus", "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)", "Postcentral Gyrus"]
KAPPA = 0.1


#---------------------------------------------------------------- EEG features
def morlet_relative_power(x):
	"""x: (C, T) at FS -> (C, F, T) relative power, complex Morlet with FWHM 3/f s at every f in FREQS (FFT convolution)."""
	C, T = x.shape
	n = int(2**np.ceil(np.log2(T+6*FS)))
	X = tf.signal.fft(tf.cast(tf.pad(x, [[0, 0], [0, n-T]]), tf.complex64))
	power = []
	for f in FREQS:
		sigma = (3./f)/(2*np.sqrt(2*np.log(2)))
		t = np.arange(-int(4*sigma*FS), int(4*sigma*FS)+1)/FS
		w = np.exp(2j*np.pi*f*t)*np.exp(-t**2/(2*sigma**2)); w /= np.sum(np.abs(w))
		W = tf.signal.fft(tf.constant(np.pad(w, (0, n-len(w))), tf.complex64))
		y = tf.signal.ifft(X*W[None])[:, len(t)//2:len(t)//2+T]
		power.append(tf.abs(y)**2)
	P = tf.stack(power, 1)
	return P/(tf.reduce_sum(P, 1, keepdims=True)+1e-30)


def eeg_features(sub, run, types):
	"""(n_TR, C, F) relative power per TR, cached."""
	os.makedirs(CACHE, exist_ok=True)
	path = os.path.join(CACHE, "%s_%s_%s.npy" % (sub, run, "-".join(types)))
	if(os.path.exists(path)):
		return np.load(path)
	from utils import eeg_utils
	import mne
	raw = mne.io.read_raw_brainvision(os.path.join(DATA, "derivatives", sub, "eeg_pp", "d_%s_task-%s_eeg_pp.vhdr" % (sub, run)), preload=True, verbose=0)
	picks = eeg_utils.pick_channels(raw, types)
	x = raw.get_data(picks=picks)
	n_tr = x.shape[1]//int(FS*TR)
	#channels in chunks (63 x 40 x 64000 floats at once exhausted the GPU next to other processes)
	F = []
	for c0 in range(0, x.shape[0], 8):
		P = morlet_relative_power(tf.constant(x[c0:c0+8], tf.float32))[..., :n_tr*int(FS*TR)]
		#anti-aliasing low-pass + downsampling to the TR: mean over each TR window
		F.append(tf.reduce_mean(tf.reshape(P, P.shape[:2]+(n_tr, int(FS*TR))), -1).numpy())
	F = np.concatenate(F, 0)
	F = np.transpose(F, (2, 0, 1)).astype(np.float32)
	np.save(path, F)
	return F


def channel_index(sub, run, names):
	import mne
	raw = mne.io.read_raw_brainvision(os.path.join(DATA, "derivatives", sub, "eeg_pp", "d_%s_task-%s_eeg_pp.vhdr" % (sub, run)), preload=False, verbose=0)
	from utils import eeg_utils
	picks = eeg_utils.pick_channels(raw, ("eeg",))
	chs = [raw.ch_names[i] for i in picks]
	return [chs.index(n) for n in names]


#---------------------------------------------------------------- task timing, BOLD and ROI
def onsets(sub, run):
	text = open(os.path.join(DATA, "derivatives", sub, "eeg_pp", "d_%s_task-%s_eeg_pp.vmrk" % (sub, run))).read()
	marks = [(m.group(1).strip(), int(m.group(2))/FS) for m in re.finditer(r"=Stimulus,([^,]+),(\d+),", text)]
	return sorted((t, code) for code, t in marks if code in ("S 99", "S  2"))


def task_regressor(sub, run, n):
	box = np.zeros(n); marks = onsets(sub, run)
	for (t0, code), nxt in zip(marks, marks[1:]+[(n*TR, None)]):
		box[int(t0/TR):int(nxt[0]/TR)] = 1. if code == "S  2" else 0.
	t = np.arange(0, 32, TR)
	hrf = gamma.pdf(t, 6)-gamma.pdf(t, 16)/6.; hrf /= hrf.sum()
	return np.convolve(box, hrf)[:n]


def remove_trial_average(Z, sub, run):
	"""Subtract the average response locked to the task onsets (S 2), epochs of the median onset interval; Z: (n, ...)."""
	starts = [int(t/TR) for t, code in onsets(sub, run) if code == "S  2"]
	L = int(np.median(np.diff(starts)))
	ep = [s for s in starts if s+L <= Z.shape[0]]
	avg = np.mean([Z[s:s+L] for s in ep], 0)
	Z = Z.copy()
	for s in starts:
		e = min(s+L, Z.shape[0]); Z[s:e] -= avg[:e-s]
	return Z


MNI_ROOT = os.path.join(os.environ.get("EEG_FMRI_CACHE", os.path.expanduser("~/eeg_to_fmri/uncertainty/cache")), "xp2_mni")


def load_bold(sub):
	"""{run: (T, V)}: cleaned BOLD registered to MNI 3 mm (experiments/register_xp2_bold.py), voxels of the
	Harvard-Oxford precentral, postcentral and SMA parcels covered by the slab in BOTH runs (temporal SD > 0.5), so the
	same voxels are used in training and test runs (their group-level SMN mask, here anatomical)."""
	atlas = np.load(os.path.join(MNI_ROOT, "atlas_cort_3mm.npy"))
	labels = {v: int(k) for k, v in json.load(open(os.path.join(MNI_ROOT, "labels_cort.json"))).items()}
	sel = np.isin(atlas, [labels[m] for m in MOTOR])
	C = {run: np.load(os.path.join(MNI_ROOT, "%s_%s_clean.npy" % (sub, run)), mmap_mode="r") for run in RUNS}
	for run in RUNS:
		sel &= np.asarray(C[run][::4], np.float32).std(0) > 0.5
	assert sel.sum() >= 50, (sub, "motor voxels covered in both runs", int(sel.sum()))
	return {run: np.asarray(C[run][:, sel], np.float32) for run in RUNS}


def functional_roi(V, design_rows):
	"""Voxel z (t) of the task regressor in a GLM on the training rows only; z > 3.1, else the top 50 voxels.
	design_rows: list of (V_rows (n, Vox), task regressor (n,)) per training segment (own intercept each)."""
	Ys, Xs = [], []
	for k, (Vr, reg) in enumerate(design_rows):
		inter = np.zeros((len(reg), len(design_rows))); inter[:, k] = 1
		Xs.append(np.c_[reg, np.gradient(reg), inter]); Ys.append(Vr)
	X, Y = np.concatenate(Xs), np.concatenate(Ys)
	beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
	res = Y-X@beta; dof = X.shape[0]-X.shape[1]
	s2 = (res**2).sum(0)/dof; c = np.linalg.inv(X.T@X)[0, 0]
	t = beta[0]/np.sqrt(s2*c+1e-12)
	sel = t > 3.1
	if(sel.sum() < 50):
		sel = np.zeros_like(sel); sel[np.argsort(-t)[:50]] = True
	return sel


#---------------------------------------------------------------- sparse group lasso by batched FISTA
class SGL:
	"""Batched over K targets that share the design X (n, G*q) with K per-target (alpha, lambda). X and y centred and
	X scaled with training statistics; groups are consecutive blocks of q columns."""
	def __init__(self, X, n_groups):
		self.mx = tf.reduce_mean(X, 0); self.sx = tf.math.reduce_std(X, 0)+1e-6
		self.X = (X-self.mx)/self.sx; self.n = float(X.shape[0]); self.G = n_groups; self.q = X.shape[1]//n_groups
		v = tf.random.stateless_normal((X.shape[1], 1), seed=[1, 2])
		for _ in range(50):  #power iteration for the Lipschitz constant ||X||^2 / n
			v = tf.matmul(self.X, tf.matmul(self.X, v), transpose_a=True); v = v/tf.norm(v)
		self.L = float(tf.norm(tf.matmul(self.X, v))**2/self.n)

	def lambda_max(self, Y):
		Yc = Y-tf.reduce_mean(Y, 0)
		return tf.reduce_max(tf.abs(tf.matmul(self.X, Yc, transpose_a=True)), 0)/self.n  #(K,), lasso lambda_max

	def prox(self, B, t, lam, alpha):
		B = tf.sign(B)*tf.nn.relu(tf.abs(B)-t*lam*alpha)
		Bg = tf.reshape(B, (self.G, self.q, -1))
		norm = tf.norm(Bg, axis=1, keepdims=True)
		shrink = tf.nn.relu(1.-t*lam*(1.-alpha)*np.sqrt(self.q)/(norm+1e-12))
		return tf.reshape(Bg*shrink, B.shape)

	def fit(self, Y, lam, alpha, B0=None):
		"""Y (n, K); lam, alpha (K,) -> coefficients (P, K) and intercepts (K,)."""
		my = tf.reduce_mean(Y, 0); Yc = Y-my
		lam = tf.constant(lam, tf.float32)[None]; alpha = tf.constant(alpha, tf.float32)[None]
		return self._fista(self.X, Yc, lam, alpha, tf.zeros((self.X.shape[1], Y.shape[1])) if B0 is None else B0), my

	@tf.function
	def _fista(self, X, Yc, lam, alpha, B):
		t = 1./self.L; Z = B; s = tf.constant(1.)
		for _ in tf.range(opt.fista_iter):
			grad = tf.matmul(X, tf.matmul(X, Z)-Yc, transpose_a=True)/self.n
			Bn = self.prox(Z-t*grad, t, lam, alpha)
			sn = (1.+tf.sqrt(1.+4.*s*s))/2.
			Z = Bn+((s-1.)/sn)*(Bn-B); B, s = Bn, sn
		return B

	def predict(self, Xnew, B, my):
		return tf.matmul((Xnew-self.mx)/self.sx, B)+my


#---------------------------------------------------------------- Bayesian optimization (GP + UCB), batched over targets
def bayes_opt(Xtr, Ytr, n_groups, seed):
	"""Per target: (alpha, lambda) minimizing the GP-smoothed block 3-fold validation MSE. Returns lam, alpha (K,)."""
	n, K = Ytr.shape
	edges = np.linspace(0, n, 4).astype(int)
	folds = []
	for k in range(3):
		te = np.arange(edges[k], edges[k+1]); tr = np.setdiff1d(np.arange(n), te)
		model = SGL(tf.gather(Xtr, tr), n_groups)
		folds.append((tr, te, model, model.lambda_max(tf.gather(Ytr, tr)).numpy()))
	lmax = SGL(Xtr, n_groups).lambda_max(Ytr).numpy()
	rng = np.random.RandomState(seed)
	def evaluate(u):  #u: (K, 2) in [0,1]^2 -> (K,) validation MSE; u[:,0] alpha, u[:,1] log-lambda position
		loss = np.zeros(K)
		for tr, te, model, lm in folds:
			lam = lm*10.**(-3+3*u[:, 1]); B, my = model.fit(tf.gather(Ytr, tr), lam.astype(np.float32), u[:, 0].astype(np.float32))
			P = model.predict(tf.gather(Xtr, te), B, my)
			loss += tf.reduce_mean((P-tf.gather(Ytr, te))**2, 0).numpy()/3
		return loss
	U = [rng.rand(K, 2) for _ in range(opt.n_init)]
	Lo = [evaluate(u) for u in U]
	cand = rng.rand(2000, 2)
	kernel = ConstantKernel(1.)*Matern(length_scale=[0.3, 0.3], nu=2.5)+WhiteKernel(1e-3)
	def gp(k):
		g = GaussianProcessRegressor(kernel, normalize_y=True, random_state=0)
		return g.fit(np.array([u[k] for u in U]), np.array([l[k] for l in Lo]))
	for _ in range(opt.n_bo):
		nxt = np.zeros((K, 2))
		for k in range(K):
			m, s = gp(k).predict(cand, return_std=True)
			nxt[k] = cand[np.argmin(m-KAPPA*s)]  #UCB on -loss
		U.append(nxt); Lo.append(evaluate(nxt))
	best = np.zeros((K, 2))
	for k in range(K):
		pts = np.r_[np.array([u[k] for u in U]), cand]
		best[k] = pts[np.argmin(gp(k).predict(pts))]  #minimum of the GP-smoothed validation loss
	return (lmax*10.**(-3+3*best[:, 1])).astype(np.float32), best[:, 0].astype(np.float32)


#---------------------------------------------------------------- IAAFT (Schreiber & Schmitz 1996), as ridge_roi.py
def iaaft(x, n, seed, n_iter=100):
	"""x: (T,) -> (n, T)."""
	T = x.shape[0]
	xb = tf.broadcast_to(tf.constant(x, tf.float32)[None], (n, T))
	amp = tf.abs(tf.signal.rfft(xb)); sorted_x = tf.sort(xb, -1)
	S = tf.gather(sorted_x, tf.argsort(tf.argsort(tf.random.stateless_uniform((n, T), seed=[seed, 7]), -1), -1), batch_dims=1)
	for _ in range(n_iter):
		spec = tf.signal.rfft(S)
		S = tf.signal.irfft(tf.cast(amp, tf.complex64)*tf.exp(tf.complex(0., tf.math.angle(spec))), fft_length=[T])
		S = tf.gather(sorted_x, tf.argsort(tf.argsort(S, -1), -1), batch_dims=1)
	return S.numpy()


def pearson(a, b):
	a = a-a.mean(0); b = b-b.mean(0)
	return (a*b).sum(0)/np.sqrt((a**2).sum(0)*(b**2).sum(0)+1e-12)


#---------------------------------------------------------------- one subject
def lag(F):
	"""(n, P) -> (n, P*lags) with lags 0..L-1, channel-major order preserved by the caller."""
	return np.concatenate([np.pad(F, [[l, 0], [0, 0]])[:F.shape[0]] for l in range(opt.lags)], 1)


def subject_data(sub, condition):
	"""Per run: X (n, G*q) grouped by channel, motor voxels V (n, Vox), task regressor, SMR regressor; first lags-1 TRs dropped."""
	types = ("eeg",) if opt.features == "eeg" else ("ecg",)
	out = {}
	bold = load_bold(sub)
	for run in RUNS:
		F = eeg_features(sub, run, types)  #(n_eeg, C, 40)
		V = bold[run]
		n = min(F.shape[0], V.shape[0]); F, V = F[:n], V[:n]
		F = (F-F.mean(0))/(F.std(0)+1e-6)  #standardized per run
		reg = task_regressor(sub, run, n)
		ci = channel_index(sub, run, ["C3", "C4"])
		smr = eeg_features(sub, run, ("eeg",))[:n][:, ci][:, :, 7:30].mean((1, 2))  #8-30 Hz relative power of C3/C4
		smr = np.r_[np.full(6, smr[0]), smr[:-6]]  #shifted by 6.3 s ~ 6 TRs
		if(condition == "TBT"):
			F = remove_trial_average(F, sub, run); V = remove_trial_average(V, sub, run); smr = remove_trial_average(smr, sub, run)
		C = F.shape[1]
		#lag every (channel, frequency) column, then order columns channel-major so that a group is one channel
		Xl = lag(F.reshape(n, -1)).reshape(n, opt.lags, C, len(FREQS))
		X = np.transpose(Xl, (0, 2, 1, 3)).reshape(n, -1)
		d = opt.lags-1
		out[run] = {"X": X[d:], "V": V[d:], "reg": reg[d:], "smr": smr[d:], "C": C}
	return out


def splits(data, scheme):
	"""-> list of (train segments, test segments), a segment = (run, rows)."""
	idx = {r: np.arange(data[r]["X"].shape[0]) for r in RUNS}
	if(scheme == "inter"):
		return [([(RUNS[0], idx[RUNS[0]])], [(RUNS[1], idx[RUNS[1]])]), ([(RUNS[1], idx[RUNS[1]])], [(RUNS[0], idx[RUNS[0]])])]
	h = {r: (idx[r][:len(idx[r])//2], idx[r][len(idx[r])//2:]) for r in RUNS}
	A = [(RUNS[0], h[RUNS[0]][0]), (RUNS[1], h[RUNS[1]][1])]; B = [(RUNS[0], h[RUNS[0]][1]), (RUNS[1], h[RUNS[1]][0])]
	return [(A, B), (B, A)]


def run_subject(sub, si, condition, scheme):
	data = subject_data(sub, condition)
	G = data[RUNS[0]]["C"]
	r_real, r_null, r_smr, r_muc, info = [], [], [], [], []
	for k, (train, test) in enumerate(splits(data, scheme)):
		cat = lambda segs, key: np.concatenate([data[r][key][rows] for r, rows in segs])
		roi = functional_roi(None, [(data[r]["V"][rows], data[r]["reg"][rows]) for r, rows in train])
		ytr, yte = cat(train, "V")[:, roi].mean(1), cat(test, "V")[:, roi].mean(1)
		Xtr, Xte = tf.constant(cat(train, "X")), tf.constant(cat(test, "X"))
		sur = iaaft(ytr, opt.n_surrogates, opt.seed*100003+si*101+k)
		Y = tf.constant(np.c_[ytr, sur.T], tf.float32)  #(n, 1+N): the real target and every surrogate
		lam, alpha = bayes_opt(Xtr, Y, G, opt.seed*7+si*13+k)
		model = SGL(Xtr, G); B, my = model.fit(Y, lam, alpha)
		P = model.predict(Xte, B, my).numpy()
		r = pearson(P, np.broadcast_to(yte[:, None], P.shape))
		r_real.append(r[0]); r_null.append(r[1:])
		#reference models, fitted (sign and scale) on the training data
		s_tr, s_te = cat(train, "smr"), cat(test, "smr")
		r_smr.append(np.sign(pearson(s_tr, ytr))*pearson(s_te, yte))
		Xtr_np, Xte_np = cat(train, "X"), cat(test, "X")
		c = pearson(Xtr_np, np.broadcast_to(ytr[:, None], Xtr_np.shape)); j = int(np.nanargmax(np.abs(c)))
		r_muc.append(np.sign(c[j])*pearson(Xte_np[:, j], yte))
		info.append({"roi_voxels": int(roi.sum()), "lambda": float(lam[0]), "alpha": float(alpha[0]),
					 "nonzero_groups": int(tf.reduce_sum(tf.cast(tf.norm(tf.reshape(B[:, 0], (G, -1)), axis=1) > 0, tf.int32)))})
	r = float(np.mean(r_real)); null = np.mean(r_null, 0)
	return {"r": r, "p": float(np.sum(null >= r)/len(null)), "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
			"r_folds": [float(v) for v in r_real], "r_smr": float(np.mean(r_smr)), "r_muc": float(np.mean(r_muc)), "folds": info}


def bh(p):
	p = np.asarray(p, float); o = np.argsort(p); m = len(p)
	q = np.minimum.accumulate((p[o]*m/np.arange(1, m+1))[::-1])[::-1]
	out = np.empty(m); out[o] = np.minimum(q, 1.); return out


def selftest():
	"""FISTA with alpha=1 against sklearn's Lasso (same objective, no intercept issue: centred data)."""
	from sklearn.linear_model import Lasso
	rng = np.random.RandomState(0); n, G, q = 120, 6, 5
	X = rng.randn(n, G*q).astype(np.float32); y = (X[:, :q]@rng.randn(q)+0.5*rng.randn(n)).astype(np.float32)
	m = SGL(tf.constant(X), G); lam = 0.05
	opt.fista_iter = 3000
	B, my = m.fit(tf.constant(y[:, None]), np.array([lam], np.float32), np.array([1.], np.float32))
	Xs = ((X-X.mean(0))/(X.std(0)+1e-6))
	ref = Lasso(alpha=lam, fit_intercept=True, tol=1e-10, max_iter=100000).fit(Xs, y).coef_
	err = float(np.max(np.abs(B.numpy()[:, 0]-ref)))
	print("SELFTEST lasso max|diff|", err)
	#group lasso (alpha=0) KKT: for zero groups ||X_g^T r||/n <= lam sqrt(q); for active groups equality
	B, my = m.fit(tf.constant(y[:, None]), np.array([0.2], np.float32), np.array([0.], np.float32))
	r = (y-y.mean())-(Xs@B.numpy()[:, 0]); g = (Xs.T@r/n).reshape(G, q); b = B.numpy()[:, 0].reshape(G, q)
	kkt = [abs(np.linalg.norm(g[i])-0.2*np.sqrt(q)) if np.linalg.norm(b[i]) > 0 else max(0, np.linalg.norm(g[i])-0.2*np.sqrt(q)) for i in range(G)]
	print("SELFTEST group-lasso KKT max violation", float(np.max(kkt)), "active groups", int(sum(np.linalg.norm(b[i]) > 0 for i in range(G))))
	assert err < 1e-3 and np.max(kkt) < 1e-3
	print("SELFTEST_OK")


if(opt.selftest):
	selftest(); raise SystemExit

subs = sorted(d for d in os.listdir(DATA) if d.startswith("sub-xp2"))
#both runs need an EEG export (sub-xp202 has none for MIpre)
missing = [s for s in subs if not all(os.path.exists(os.path.join(DATA, "derivatives", s, "eeg_pp", "d_%s_task-%s_eeg_pp.vhdr" % (s, r))) for r in RUNS)]
print("I: skipped (no EEG export for both runs):", missing, flush=True)
subs = [s for s in subs if s not in missing]
if(opt.subjects):
	subs = [s for s in subs if s in opt.subjects.split(",")]
os.makedirs(opt.out, exist_ok=True)
for condition in opt.conditions.split(","):
	res = {}
	for scheme in opt.schemes.split(","):
		for si, s in enumerate(subs):
			res.setdefault(s, {})[scheme] = run_subject(s, si, condition, scheme)
			print("I:", condition, scheme, s, json.dumps({k: v for k, v in res[s][scheme].items() if k != "folds"}), flush=True)
	group = {"features": opt.features, "condition": condition, "n": len(subs), "lags": opt.lags, "n_surrogates": opt.n_surrogates}
	pvals, names = [], []
	for scheme in opt.schemes.split(","):
		r = [res[s][scheme]["r"] for s in subs]; k = int(sum(res[s][scheme]["p"] < 0.05 for s in subs))
		group[scheme] = {"mean_r": float(np.mean(r)), "k_significant": k, "binomial_p": float(binomtest(k, len(subs), 0.05, alternative="greater").pvalue),
						 "mean_r_smr": float(np.mean([res[s][scheme]["r_smr"] for s in subs])), "mean_r_muc": float(np.mean([res[s][scheme]["r_muc"] for s in subs]))}
		for ref in ["smr", "muc"]:
			pvals.append(wilcoxon(r, [res[s][scheme]["r_"+ref] for s in subs], alternative="greater").pvalue); names.append(scheme+"_sgl_vs_"+ref)
	if(set(opt.schemes.split(",")) == {"intra", "inter"}):
		pvals.append(wilcoxon([res[s]["intra"]["r"] for s in subs], [res[s]["inter"]["r"] for s in subs]).pvalue); names.append("intra_vs_inter")
	group["wilcoxon_p"] = dict(zip(names, map(float, pvals))); group["wilcoxon_p_fdr"] = dict(zip(names, map(float, bh(pvals))))
	print("GROUP", json.dumps(group), flush=True)
	with open(os.path.join(opt.out, "mehta_sgl_%s_%s.json" % (opt.features, condition)), "w") as f:
		json.dump({"group": group, "subjects": res}, f, indent=1)
