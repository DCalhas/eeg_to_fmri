"""
ROI ridge arm (the counterpoint to the deep synthesizer, audit D1 / Mehta et al. arXiv 2504.10752): does lagged EEG
band power predict motor-ROI BOLD on CHUR-XP2 beyond IAAFT-surrogate nulls, with the block design regressed out?

Everything numeric runs in TensorFlow on the GPU.
	* EEG: preprocessed eeg_pp (EEG channels only), log band power per TR window (1 s = 200 samples) by rfft,
	  bands delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz (the export is low-passed at 50 Hz),
	  z-scored per channel and band over the run; lags 0..L-1 TRs (EEG at t-l predicts BOLD at t). The run-level
	  z-score is harmless for the within design: Ridge re-standardizes every feature with training-row statistics,
	  which removes any per-feature affine map exactly.
	* Confound feature sets (-features): motion (6 rigid parameters + FD from the BOLD preprocessing, lagged) and
	  ecg (band power of the ECG channel, lagged, audit A1), fitted and tested exactly like the EEG ridge;
	  -motion_regress removes the motion parameters (+FD) from the BOLD with training-only weights before the fit.
	* BOLD: data/ds002338/derivatives_e2f (shared preprocess_bold, T1-based registration), ROIs by parcel name.
	  EEG window k and fMRI volume k start together (eeg_pp is cut 0.06 s after the first scanner pulse, audit A7).
	* Block design (audit A17): the task boxcar from the S 99 / S 2 markers convolved with a canonical double-gamma
	  HRF, plus its temporal derivative, regressed out of the BOLD with weights fitted on the training data only.
	* Designs:
	    within - per subject: first 60% of volumes train (alpha by blocked 5-fold CV inside), 15-TR gap, last 30% test
	    cross  - leave one subject out: train on the others (alpha on 2 validation subjects), test on the held-out one
	* Statistic: test r of the detrended ROI time courses (mean over ROIs) and pooled detrended R^2.
	* Null: IAAFT surrogates of the training BOLD (N per subject), refit with alpha re-tuned per surrogate;
	  p = (1 + #{null >= real}) / (1 + N); binomial group test over subjects (vendored shared surrogates.py logic).
* Heart rate (-features hr, or as a confound): R peaks on the eeg_pp ECG lead (heart_rate below), instantaneous HR
  interpolated to the TR windows, z-scored over the run, lagged; per-subject QC (mean HR, beats, implausible RR) in the JSON.
* Nested design (-nested ecg+hr+motion, reviewer R2): reduced ridge on the confound blocks vs full ridge on EEG + the same
  confound blocks, each alpha tuned as above; statistic delta R^2 (and delta r) on the test data. Null: the training EEG
  block circularly shifted by >= 30 TRs (per training subject in cross), full model refitted with alpha re-tuned, scored
  against the same reduced model; cross also the swap null (test EEG block from another subject, test confounds kept).
* -cardiac_qc (no ridge): residual cardiac content of the EEG band power, |r| with HR and R^2 of OLS on HR lags 0..5,
  in-sample, against the same statistics with HR circularly shifted (>= 30 TRs, mean of 20 shifts).

python ridge_roi.py -design within -n_surrogates 100 -out RUNS/ridge_roi
python ridge_roi.py -design cross -nested ecg+hr+motion -n_surrogates 100 -out RUNS/ridge_roi
python ridge_roi.py -cardiac_qc -out RUNS/ridge_roi
"""
import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-design', default="within", choices=["within", "cross"])
parser.add_argument('-target', default="resid", choices=["resid", "raw"], help="ROI BOLD with the block design regressed out, or as preprocessed")
parser.add_argument('-rois', default="ctx:Precentral Gyrus,ctx:Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex),ctx:Postcentral Gyrus")
parser.add_argument('-roi_mode', default="composite", choices=["composite", "each"],
					help="composite: one sensorimotor ROI = mean of the listed parcels present in the subject (XP2 coverage and "
						"registration vary: SMA present in 12/17, precentral in 15/17); each: every listed parcel, all must exist")
parser.add_argument('-exclude', default="", type=str, help="comma separated subjects to drop (registration QC: sub-xp206,sub-xp207,sub-xp218)")
parser.add_argument('-features', default="eeg", choices=["eeg", "motion", "ecg", "hr"])
parser.add_argument('-nested', default="", type=str, help="'+'-separated confound blocks from ecg,hr,motion: reduced (confounds) vs full (EEG + confounds) ridge, delta R^2 with an EEG-shift null; -features is ignored")
parser.add_argument('-cardiac_qc', action="store_true", help="only the residual cardiac content of the EEG band power (no ridge), cardiac_qc*.json")
parser.add_argument('-motion_regress', action="store_true", help="regress motion (6 params + FD) out of the BOLD, training-only weights")
parser.add_argument('-lags', default=20, type=int)
parser.add_argument('-skip', default=6, type=int, help="initial volumes dropped (steady state, as bold_shift)")
parser.add_argument('-n_volumes', default=320, type=int, help="volumes covered by the EEG export")
parser.add_argument('-n_surrogates', default=100, type=int)
parser.add_argument('-gpu_mem', default=8000, type=int)
parser.add_argument('-seed', default=0, type=int)
parser.add_argument('-subjects', default=None, type=str, help="comma separated subset (smoke tests)")
parser.add_argument('-roi_source', default="native", choices=["native", "mni"], help="native: parcels of derivatives_e2f (shared epi2mni, failed on some subjects); mni: registered cleaned BOLD (register_xp2_bold.py)")
parser.add_argument('-dataset', default="xp2", choices=["xp2", "xp1"], help="xp1: CHUR-XP1 replication (ds002336 motor localizer, TR 2 s, EEG aligned to the scanner pulses); use with -roi_source mni -n_volumes 162 -lags 10 -skip 3")
parser.add_argument('-out', required=True)
opt = parser.parse_args()

from utils import process_utils
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=opt.seed)
import tensorflow as tf
from scipy.stats import binomtest, gamma

CACHE = os.environ.get("EEG_FMRI_CACHE", os.path.expanduser("~/eeg_to_fmri/uncertainty/cache"))
DS = {"xp2": {"id": "ds002338", "run": "MIpost", "eeg": "d_{sub}_task-{run}_eeg_pp", "tr": 1.0, "task_code": "S  2", "prefix": "sub-xp2",
			  "motion": os.path.join(os.environ['EEG_FMRI_DATASETS'], "ds002338", "derivatives_e2f"), "mni": os.path.join(CACHE, "xp2_mni")},
	  "xp1": {"id": "ds002336", "run": "motorloc", "eeg": "{sub}_task-{run}_eeg_pp", "tr": 2.0, "task_code": "S  1", "prefix": "sub-xp1",
			  "motion": os.path.join(CACHE, "xp1_e2f"), "mni": os.path.join(CACHE, "xp1_mni")}}[opt.dataset]
assert opt.dataset == "xp2" or opt.roi_source == "mni", "CHUR-XP1 uses the registered ROIs"
DATA = os.environ['EEG_FMRI_DATASETS']+"/"+DS["id"]


def eeg_path(sub, ext):
	return os.path.join(DATA, "derivatives", sub, "eeg_pp", DS["eeg"].format(sub=sub, run=DS["run"])+ext)


def run_start(sub):
	"""Sample of volume 0 in the EEG export: XP2's export starts at the first scanner pulse (checked, 0.06 s); XP1's
	motor-localizer export starts ~11 s earlier, so windows are aligned to the first R128 pulse marker."""
	if(opt.dataset == "xp2"):
		return 0
	import re
	return int(re.search(r"=Response,R128,(\d+),", open(eeg_path(sub, ".vmrk")).read()).group(1))-1
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
ALPHAS = 10.**np.arange(-1, 10)
last_swap = (float("nan"), float("nan"))  #up to 1e9: with ~6300 standardized lagged features and ~180 training volumes the smoke run picked the old upper end (1e6)
FS, TR = 200, DS["tr"]
MIN_SHIFT = 30  #TRs: circular-shift nulls (nested EEG block, cardiac QC) move the series by at least this much
RR_OK = (0.33, 2.0)  #s: plausible RR intervals (30-180 bpm)
CONFOUNDS = ("ecg", "hr", "motion")


#---------------------------------------------------------------- data
def eeg_bandpower(sub, types=("eeg",)):
	"""(n_volumes, channels*bands) log band power per TR, z-scored over the run; types=("ecg",) for the ECG channel."""
	from utils import eeg_utils
	import mne
	raw = mne.io.read_raw_brainvision(eeg_path(sub, ".vhdr"), preload=True, verbose=0)
	picks = eeg_utils.pick_channels(raw, types)
	s0 = run_start(sub)
	x = tf.constant(raw.get_data(picks=picks)[:, s0:s0+opt.n_volumes*int(FS*TR)], tf.float32)  #(C, T*FS*TR)
	assert x.shape[1] == opt.n_volumes*int(FS*TR), (sub, "EEG shorter than n_volumes", x.shape)
	w = tf.reshape(x, (x.shape[0], opt.n_volumes, int(FS*TR)))
	power = tf.abs(tf.signal.rfft(w-tf.reduce_mean(w, -1, keepdims=True)))**2  #(C, T, FS*TR/2+1), bins of 1/TR Hz
	feats = tf.stack([tf.math.log(tf.reduce_mean(power[..., int(lo*TR):int(hi*TR)], -1)+1e-20) for lo, hi in BANDS], -1)  #(C, T, B)
	feats = tf.transpose(feats, (1, 0, 2))
	feats = (feats-tf.reduce_mean(feats, 0))/(tf.math.reduce_std(feats, 0)+1e-6)
	return tf.reshape(feats, (opt.n_volumes, -1))


def r_peaks(ecg):
	"""R-peak samples of a (scanner) ECG lead. scipy rather than mne.preprocessing.find_ecg_events: the latter band-passes
	and auto-thresholds for QRS templates of out-of-scanner ECG and gives no handle on polarity or the refractory period.
	5-30 Hz zero-phase band-pass (removes baseline wander/respiration and residual high-frequency artefact); polarity chosen
	so the larger typical extreme is positive (scanner ECG is often inverted; if the S wave wins, the beat count is unchanged);
	peaks above half the median of the per-2-s maxima (robust R amplitude: >= 1 beat per 2 s at >= 30 bpm, insensitive to
	sparse residual gradient spikes) with a 0.25 s refractory period, deliberately below RR_OK[0] so double detections
	(T waves, artefacts) remain visible in the implausible-RR QC instead of being suppressed."""
	from scipy.signal import butter, sosfiltfilt, find_peaks
	x = sosfiltfilt(butter(3, (5, 30), "bandpass", fs=FS, output="sos"), np.asarray(ecg, np.float64))
	x = x-np.median(x)
	seg = 2*FS; n = len(x)//seg*seg
	flip = bool(np.median((-x[:n]).reshape(-1, seg).max(1)) > np.median(x[:n].reshape(-1, seg).max(1)))
	if(flip):
		x = -x
	peaks, _ = find_peaks(x, distance=int(0.25*FS), height=0.5*np.median(x[:n].reshape(-1, seg).max(1)))
	return peaks, flip


def heart_rate(sub):
	"""-> (n_volumes, 1) instantaneous HR per TR window, z-scored over the run, and its QC dict.
	HR of each RR interval (60/RR) placed at the interval midpoint; implausible intervals (outside RR_OK) are dropped and
	bridged by linear interpolation (edges held constant); the sample-rate HR is averaged over window k = samples
	[k*200, (k+1)*200), aligned like eeg_bandpower. Peaks are detected on the whole export (beats just outside the run
	anchor the edges); the QC counts the beats/intervals inside the run."""
	from utils import eeg_utils
	import mne
	raw = mne.io.read_raw_brainvision(eeg_path(sub, ".vhdr"), preload=True, verbose=0)
	picks = eeg_utils.pick_channels(raw, ("ecg",))
	assert len(picks) == 1, (sub, "expected one ECG channel", [raw.ch_names[p] for p in picks])
	peaks, flip = r_peaks(raw.get_data(picks=picks)[0])
	peaks = peaks-run_start(sub)  #volume 0 at sample 0
	n = opt.n_volumes*int(FS*TR)
	t = peaks/FS; rr = np.diff(t); mid = (t[1:]+t[:-1])/2.
	ok = (rr >= RR_OK[0])&(rr <= RR_OK[1])
	assert ok.sum() >= 2, (sub, "fewer than 2 plausible RR intervals", len(peaks))
	hr = np.interp(np.arange(n)/FS, mid[ok], 60./rr[ok]).reshape(opt.n_volumes, -1).mean(1)
	inside = (mid >= 0)&(mid < n/FS)
	qc = {"mean_hr_bpm": float(hr.mean()), "sd_hr_bpm": float(hr.std()), "n_beats": int(np.sum(peaks < n)),
		"n_rr": int(inside.sum()), "frac_implausible_rr": float(np.mean(~ok[inside])) if inside.any() else float("nan"),
		"frac_rr_short": float(np.mean(rr[inside] < RR_OK[0])) if inside.any() else float("nan"), "ecg_flipped": flip}
	hr = (hr-hr.mean())/(hr.std()+1e-6)
	return tf.constant(hr[:, None], tf.float32), qc


def cardiac_qc(B, h, seed):
	"""Residual cardiac content of the EEG (descriptive, IN-SAMPLE: no train/test split). B: (n_volumes, C*bands) z-scored
	band power (eeg_bandpower, before lagging), h: (n_volumes,) z-scored HR; the first opt.skip rows dropped after lagging.
	Per column: r with HR (lag 0) and R^2 of OLS on [1, HR lags 0..5]; summarized over channel x band. Reference: the same
	with HR circularly shifted by >= MIN_SHIFT TRs, mean over 20 shifts (in-sample R^2 with 7 regressors is > 0 by chance)."""
	B = np.asarray(B, np.float64)[opt.skip:]; Bc = B-B.mean(0)
	def stats(x):
		H = np.stack([np.r_[np.zeros(l), x[:len(x)-l]] for l in range(6)], 1)[opt.skip:]
		H = np.c_[np.ones(len(H)), H]
		R2 = 1-np.sum((B-H@np.linalg.lstsq(H, B, rcond=None)[0])**2, 0)/(np.sum(Bc**2, 0)+1e-12)
		xc = x[opt.skip:]-x[opt.skip:].mean()
		a = np.abs(Bc.T@xc/np.sqrt(np.sum(Bc**2, 0)*np.sum(xc**2)+1e-12))
		return {"median_abs_r": float(np.median(a)), "q95_abs_r": float(np.quantile(a, 0.95)), "frac_abs_r_gt_0.2": float(np.mean(a > 0.2)),
			"median_R2": float(np.median(R2)), "q95_R2": float(np.quantile(R2, 0.95)),
			"median_abs_r_by_band": [float(v) for v in np.median(a.reshape(-1, len(BANDS)), 0)]}  #columns are channel-major (C, B)
	h = np.asarray(h, np.float64)
	rng = np.random.default_rng(seed)
	nulls = [stats(np.roll(h, k)) for k in rng.integers(MIN_SHIFT, len(h)-MIN_SHIFT+1, 20)]
	return {"real": stats(h), "null_shift_mean": {k: (np.mean([v[k] for v in nulls], 0).tolist() if k.endswith("by_band") else float(np.mean([v[k] for v in nulls]))) for k in nulls[0]}}


def lagged(F):
	"""(T, P) -> (T, P*lags): row t holds F[t-l] for l = 0..lags-1 (zero before the run start)."""
	return tf.concat([tf.pad(F, [[l, 0], [0, 0]])[:F.shape[0]] for l in range(opt.lags)], 1)


def task_regressors(sub, n):
	"""HRF-convolved task boxcar (+ temporal derivative) from the eeg_pp markers, in TR units."""
	import re
	text = open(eeg_path(sub, ".vmrk")).read(); s0 = run_start(sub)
	marks = [(m.group(1).strip(), (int(m.group(2))-1-s0)/FS) for m in re.finditer(r"=Stimulus,([^,]+),(\d+),", text)]
	box = np.zeros(n); state = 0.
	starts = sorted((max(t, 0.), 1. if code == DS["task_code"].strip() else 0.) for code, t in marks if code in ("S 99", DS["task_code"].strip()))
	for (t0, s), nxt in zip(starts, starts[1:]+[(n*TR, None)]):
		box[int(t0/TR):int(nxt[0]/TR)] = s
	t = np.arange(0, 32, TR)
	hrf = gamma.pdf(t, 6)-gamma.pdf(t, 16)/6.; hrf /= hrf.sum()
	reg = np.convolve(box, hrf)[:n]
	return np.stack([reg, np.gradient(reg), np.ones(n)], 1).astype(np.float32)


def motion_features(sub):
	"""(n_volumes, 7): the 6 rigid motion parameters and FD of the BOLD preprocessing, z-scored over the run."""
	with np.load(os.path.join(DS["motion"], sub+"_task-"+DS["run"]+"_bold_e2f.npz")) as f:
		M = np.c_[f["motion"], f["fd"]][:opt.n_volumes].astype(np.float32)
	return (M-M.mean(0))/(M.std(0)+1e-6)


MNI_ROOT = os.path.join(os.environ.get("EEG_FMRI_CACHE", os.path.expanduser("~/eeg_to_fmri/uncertainty/cache")), "xp2_mni")


def roi_bold_mni(sub, names, run=None):
	"""ROI time courses from the cleaned BOLD registered to MNI (experiments/register_xp2_bold.py, world-coordinate
	initialized EPI->T1): mean over the parcel's voxels covered by the slab (temporal SD > 0.5 of the z-scored data).
	The native-space parcels of derivatives_e2f came from an EPI->T1 registration that failed on some subjects."""
	import json as _json
	run = run or DS["run"]
	C = np.load(os.path.join(DS["mni"], "%s_%s_clean.npy" % (sub, run)), mmap_mode="r")[:opt.n_volumes].astype(np.float32)
	atlas = np.load(os.path.join(DS["mni"], "atlas_cort_3mm.npy")); labels = {v: int(k) for k, v in _json.load(open(os.path.join(DS["mni"], "labels_cort.json"))).items()}
	covered = C.std(0) > 0.5
	cols, present = [], []
	for r in names:
		sel = (atlas == labels[r.replace("ctx:", "")]) & covered
		if(sel.sum() >= 5):
			cols.append(C[:, sel].mean(1)); present.append(r)
	if(opt.roi_mode == "composite"):
		assert len(cols), (sub, "no sensorimotor parcel covered")
		return np.stack(cols, 1).mean(1, keepdims=True), present
	assert len(cols) == len(names), (sub, "missing ROIs (coverage)", sorted(set(names)-set(present)))
	return np.stack(cols, 1), present


def roi_bold(sub, names):
	if(opt.roi_source == "mni"):
		return roi_bold_mni(sub, names)
	with np.load(os.path.join(DATA, "derivatives_e2f", sub+"_task-MIpost_bold_e2f.npz")) as f:
		have = list(f["parcel_names"])
		present = [r for r in names if r in have]
		P = f["parcels"][:opt.n_volumes].astype(np.float32)
		if(opt.roi_mode == "composite"):
			assert len(present), (sub, "no sensorimotor parcel present")
			return P[:, [have.index(r) for r in present]].mean(1, keepdims=True), present
		missing = [r for r in names if r not in have]
		assert not missing, (sub, "missing ROIs (registration/coverage)", missing)
		return P[:, [have.index(r) for r in names]], present


#---------------------------------------------------------------- ridge (thin SVD computed once per design matrix)
class Ridge:
	"""Ridge for a fixed training design Xtr and test design Xte: the decomposition depends only on X, so every target
	(real BOLD and each surrogate) and every alpha reuses it. X standardized with training statistics, Y centred."""
	def __init__(self, Xtr, Xte, keep=False):
		#kernel (dual) form: beta = Xs^T (Xs Xs^T + alpha I)^-1 y, via the eigendecomposition of the n x n Gram matrix.
		#Equivalent to the primal/SVD ridge; the GPU SVD of the ~4100 x 6300 cross-subject design segfaulted (205578).
		mx, sx = tf.reduce_mean(Xtr, 0), tf.math.reduce_std(Xtr, 0)+1e-6
		Xs = tf.cast((Xtr-mx)/sx, tf.float64); Xt = tf.cast((Xte-mx)/sx, tf.float64)
		lam, self.Q = tf.linalg.eigh(tf.matmul(Xs, Xs, transpose_b=True))
		self.lam = tf.maximum(lam, 0.)
		self.KtQ = tf.matmul(tf.matmul(Xt, Xs, transpose_b=True), self.Q)
		if(keep):
			self.mx, self.sx, self.Xs = mx, sx, Xs

	def at(self, Xte):
		"""The same fit on another test design, no new decomposition (needs keep=True); identical ops to a fresh Ridge."""
		other = object.__new__(Ridge); other.lam, other.Q = self.lam, self.Q
		other.KtQ = tf.matmul(tf.matmul(tf.cast((Xte-self.mx)/self.sx, tf.float64), self.Xs, transpose_b=True), self.Q)
		return other

	def predict(self, Ytr, alphas):
		"""-> (A, n_te, K)"""
		Y = tf.cast(Ytr, tf.float64); my = tf.reduce_mean(Y, 0)
		QtY = tf.matmul(self.Q, Y-my, transpose_a=True)
		d = 1./(self.lam[None]+tf.constant(alphas, tf.float64)[:, None])
		return tf.cast(tf.einsum("nr,ar,rk->ank", self.KtQ, d, QtY)+my, tf.float32)


def detrend(Y):
	t = tf.cast(tf.range(Y.shape[0]), tf.float32)[:, None]; t = t-tf.reduce_mean(t)
	Yc = Y-tf.reduce_mean(Y, 0)
	return Yc-t*tf.reduce_sum(t*Yc, 0)/tf.reduce_sum(t**2)


def score(P, Y):
	"""mean over ROIs of the detrended r, and pooled detrended R^2."""
	Pd, Yd = detrend(P), detrend(Y)
	r = tf.reduce_sum(Pd*Yd, 0)/tf.sqrt(tf.reduce_sum(Pd**2, 0)*tf.reduce_sum(Yd**2, 0)+1e-12)
	R2 = 1-tf.reduce_sum((Yd-Pd)**2)/tf.reduce_sum(Yd**2)
	return float(tf.reduce_mean(r)), float(R2)


class Tuner:
	"""alpha by blocked n-fold CV over time inside the training part (pooled R^2), lag-sized gaps around each fold."""
	def __init__(self, Xtr, n_folds=5):
		n = Xtr.shape[0]; edges = np.linspace(0, n, n_folds+1).astype(int); self.folds = []
		for k in range(n_folds):
			te = np.arange(edges[k], edges[k+1]); tr = np.setdiff1d(np.arange(n), np.arange(max(0, edges[k]-opt.lags), min(n, edges[k+1]+opt.lags)))
			self.folds.append((tr, te, Ridge(tf.gather(Xtr, tr), tf.gather(Xtr, te))))

	def __call__(self, Ytr):
		total = np.zeros(len(ALPHAS))
		for tr, te, model in self.folds:
			P = model.predict(tf.gather(Ytr, tr), ALPHAS); Yte = tf.gather(Ytr, te)
			total += (1-tf.reduce_sum((P-Yte)**2, (1, 2))/tf.reduce_sum((Yte-tf.reduce_mean(Yte, 0))**2)).numpy()
		return ALPHAS[int(np.argmax(total))]


def regress_out(Y, D, train_rows):
	"""Remove the design D from Y with weights fitted on the training rows only."""
	Dt, Yt = tf.gather(D, train_rows), tf.gather(Y, train_rows)
	beta = tf.linalg.lstsq(Dt, Yt, l2_regularizer=1e-6)
	return Y-tf.matmul(D, beta)


#---------------------------------------------------------------- IAAFT on the GPU (Schreiber & Schmitz 1996)
def iaaft(x, n, seed, n_iter=100):
	"""x: (T, K) -> (n, T, K) surrogates with the amplitude distribution and ~power spectrum of every column."""
	T = x.shape[0]
	xb = tf.broadcast_to(x[None], (n,)+tuple(x.shape))
	amp = tf.abs(tf.signal.rfft(tf.transpose(xb, (0, 2, 1))))  #(n, K, F)
	sorted_x = tf.sort(tf.transpose(xb, (0, 2, 1)), -1)
	noise = tf.random.stateless_uniform((n, x.shape[1], T), seed=[seed, 7])
	S = tf.gather(sorted_x, tf.argsort(tf.argsort(noise, -1), -1), batch_dims=2)  #random permutation per column
	for _ in range(n_iter):
		spec = tf.signal.rfft(S)
		S = tf.signal.irfft(tf.cast(amp, tf.complex64)*tf.exp(tf.complex(0., tf.math.angle(spec))), fft_length=[T])
		S = tf.gather(sorted_x, tf.argsort(tf.argsort(S, -1), -1), batch_dims=2)
	return tf.transpose(S, (0, 2, 1))


def perm_p(real, null):
	return float((1+np.sum(np.asarray(null) >= real))/(1+len(null)))


#---------------------------------------------------------------- designs
def block_baseline(Dtr, Ytr, Dte, Yte):
	"""EEG-free task baseline (audit A17): the HRF-convolved block regressors fitted on the training data, scored on the test."""
	beta = tf.linalg.lstsq(Dtr, Ytr, l2_regularizer=1e-6)
	return score(tf.matmul(Dte, beta), Yte)


def within(sub, X, Y, D):
	n = X.shape[0]; n_tr = int(0.6*n); gap = 15; te = np.arange(n_tr+gap, n); tr = np.arange(n_tr)
	block = block_baseline(tf.gather(D, tr), tf.gather(Y, tr), tf.gather(D, te), tf.gather(Y, te))  #on the raw target
	if(opt.target == "resid"):
		Y = regress_out(Y, D, tr)
	if(opt.motion_regress):
		Y = regress_out(Y, data[sub]["M"], tr)
	Xtr, Ytr, Xte, Yte = tf.gather(X, tr), tf.gather(Y, tr), tf.gather(X, te), tf.gather(Y, te)
	tune, model = Tuner(Xtr), Ridge(Xtr, Xte)
	alpha = tune(Ytr)
	real = score(model.predict(Ytr, [alpha])[0], Yte)
	null = []
	for Ys in iaaft(Ytr, opt.n_surrogates, opt.seed*1000+subs.index(sub)):
		a = tune(Ys)  #hyperparameters re-tuned on every surrogate (D1 fix)
		null.append(score(model.predict(Ys, [a])[0], Yte))
	return real, null, alpha, block


def cross_target(s):
	"""cross-design target: block design (resid) and motion (-motion_regress) removed with the subject's whole run."""
	Y, all_rows = data[s]["Y"], np.arange(data[s]["Y"].shape[0])
	if(opt.target == "resid"):
		Y = regress_out(Y, data[s]["D"], all_rows)
	if(opt.motion_regress):
		Y = regress_out(Y, data[s]["M"], all_rows)
	return Y


def cross(test_sub, data, subs):
	others = [s for s in subs if s != test_sub]
	val = others[:2]; train = others[2:]
	block = block_baseline(tf.concat([data[s]["D"] for s in train], 0), tf.concat([data[s]["Y"] for s in train], 0), data[test_sub]["D"], data[test_sub]["Y"])
	def stack(ss, key):
		return tf.concat([data[s][key] for s in ss], 0)
	Ys = {s: cross_target(s) for s in subs}
	Xtr, Ytr = stack(train, "X"), tf.concat([Ys[s] for s in train], 0)
	Xva, Yva = stack(val, "X"), tf.concat([Ys[s] for s in val], 0)
	on_val, on_test = Ridge(Xtr, Xva), Ridge(Xtr, data[test_sub]["X"])
	def best_alpha(Ytrain):
		P = on_val.predict(Ytrain, ALPHAS)
		return ALPHAS[int(np.argmax((1-tf.reduce_sum((P-Yva)**2, (1, 2))/tf.reduce_sum((Yva-tf.reduce_mean(Yva, 0))**2)).numpy()))]
	alpha = best_alpha(Ytr)
	real = score(on_test.predict(Ytr, [alpha])[0], Ys[test_sub])
	#swap null (no refit): the same trained model fed another subject's features over the same run timing; real ~ swap
	#means run/task-locked rather than subject-specific information (audit post-hoc v2)
	swaps = [score(Ridge(Xtr, data[o]["X"]).predict(Ytr, [alpha])[0], Ys[test_sub]) for o in subs if o != test_sub and data[o]["X"].shape == data[test_sub]["X"].shape]
	global last_swap
	last_swap = (float(np.mean([w[0] for w in swaps])), float(np.mean([w[1] for w in swaps])))
	null = []
	#each training subject's BOLD replaced by its own IAAFT surrogate; all N surrogates of a subject in one batched call
	#(one call per subject and surrogate meant ~1e3 small IAAFT loops per test subject and segfaulted after 1-2 subjects)
	surrogates = [iaaft(Ys[s], opt.n_surrogates, opt.seed*100003+subs.index(test_sub)*997+i) for i, s in enumerate(train)]
	for b in range(opt.n_surrogates):
		Ysur = tf.concat([S[b] for S in surrogates], 0)
		a = best_alpha(Ysur)
		null.append(score(on_test.predict(Ysur, [a])[0], Ys[test_sub]))
	return real, null, alpha, block


#---------------------------------------------------------------- nested: EEG + confounds vs confounds (delta R^2)
def delta(full, red):
	return (full[0]-red[0], full[1]-red[1])


def nested_within(sub):
	"""Same split, target and alpha tuning as within(); E: lagged EEG block, C: lagged confound blocks."""
	E, C, Y = data[sub]["E"], data[sub]["C"], data[sub]["Y"]
	n = Y.shape[0]; n_tr = int(0.6*n); gap = 15; te = np.arange(n_tr+gap, n); tr = np.arange(n_tr)
	if(opt.target == "resid"):
		Y = regress_out(Y, data[sub]["D"], tr)
	if(opt.motion_regress):
		Y = regress_out(Y, data[sub]["M"], tr)
	Ytr, Yte = tf.gather(Y, tr), tf.gather(Y, te)
	Etr, Ete, Ctr, Cte = tf.gather(E, tr), tf.gather(E, te), tf.gather(C, tr), tf.gather(C, te)
	def fit(Xtr, Xte):
		a = Tuner(Xtr)(Ytr)
		return score(Ridge(Xtr, Xte).predict(Ytr, [a])[0], Yte), a
	red, a_red = fit(Ctr, Cte)
	Xte = tf.concat([Ete, Cte], 1)
	full, a_full = fit(tf.concat([Etr, Ctr], 1), Xte)
	#EEG-shift null: training EEG rows rolled (>= MIN_SHIFT TRs) against BOLD + confounds, Tuner rebuilt (alpha re-tuned)
	rng = np.random.default_rng([opt.seed, subs.index(sub)])
	null = [delta(fit(tf.concat([tf.roll(Etr, int(k), 0), Ctr], 1), Xte)[0], red) for k in rng.integers(MIN_SHIFT, n_tr-MIN_SHIFT+1, opt.n_surrogates)]
	return {"red": red, "full": full, "a_red": a_red, "a_full": a_full, "null": null, "swap": (float("nan"), float("nan"))}


def nested_cross(test_sub):
	"""Same split, target and alpha selection (2 validation subjects) as cross(); one Gram decomposition per training design
	serves the validation and test designs (Ridge.at). The validation and test EEG keep their pairing in the null."""
	others = [s for s in subs if s != test_sub]
	val = others[:2]; train = others[2:]
	Ys = {s: cross_target(s) for s in subs}
	Ytr, Yva, Yte = tf.concat([Ys[s] for s in train], 0), tf.concat([Ys[s] for s in val], 0), Ys[test_sub]
	def stack(ss, key):
		return tf.concat([data[s][key] for s in ss], 0)
	def full_X(ss):
		return tf.concat([stack(ss, "E"), stack(ss, "C")], 1)
	def fit(Xtr, Xva, Xte):
		m = Ridge(Xtr, Xva, keep=True); P = m.predict(Ytr, ALPHAS)
		a = ALPHAS[int(np.argmax((1-tf.reduce_sum((P-Yva)**2, (1, 2))/tf.reduce_sum((Yva-tf.reduce_mean(Yva, 0))**2)).numpy()))]
		return score(m.at(Xte).predict(Ytr, [a])[0], Yte), a, m
	red, a_red, _ = fit(stack(train, "C"), stack(val, "C"), data[test_sub]["C"])
	Xva, Xte = full_X(val), full_X([test_sub])
	full, a_full, m = fit(full_X(train), Xva, Xte)
	#swap null (no refit): the test subject's EEG block from another subject, its own confounds kept
	swaps = [delta(score(m.at(tf.concat([data[o]["E"], data[test_sub]["C"]], 1)).predict(Ytr, [a_full])[0], Yte), red)
			for o in subs if o != test_sub and data[o]["E"].shape == data[test_sub]["E"].shape]
	del m
	#EEG-shift null: each training subject's EEG rows rolled by its own offset (>= MIN_SHIFT TRs), alpha re-selected
	rng = np.random.default_rng([opt.seed, subs.index(test_sub)])
	Ctr = stack(train, "C"); null = []
	for b in range(opt.n_surrogates):
		Etr = tf.concat([tf.roll(data[s]["E"], int(rng.integers(MIN_SHIFT, data[s]["E"].shape[0]-MIN_SHIFT+1)), 0) for s in train], 0)
		null.append(delta(fit(tf.concat([Etr, Ctr], 1), Xva, Xte)[0], red))
	return {"red": red, "full": full, "a_red": a_red, "a_full": a_full, "null": null,
			"swap": (float(np.mean([w[0] for w in swaps])), float(np.mean([w[1] for w in swaps])))}


def wilcoxon_greater(a, b=None):
	from scipy.stats import wilcoxon
	return float(wilcoxon(a, b, alternative="greater").pvalue) if len(a) >= 5 else None


def run_nested():
	res = {}
	for s in subs:
		o = nested_within(s) if opt.design == "within" else nested_cross(s)
		null = np.array(o["null"]); d = delta(o["full"], o["red"])
		res[s] = {"r_full": o["full"][0], "R2_full": o["full"][1], "r_reduced": o["red"][0], "R2_reduced": o["red"][1], "delta_r": d[0], "delta_R2": d[1],
				"alpha_full": float(o["a_full"]), "alpha_reduced": float(o["a_red"]), "alpha_at_grid_edge": bool(o["a_full"] in (ALPHAS[0], ALPHAS[-1]) or o["a_red"] in (ALPHAS[0], ALPHAS[-1])),
				"p_delta_R2": perm_p(d[1], null[:, 1]), "p_delta_r": perm_p(d[0], null[:, 0]),
				"null_delta_R2_mean": float(null[:, 1].mean()), "null_delta_R2_q95": float(np.quantile(null[:, 1], 0.95)), "null_delta_r_mean": float(null[:, 0].mean()),
				"swap_delta_r": o["swap"][0], "swap_delta_R2": o["swap"][1], "rois_present": present[s]}
		if(s in hr_qc):
			res[s]["hr_qc"] = hr_qc[s]
		print("I:", s, json.dumps(res[s]), flush=True)
	v = list(res.values())
	k = int(sum(x["p_delta_R2"] < 0.05 for x in v)); k_r = int(sum(x["p_delta_r"] < 0.05 for x in v))
	group = {"design": opt.design, "target": opt.target, "nested": confounds, "null": "EEG-block circular shift >= %d TRs, alpha re-tuned" % MIN_SHIFT,
			"motion_regress": opt.motion_regress, "rois": rois, "roi_mode": opt.roi_mode, "excluded": opt.exclude, "lags": opt.lags, "n_surrogates": opt.n_surrogates,
			"mean_delta_R2": float(np.mean([x["delta_R2"] for x in v])), "mean_delta_r": float(np.mean([x["delta_r"] for x in v])),
			"mean_R2_full": float(np.mean([x["R2_full"] for x in v])), "mean_R2_reduced": float(np.mean([x["R2_reduced"] for x in v])),
			"delta_R2_gt0_wilcoxon_p": wilcoxon_greater([x["delta_R2"] for x in v]),
			"k_significant": k, "n": len(v), "binomial_p": float(binomtest(k, len(v), 0.05, alternative="greater").pvalue),
			"k_significant_delta_r": k_r, "binomial_p_delta_r": float(binomtest(k_r, len(v), 0.05, alternative="greater").pvalue)}
	if(opt.design == "cross"):
		group.update({"swap_mean_delta_R2": float(np.nanmean([x["swap_delta_R2"] for x in v])), "swap_mean_delta_r": float(np.nanmean([x["swap_delta_r"] for x in v])),
				"real_vs_swap_delta_R2_wilcoxon_p": wilcoxon_greater([x["delta_R2"] for x in v], [x["swap_delta_R2"] for x in v])})
	print("GROUP", json.dumps(group), flush=True)
	with open(os.path.join(opt.out, "ridge_%s_%s_nested-%s_%s%s%s.json" % (opt.design, opt.target, "+".join(confounds), opt.roi_mode, "_motreg" if opt.motion_regress else "", "_qcexcl" if opt.exclude else "")), "w") as f:
		json.dump({"group": group, "subjects": res}, f, indent=1)


def run_cardiac_qc():
	res = {}
	for s in subs:
		h, qc = heart_rate(s)
		res[s] = {"hr_qc": qc, **cardiac_qc(eeg_bandpower(s).numpy(), h.numpy()[:, 0], [opt.seed, subs.index(s)])}
		print("I:", s, json.dumps(res[s]), flush=True)
	v = list(res.values()); keys = [k for k in v[0]["real"] if not k.endswith("by_band")]
	group = {"excluded": opt.exclude, "skip": opt.skip, "n": len(v), "in_sample": True, "hr_lags": 6, "null": "HR circular shift >= %d TRs, mean of 20" % MIN_SHIFT,
			"mean_hr_bpm": float(np.mean([x["hr_qc"]["mean_hr_bpm"] for x in v])), "max_frac_implausible_rr": float(np.nanmax([x["hr_qc"]["frac_implausible_rr"] for x in v])),
			**{"mean_"+k: float(np.mean([x["real"][k] for x in v])) for k in keys}, **{"null_mean_"+k: float(np.mean([x["null_shift_mean"][k] for x in v])) for k in keys},
			**{"real_gt_null_wilcoxon_p_"+k: wilcoxon_greater([x["real"][k] for x in v], [x["null_shift_mean"][k] for x in v]) for k in keys}}
	print("GROUP", json.dumps(group), flush=True)
	with open(os.path.join(opt.out, "cardiac_qc%s.json" % ("_qcexcl" if opt.exclude else "")), "w") as f:
		json.dump({"group": group, "subjects": res}, f, indent=1)


#---------------------------------------------------------------- main
rois = opt.rois.split(",")
subs = sorted(d for d in os.listdir(DATA) if d.startswith(DS["prefix"]))
if(opt.subjects):
	subs = [s for s in subs if s in opt.subjects.split(",")]
subs = [s for s in subs if s not in opt.exclude.split(",")]
os.makedirs(opt.out, exist_ok=True)
if(opt.cardiac_qc):
	run_cardiac_qc()
	raise SystemExit(0)
confounds = opt.nested.split("+") if opt.nested else []
assert len(set(confounds)) == len(confounds) and all(c in CONFOUNDS for c in confounds), ("-nested takes '+'-separated blocks from", CONFOUNDS, confounds)
present = {}
data = {}
hr_qc = {}
for s in subs:
	M = tf.constant(motion_features(s))
	if(opt.features == "hr" and not confounds or "hr" in confounds):
		hr, hr_qc[s] = heart_rate(s)
	def feature_block(name):
		return {"eeg": lambda: eeg_bandpower(s), "ecg": lambda: eeg_bandpower(s, ("ecg",)), "motion": lambda: M, "hr": lambda: hr}[name]()
	if(confounds):
		E = lagged(feature_block("eeg"))[opt.skip:]
		C = lagged(tf.concat([feature_block(c) for c in confounds], 1))[opt.skip:]
		X = tf.concat([E, C], 1)
	else:
		X = lagged(feature_block(opt.features))[opt.skip:]
	Y, present[s] = roi_bold(s, rois); Y = tf.constant(Y)[opt.skip:]
	D = tf.constant(task_regressors(s, opt.n_volumes))[opt.skip:]
	data[s] = {"X": X, "Y": Y, "D": D, "M": tf.concat([M[opt.skip:], tf.ones((M.shape[0]-opt.skip, 1))], 1)}
	if(confounds):
		data[s].update({"E": E, "C": C})
	print("I:", s, "X", X.shape, "Y", Y.shape, flush=True)

if(confounds):
	run_nested()
	raise SystemExit(0)
res = {}
for s in subs:
	real, null, alpha, block = within(s, data[s]["X"], data[s]["Y"], data[s]["D"]) if opt.design == "within" else cross(s, data, subs)
	null = np.array(null)
	sw = last_swap if opt.design == "cross" else (float("nan"), float("nan"))
	res[s] = {"r": real[0], "R2": real[1], "swap_r": sw[0], "swap_R2": sw[1], "alpha": float(alpha), "alpha_at_grid_edge": bool(alpha in (ALPHAS[0], ALPHAS[-1])), "rois_present": present[s], "block_r": block[0], "block_R2": block[1], "p_r": perm_p(real[0], null[:, 0]), "p_R2": perm_p(real[1], null[:, 1]),
			"null_r_mean": float(null[:, 0].mean()), "null_r_q95": float(np.quantile(null[:, 0], 0.95))}
	if(s in hr_qc):
		res[s]["hr_qc"] = hr_qc[s]
	print("I:", s, json.dumps(res[s]), flush=True)
k = int(sum(v["p_r"] < 0.05 for v in res.values()))
group = {"design": opt.design, "target": opt.target, "features": opt.features, "motion_regress": opt.motion_regress, "rois": rois, "roi_mode": opt.roi_mode, "excluded": opt.exclude, "lags": opt.lags, "n_surrogates": opt.n_surrogates,
		"mean_r": float(np.mean([v["r"] for v in res.values()])), "mean_R2": float(np.mean([v["R2"] for v in res.values()])),
		"block_mean_r": float(np.mean([v["block_r"] for v in res.values()])), "block_mean_R2": float(np.mean([v["block_R2"] for v in res.values()])),
		"swap_mean_r": float(np.nanmean([v["swap_r"] for v in res.values()])),
		"real_vs_swap_wilcoxon_p": float(__import__("scipy.stats", fromlist=["wilcoxon"]).wilcoxon([v["r"] for v in res.values()], [v["swap_r"] for v in res.values()], alternative="greater").pvalue) if opt.design == "cross" and len(res) >= 5 else None,
		#paired over subjects: EEG ridge vs the block-design baseline (raw target; on the resid target the block is already removed)
		"ridge_vs_block_wilcoxon_p": float(__import__("scipy.stats", fromlist=["wilcoxon"]).wilcoxon([v["r"] for v in res.values()], [v["block_r"] for v in res.values()], alternative="greater").pvalue) if len(res) >= 5 else None,
		"k_significant": k, "n": len(res), "binomial_p": float(binomtest(k, len(res), 0.05, alternative="greater").pvalue)}
print("GROUP", json.dumps(group), flush=True)
with open(os.path.join(opt.out, "ridge_%s_%s_%s_%s%s%s.json" % (opt.design, opt.target, opt.features, opt.roi_mode, "_motreg" if opt.motion_regress else "", "_qcexcl" if opt.exclude else "")), "w") as f:
	json.dump({"group": group, "subjects": res}, f, indent=1)
