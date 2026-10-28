"""Second synthesis training set: XP1 (OpenNeuro ds002336, Lioi et al. 2020), simultaneous EEG-fMRI, 10 subjects, 3 T,
TR 2 s, 32 slices of 4 mm, 64-channel BrainCap MR (the 57 NODDI channels are all present, by name), four runs per
subject (eegNF, eegfmriNF, fmriNF, motorloc). Data are shared read-only (downloaded by the uncertainty session).

Alignment. The released gradient- and pulse-corrected EEG (derivatives/<sub>/eeg_pp, 200 Hz) is cut to the task
protocol and keeps one `R128` marker per volume for that span only; the raw 5 kHz EEG keeps every scanner trigger
(e.g. 206 triggers for 205 volumes). Each corrected-EEG trigger is matched to a raw trigger by time, anchored on the
first task-start marker (`S  2`, present in both files). The raw trigger count does not always equal the number of
BOLD volumes (`trigger_audit`: one extra or up to six missing triggers in 15 of 35 runs; the scanner is already
acquiring before the first trigger, so the gradient artefact cannot disambiguate), so only runs with exactly one raw
trigger per volume are used, and raw trigger k is volume k.
Cache per run (runs/xp1_aligned/<sub>_<task>.npz):
  eeg     (57, N) float32, NODDI channels by name; 0.5-45 Hz, notch 50 Hz, average reference, 250 Hz
  onsets  (V,) seconds of each kept volume's trigger in the EEG time base
  vol     (V,) volume indices into the BOLD run
  fmri    (V, 64, 64, 30) float32, raw EPI intensities resampled from 106x106x32 (1.98x1.98x4 mm) to the NODDI grid
"""
import os, glob, re, datetime
import numpy as np
import mne
import nibabel as nib

from e2f.data import preprocess as P

ROOT = os.path.expanduser("~/eeg_to_fmri/data/ds002336")
OUT = os.path.expanduser("~/eeg_to_fmri/runs/xp1_aligned")
TASKS = ["eegNF", "eegfmriNF", "fmriNF", "motorloc"]
TR = 2.0
GRID = (64, 64, 30)


def subjects():
	return sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "sub-xp1*")))


def has_run(sub, task):
	return all(os.path.exists(x) for x in _paths(sub, task))


def _paths(sub, task):
	pp = os.path.join(ROOT, "derivatives", sub, "eeg_pp", f"{sub}_task-{task}_eeg_pp.vhdr")
	raw = os.path.join(ROOT, sub, "eeg", f"{sub}_task-{task}_eeg.vhdr")
	bold = os.path.join(ROOT, sub, "func", f"{sub}_task-{task}_bold.nii.gz")
	return pp, raw, bold


def _markers(vmrk, stim=None):
	"""R128 sample positions (1-based in the file -> 0-based) and the first New Segment date; with `stim`, the
	positions of that stimulus description instead."""
	pos, date = [], None
	for line in open(vmrk, encoding="utf-8", errors="replace"):
		m = re.match(r"Mk\d+=([^,]*),([^,]*),(\d+),", line)
		if not m:
			continue
		typ, desc, p = m.group(1), m.group(2).strip(), int(m.group(3))
		if (stim is None and desc == "R128") or (stim is not None and desc == stim):
			pos.append(p - 1)
		if typ == "New Segment" and date is None:
			d = line.strip().split(",")[-1]
			if re.fullmatch(r"\d{20}", d):
				date = datetime.datetime.strptime(d, "%Y%m%d%H%M%S%f")
	return np.array(pos), date


def _sfreq(vhdr):
	for line in open(vhdr, encoding="utf-8", errors="replace"):
		if line.startswith("SamplingInterval="):
			return 1e6 / float(line.split("=")[1])


def trigger_audit(sub, task, win_s=0.5):
	"""Gradient-artefact power (>100 Hz RMS over all channels) in the `win_s` after the first two and last two raw
	triggers, plus the number of raw triggers and BOLD volumes. Run on CPU nodes (reads the 5 kHz raw EEG)."""
	_, raw_v, bold = _paths(sub, task)
	trig, _ = _markers(raw_v[:-5] + ".vmrk")
	raw = mne.io.read_raw_brainvision(raw_v, preload=False, verbose="ERROR")
	sf = raw.info["sfreq"]
	n = int(win_s * sf)
	def hf(p0):
		x = raw.get_data(start=p0, stop=min(p0 + n, raw.n_times))
		x = x - x.mean(1, keepdims=True)
		X = np.abs(np.fft.rfft(x, axis=1)); f = np.fft.rfftfreq(x.shape[1], 1 / sf)
		return float(np.sqrt((X[:, f > 100] ** 2).mean()))
	before_first = hf(max(trig[0] - n - 10, 0))
	after = [hf(p) for p in (trig[0], trig[1], trig[-2], trig[-1])]
	return {"sub": sub, "task": task, "n_trig": len(trig), "n_vol": int(nib.load(bold).shape[-1]),
			"hf_before_first": before_first, "hf_after_first": after[0], "hf_after_second": after[1],
			"hf_after_penultimate": after[2], "hf_after_last": after[3]}


def _pp_to_raw_index(sub, task):
	"""Index of each corrected-EEG trigger among the raw triggers, by time, anchored on the first stimulus marker kind present in both files
	(task start `S  2`, else `S  1`, else `S 99`) and validated on all stimulus markers."""
	pp_v, raw_v, _ = _paths(sub, task)
	pp_pos, _ = _markers(pp_v[:-5] + ".vmrk")
	raw_pos, _ = _markers(raw_v[:-5] + ".vmrk")
	sf_pp, sf_raw = _sfreq(pp_v), _sfreq(raw_v)
	anchor = next(d for d in ("S  2", "S  1", "S 99") if len(_markers(pp_v[:-5] + ".vmrk", stim=d)[0]) and len(_markers(raw_v[:-5] + ".vmrk", stim=d)[0]))
	off = _markers(raw_v[:-5] + ".vmrk", stim=anchor)[0][0] / sf_raw - _markers(pp_v[:-5] + ".vmrk", stim=anchor)[0][0] / sf_pp
	for d in ("S  1", "S  2", "S 99"):  # every stimulus marker of the corrected EEG must land on a raw marker of the same kind
		pp_d, raw_d = _markers(pp_v[:-5] + ".vmrk", stim=d)[0] / sf_pp + off, _markers(raw_v[:-5] + ".vmrk", stim=d)[0] / sf_raw
		if len(pp_d):
			assert len(raw_d) and np.abs(raw_d[None] - pp_d[:, None]).min(1).max() < 0.05, (sub, task, d, "stimulus markers do not match")
	t_pp = pp_pos / sf_pp + off
	t_raw = raw_pos / sf_raw
	idx = np.array([int(np.argmin(np.abs(t_raw - t))) for t in t_pp])
	err = np.abs(t_raw[idx] - t_pp)
	assert err.max() < 0.05, (sub, task, float(err.max()))
	assert np.all(np.diff(idx) == 1), (sub, task, "corrected-EEG triggers are not consecutive raw triggers")
	return pp_pos, sf_pp, idx, len(raw_pos)


def build(sub, task, force=False):
	f = os.path.join(OUT, f"{sub}_{task}.npz")
	if os.path.exists(f) and not force:
		return f
	pp_v, raw_v, bold = _paths(sub, task)
	if not has_run(sub, task):
		return None
	mne.set_log_level("ERROR")
	try:
		pp_pos, sf_pp, vol, n_trig = _pp_to_raw_index(sub, task)
	except AssertionError as e:
		print("excluded (alignment validation failed)", sub, task, e, flush=True)
		return None
	img = nib.load(bold)
	if n_trig != img.shape[-1]:
		print("excluded (triggers != volumes)", sub, task, n_trig, img.shape[-1], flush=True)
		return None  # ambiguous trigger-to-volume mapping: run excluded
	keep = (vol >= 0) & (vol < img.shape[-1])
	raw = mne.io.read_raw_brainvision(pp_v, preload=True)
	onsets = pp_pos[keep] / sf_pp
	raw = P.set_positions(raw)  # EEG channels with 10-05 positions (drops ECG)
	raw = P.clean(raw, line_freq=50.0)  # average reference over all recorded EEG channels, as for NODDI
	assert set(P.NODDI_CHANNELS) <= set(raw.ch_names), sorted(set(P.NODDI_CHANNELS) - set(raw.ch_names))
	raw.pick(P.NODDI_CHANNELS)
	from scipy.ndimage import zoom
	data = np.asarray(img.dataobj, dtype=np.float32)
	zf = [g / s for g, s in zip(GRID, data.shape[:3])]
	fmri = np.stack([zoom(data[..., v], zf, order=1) for v in vol[keep]]).astype(np.float32)
	os.makedirs(OUT, exist_ok=True)
	tmp = f + ".tmp.npz"
	np.savez(tmp, eeg=raw.get_data().astype(np.float32), onsets=onsets, vol=vol[keep], fmri=fmri, sfreq=raw.info["sfreq"],
			 ch=np.array(raw.ch_names), zoom=np.array(zf))
	os.replace(tmp, f)
	return f


def load_subject(sub, n_frames=20):
	"""All runs of a subject as synthesizer pairs: X (V, frames, C, F) log-spectral frames of 2.16 s segments (the
	NODDI/clinical featurisation, so clinical inputs are unchanged), Y (V, 30, 64, 64) volumes min-max scaled to [0,1]."""
	from e2f.data import noddi
	Xs, Ys = [], []
	for task in TASKS:
		f = os.path.join(OUT, f"{sub}_{task}.npz")
		if not os.path.exists(f):
			continue
		d = np.load(f)
		X, idx, _ = noddi.tr_frames(d["eeg"], d["onsets"], n_frames=n_frames)
		Y = d["fmri"][idx]
		lo, hi = Y.min(axis=(1, 2, 3), keepdims=True), Y.max(axis=(1, 2, 3), keepdims=True)
		Xs.append(np.transpose(X, (0, 3, 1, 2))); Ys.append(np.transpose((Y - lo) / (hi - lo + 1e-10), (0, 3, 1, 2)).astype(np.float32))
	return np.concatenate(Xs), np.concatenate(Ys)


if __name__ == "__main__":
	import json, argparse
	from joblib import Parallel, delayed
	ap = argparse.ArgumentParser()
	ap.add_argument("--audit", action="store_true")
	ap.add_argument("--build", action="store_true")
	a = ap.parse_args()
	if a.audit:
		res = Parallel(n_jobs=8)(delayed(trigger_audit)(s, t) for s in subjects() for t in TASKS if has_run(s, t))
		os.makedirs(OUT, exist_ok=True)
		json.dump(res, open(os.path.join(OUT, "trigger_audit.json"), "w"), indent=1)
		for r in res:
			print(json.dumps(r), flush=True)
		for s in subjects():
			for t in TASKS:
				if has_run(s, t):
					try:
						_, _, idx, n_trig = _pp_to_raw_index(s, t)
						print("pp->raw", s, t, "first raw index", int(idx[0]), "n", len(idx), "n_trig", n_trig, flush=True)
					except AssertionError as e:
						print("pp->raw FAILED", s, t, e, flush=True)
	if a.build:
		print(Parallel(n_jobs=8)(delayed(build)(s, t) for s in subjects() for t in TASKS), flush=True)
