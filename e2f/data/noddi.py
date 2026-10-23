"""Trigger-aligned NODDI EEG-fMRI (OSF 94c5t, v2: 16 subjects).

Fixes ledger A7: the BVA exports start 13-130 s before the first fMRI volume in 15/16 subjects, so EEG must be
indexed by the `Scanner/Scan Start` markers (one per volume, TR = 2.16 s), never from the file start.
Cache per subject (runs/noddi_aligned/<sub>.npz):
  eeg     (61, N) float32, NODDI montage minus ECG/O9/O10, 0.5-45 Hz, notch 50 Hz, average reference, 250 Hz
  ecg     (1, N) float32, the ECG channel (same filtering), for the ECG-only control arm only
  onsets  (300,)  seconds of each volume's Scan Start marker in the EEG time base
  fmri    (300, 64, 64, 30) float32, raw EPI intensities (no motion correction; ledger A10)
"""
import os, glob
import numpy as np
import mne
import nibabel as nib

from e2f.data import preprocess as P

ROOT = os.path.expanduser("~/eeg_to_fmri/data/noddi")
OUT = os.path.expanduser("~/eeg_to_fmri/runs/noddi_aligned")
TR = 2.16


def subjects():
	return sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "fMRI", "*")))


def _vhdr(sub):
	v = glob.glob(os.path.join(ROOT, "EEG*", sub, "export", "*_bin.vhdr"))
	assert len(v) == 1, (sub, v)
	return v[0]


def build(sub, force=False):
	f = os.path.join(OUT, f"{sub}.npz")
	if os.path.exists(f) and not force:
		return f
	mne.set_log_level("ERROR")
	raw = mne.io.read_raw_brainvision(_vhdr(sub), preload=True)
	onsets = np.array([o for o, d in zip(raw.annotations.onset, raw.annotations.description) if d == "Scanner/Scan Start"])
	img = nib.load(glob.glob(os.path.join(ROOT, "fMRI", sub, "*nw_mepi_rest_with_cross.nii.gz"))[0])
	fmri = np.moveaxis(np.asarray(img.dataobj, dtype=np.float32), -1, 0)
	assert len(onsets) == fmri.shape[0] == 300, (sub, len(onsets), fmri.shape)
	assert np.allclose(np.diff(onsets), TR, atol=0.01), (sub, np.diff(onsets).min(), np.diff(onsets).max())
	raw.set_channel_types({c: "ecg" for c in raw.ch_names if c.upper() in ("ECG", "EKG")})
	ecg = raw.copy().pick("ecg").load_data().filter(0.5, 45.0, verbose=0).resample(P.SFREQ, verbose=0).get_data().astype(np.float32)
	assert ecg.shape[0] == 1, ecg.shape  # ledger A1: kept separately for the ECG-only control arm, never in the EEG input
	raw = P.set_positions(raw)  # picks EEG (drops ECG), keeps the 61 channels named in 10-05 (drops O9/O10)
	raw = P.clean(raw, line_freq=50.0)
	assert raw.ch_names == [c for c in raw.ch_names if c in P.NODDI_CHANNELS] and len(raw.ch_names) == 61, raw.ch_names
	raw.reorder_channels(P.NODDI_CHANNELS)
	os.makedirs(OUT, exist_ok=True)
	eeg = raw.get_data().astype(np.float32)
	assert ecg.shape[1] == eeg.shape[1], (ecg.shape, eeg.shape)
	np.savez(f, eeg=eeg, ecg=ecg, onsets=onsets, fmri=fmri, sfreq=raw.info["sfreq"],
			 affine=img.affine, ch=np.array(raw.ch_names))
	return f


def load(sub):
	d = np.load(build(sub))
	return {k: d[k] for k in d.files}


def tr_frames(eeg, onsets, sfreq=P.SFREQ, n_frames=20, lag=0, fmax=45.0, log=True):
	"""Spectral frames aligned to volumes. For volume k, frames j = 0..n_frames-1 are the |FFT| of the TR-long
	EEG segments ending `lag` TRs before onset_k + TR, i.e. segment j covers
	[onset_k - (n_frames - 1 - j + lag) * TR, ... + TR). Volumes whose earliest frame precedes the EEG are dropped.
	Returns X (V, C, F, n_frames), valid volume indices, freqs."""
	n = int(round(TR * sfreq))
	freqs = np.fft.rfftfreq(n, 1 / sfreq)
	keep = (freqs > 0) & (freqs <= fmax)
	starts_all = np.round(onsets * sfreq).astype(int)
	X, idx = [], []
	for k, s in enumerate(starts_all):
		starts = s - np.arange(n_frames - 1 + lag, lag - 1, -1) * n
		if starts[0] < 0 or starts[-1] + n > eeg.shape[1]:
			continue
		seg = np.stack([eeg[:, a:a + n] for a in starts], -1)  # (C, n, frames)
		spec = np.abs(np.fft.rfft(seg, axis=1))[:, keep]
		X.append(np.log(spec + 1e-12) if log else spec); idx.append(k)
	return np.stack(X).astype(np.float32), np.array(idx), freqs[keep]


if __name__ == "__main__":
	from joblib import Parallel, delayed
	print(Parallel(n_jobs=8)(delayed(build)(s) for s in subjects()))
