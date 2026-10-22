"""Fribourg Ultimatum Game schizophrenia cohort (OpenNeuro ds004000).

`legacy_stft` reproduces the 2023 feature pipeline (src/utils/data_utils.get_data_classification +
create_clf_pairs) so its numbers can be compared one to one; `channel_mode` exposes the channel
selection, because the 2023 loader kept the first 132 channels and the EXG (EOG) layout differs
by group in this release: controls have EXG1,2,7,8 and patients EXG1,2,3,4,7,8.
"""
import os
import numpy as np
import mne
from scipy.fftpack import fft
from scipy.stats import zscore

ROOT = os.path.expanduser("~/eeg_to_fmri/data/ds004000")
FS = 2048


def subjects(root=ROOT):
	rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(root, "participants.tsv"))][1:]
	return [r[0] for r in rows], np.array([1 if r[1].startswith("P") else 0 for r in rows])


def read(sub, task="responder", root=ROOT):
	return mne.io.read_raw_brainvision(os.path.join(root, sub, "eeg", f"{sub}_task-{task}_run-1_eeg.vhdr"),
									   preload=True, verbose=0)


def select_channels(raw, mode):
	"""legacy: first 132 channels in file order (group-dependent EXG content);
	eeg: the 128 EEG channels only; exg_legacy: positions 128..131 of the legacy selection."""
	data = raw.get_data()
	if mode == "legacy":
		return data[:132]
	eeg = [i for i, n in enumerate(raw.ch_names) if not n.startswith("EXG")]
	assert len(eeg) == 128, len(eeg)
	if mode == "eeg":
		return data[eeg]
	if mode == "exg_legacy":
		return data[:132][128:]
	raise ValueError(mode)


def stft_legacy(x, fs=FS, window=2, f_high=135):
	"""|FFT| of non-overlapping `window`-s rectangular windows, bins 1..f_high-1 (0.5 Hz steps).
	x: (C, N) -> (C, f_high-1, n_windows). Matches eeg_utils.stft(limit=True) incl. dropping the last window."""
	w = int(window * fs)
	starts = list(range(0, x.shape[1], w))[:-1]
	out = np.empty((x.shape[0], f_high - 1, len(starts)), dtype=np.float32)
	for j, s in enumerate(starts):
		seg = x[:, s:s + w]
		out[:, :, j] = np.abs(fft(seg, axis=1)[:, :f_high][:, 1:])
	return out


def legacy_instances(sub, channel_mode="legacy", recording_time=203, interval=10, task="responder"):
	"""Returns X (n_inst, C, 134, interval) for one subject, as create_clf_pairs (20 instances for 203 frames)."""
	raw = read(sub, task)
	Z = zscore(stft_legacy(select_channels(raw, channel_mode)), axis=0)  # z-score across channels, as in 2023
	Z = np.transpose(Z, (2, 0, 1))[:recording_time]  # (T, C, F)
	X = [np.transpose(Z[t:t + interval], (1, 2, 0)) for t in range(0, recording_time, interval) if t + interval < recording_time]
	return np.stack(X).astype(np.float32)
