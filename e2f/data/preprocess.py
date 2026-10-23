"""Common EEG preprocessing and montage harmonization.

All cohorts and the NODDI pretraining data go through the same steps:
  EEG channels picked by type -> 10-05 positions -> notch (line freq harmonics) -> 0.5-45 Hz band-pass ->
  average reference -> resample 250 Hz -> spherical-spline interpolation onto the 61-channel NODDI montage.
The NODDI montage is the BrainCap MR 64 minus ECG, O9 and O10 (not in standard_1005).
"""
import numpy as np
import mne

SFREQ = 250.0
NODDI_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz',
				  'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'AF3', 'AF4', 'FC3',
				  'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8',
				  'PO7', 'PO8', 'FT9', 'FT10', 'P9', 'P10', 'PO9', 'PO10', 'Fpz', 'CPz']
_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _std1005():
	return mne.channels.make_standard_montage("standard_1005")


def set_positions(raw, electrodes_tsv=None):
	"""Rename to canonical 10-05 case, drop EEG channels without a position, attach the montage.
	electrodes_tsv: BIDS electrodes file for caps not named in 10-05 (e.g. Biosemi 128)."""
	raw.pick("eeg")
	if electrodes_tsv is not None:
		import pandas as pd
		el = pd.read_csv(electrodes_tsv, sep="\t")
		pos = {r.name: np.array([r.x, r.y, r.z], float) for r in el.itertuples() if r.name in raw.ch_names}
		scale = 1e-3 if np.nanmax(np.abs(np.stack(list(pos.values())))) > 1 else 1.0  # mm -> m
		mont = mne.channels.make_dig_montage({k: v * scale for k, v in pos.items()}, coord_frame="head")
		raw.pick([c for c in raw.ch_names if c in pos])
		raw.set_montage(mont)
		return raw
	std = {c.lower(): c for c in _std1005().ch_names}
	ren = {}
	for c in raw.ch_names:
		k = _ALIASES.get(c.upper(), c)
		if k.lower() in std:
			ren[c] = std[k.lower()]
	raw.pick(list(ren))
	raw.rename_channels(ren)
	raw.set_montage(_std1005())
	return raw


def clean(raw, line_freq=50.0, l_freq=0.5, h_freq=45.0, sfreq=SFREQ):
	raw.load_data()
	if line_freq:
		freqs = np.arange(line_freq, min(raw.info["sfreq"] / 2, 3 * line_freq + 1), line_freq)
		if len(freqs):
			raw.notch_filter(freqs, verbose=0)
	raw.filter(l_freq, h_freq, verbose=0)
	raw.set_eeg_reference("average", verbose=0)
	if raw.info["sfreq"] != sfreq:
		raw.resample(sfreq, verbose=0)
	return raw


def to_noddi_montage(raw):
	"""Spherical-spline interpolation onto NODDI_CHANNELS (channels already present are copied)."""
	target = mne.channels.make_standard_montage("standard_1005")
	pos = target.get_positions()["ch_pos"]
	mont = mne.channels.make_dig_montage({c: pos[c] for c in NODDI_CHANNELS}, coord_frame="head")
	out = raw.copy().interpolate_to(mont, method="spline")
	return out.reorder_channels(NODDI_CHANNELS)


def harmonized(raw, line_freq, electrodes_tsv=None, crop=None):
	"""Full pipeline. crop: (tmin, tmax) seconds applied after cleaning."""
	raw = set_positions(raw, electrodes_tsv)
	raw = clean(raw, line_freq=line_freq)
	if crop is not None:
		raw.crop(*crop)
	return to_noddi_montage(raw)


def log_bandpower(x, sfreq=SFREQ, bands=((1, 4), (4, 8), (8, 13), (13, 30), (30, 45)), win_s=4.0, relative=True):
	"""Welch log band power per channel. x: (C, N) -> (C, n_bands)."""
	from scipy.signal import welch
	f, p = welch(x, fs=sfreq, nperseg=int(win_s * sfreq), axis=-1)
	bp = np.stack([p[:, (f >= lo) & (f < hi)].mean(-1) for lo, hi in bands], -1)
	if relative:
		bp = bp / p[:, (f >= bands[0][0]) & (f < bands[-1][1])].mean(-1, keepdims=True)
	return np.log(bp)
