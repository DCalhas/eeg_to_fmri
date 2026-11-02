"""
EEG icons of the introduction illustration (figures/original/idea_description.tex), as savefig PDFs: 10 s of the
preprocessed CHUR-XP2 EEG export (sub-xp201, MIpost), 20 channels, and the STFT magnitude of C3 over the same 10 s with
the models' 1-s windows. Replaces the raster eeg.png / stft.png.

CPU-only: run as a CPU Slurm job (-p amd).
python experiments/intro_icons_eeg.py -out ~/eeg_to_fmri/uncertainty/icons
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import stft

parser = argparse.ArgumentParser()
parser.add_argument('-subject', default="sub-xp201")
parser.add_argument('-out', required=True)
opt = parser.parse_args()
os.makedirs(opt.out, exist_ok=True)

root = os.path.expanduser("~/eeg_to_fmri/data/ds002338/derivatives")
vhdr = sorted(glob.glob(os.path.join(root, "**", "*"+opt.subject+"*MIpost*.vhdr"), recursive=True))
assert vhdr, "no MIpost EEG export for "+opt.subject
raw = mne.io.read_raw_brainvision(vhdr[0], preload=True, verbose="ERROR")
raw.pick([c for c in raw.ch_names if c.upper() != "ECG"])
fs = raw.info["sfreq"]; start = int(60*fs); n = int(10*fs)
x = raw.get_data()[:, start:start+n]

fig, ax = plt.subplots(figsize=(3, 2.3))
chans = np.linspace(0, x.shape[0]-1, 20).astype(int)
for i, c in enumerate(chans):
	s = x[c]-x[c].mean(); ax.plot(s/(4*s.std()+1e-12)+i, color="k", lw=0.4)
ax.axis("off"); fig.savefig(os.path.join(opt.out, "eeg.pdf"), format="pdf", bbox_inches="tight", pad_inches=0)

c3 = raw.ch_names.index("C3") if "C3" in raw.ch_names else 0
f, t, Z = stft(x[c3], fs=fs, nperseg=int(fs), noverlap=0)
fig, ax = plt.subplots(figsize=(3, 2))
ax.imshow(np.log(np.abs(Z[(f >= 1) & (f <= 50)])+1e-12), aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
ax.axis("off"); fig.savefig(os.path.join(opt.out, "stft.pdf"), format="pdf", bbox_inches="tight", pad_inches=0)
print("I: icons from", vhdr[0])
