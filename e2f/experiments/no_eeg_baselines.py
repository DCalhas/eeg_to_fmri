"""How much of published EEG-to-fMRI fidelity is achievable without EEG?

Uses the E2fNet NODDI protocol (their preprocessing, h5 files with fmri [N, 30, 64, 64] in [0,1] per volume)
and their metrics (torchmetrics SSIM, data_range=1, volumes as 30-channel 2D images; PSNR data_range=1).
For each held-out subject (LOSO):
  template   population mean volume of the training subjects, predicted for every test volume (no EEG, no test data)
  own_mean   the held-out subject's own temporal mean volume (no EEG; uses the test subject's anatomy: upper bound
             for any predictor that only gets the static image right)
"""
import os, sys, json, glob
import numpy as np
import h5py
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

H5 = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/eeg_to_fmri/runs/e2fnet_protocol/h5_noddi")
OUT = os.path.join(os.path.dirname(H5), "no_eeg_baselines.json")


def metrics(pred, target, bs=32):
	ssim, psnr = StructuralSimilarityIndexMeasure(data_range=1.0), PeakSignalNoiseRatio(data_range=1.0)
	s, p, n = 0.0, 0.0, 0
	for i in range(0, len(target), bs):
		t = torch.from_numpy(target[i:i + bs]).float()
		q = torch.from_numpy(np.broadcast_to(pred, t.shape).copy()).float() if pred.ndim == 3 else torch.from_numpy(pred[i:i + bs]).float()
		s += ssim(q, t).item() * len(t); p += psnr(q, t).item() * len(t); n += len(t)
	return s / n, p / n


if __name__ == "__main__":
	files = sorted(glob.glob(os.path.join(H5, "*.h5")))
	vols = {os.path.basename(f)[:-3]: np.array(h5py.File(f, "r")["fmri"], dtype=np.float32) for f in files}
	means = {k: v.mean(0) for k, v in vols.items()}
	res = {}
	for s in vols:
		template = np.mean([means[k] for k in vols if k != s], axis=0)
		res[s] = {"template": metrics(template, vols[s]), "own_mean": metrics(means[s], vols[s]), "n": len(vols[s])}
		print(s, res[s], flush=True)
	summ = {m: {"ssim_mean": float(np.mean([r[m][0] for r in res.values()])), "psnr_mean": float(np.mean([r[m][1] for r in res.values()]))}
			for m in ("template", "own_mean")}
	print(json.dumps(summ, indent=1))
	json.dump({"per_subject": res, "summary": summ}, open(OUT, "w"), indent=1)
