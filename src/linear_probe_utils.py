"""Shared by linear_probe.py and ablation_ladder.py: input features and voxel temporal correlations."""
import numpy as np

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]  #STFT bins are 1..f Hz: bin index = Hz-1


def features(eeg, tr=1.0):
	"""(N, 63, 134, W, 1) z-scored STFT magnitude -> (N, 63*5*W) band means; STFT bin k is (k+1)/tr Hz."""
	x = eeg[..., 0]
	return np.concatenate([x[:, :, int(lo*tr)-1:int(hi*tr)-1].mean(2) for lo, hi in BANDS], 1).reshape(len(x), -1).astype(np.float32)


def voxel_r(P, Y):
	P = P-P.mean(0); Y = Y-Y.mean(0)
	r = (P*Y).sum(0)/np.sqrt((P**2).sum(0)*(Y**2).sum(0)+1e-12)
	r[(Y**2).sum(0) <= 1e-12] = np.nan
	return r
