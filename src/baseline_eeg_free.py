"""
EEG-free baselines on the same folds and evaluation as the models (audit A3):
	* eeg_free_template - voxelwise Normal(mean, std) over the training subjects' volumes (no EEG, no test data),
	                      i.e. fidelity_shared.gaussian_template, scored with NLL/CRPS/PICP/conformal like any model
	* oracle_own_mean   - Normal(mean, std) of the held-out subject's own volumes: uses the test subject's data, so it
	                      is an upper bound for any predictor that only gets the static image (and its spread) right
	* stimulus_locked   - (audit A17) for every volume index t within the run, Normal(mean, std) over the training
	                      subjects' volumes at t. ds002338 MIpost is a block design with identical timing in every
	                      subject, so on the dynamics target (-target fluct) this predicts the task-locked response
	                      without EEG; a model must beat it to claim EEG-specific information.
The first two predict the same volume for every TR, so their voxel temporal r is 0.

python baseline_eeg_free.py 05 -fold 0 -out_dir RUNS -cache_dir CACHE
"""
import argparse
import os

import numpy as np

import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
opt = parser.parse_args()

from utils import process_utils
#GPU memory limit, as train_cv.py: without it the first of several parallel processes takes the whole GPU
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=0)
import evaluate_uq
from utils import fidelity_shared

splits, test, calib = experiment.load_split(opt)


class Template:
	def __init__(self, mu, sigma):
		self.mu, self.sigma = mu[..., None].astype(np.float32), sigma[..., None].astype(np.float32)

	def __call__(self, eeg):
		b = eeg.shape[0]
		return {"family": "gaussian", "params": [np.broadcast_to(self.mu, (b,)+self.mu.shape), np.broadcast_to(self.sigma, (b,)+self.sigma.shape)]}


class OwnMean:
	"""Per-subject Normal(mean, std) of that subject's own volumes, for the calibration and test splits."""
	wants_index = True

	def __init__(self, splits):
		self.stats = {}
		for name in ["calib", "test"]:
			_, fmri, subjects = splits[name]
			per = {s: (fmri[subjects == s].mean(0), fmri[subjects == s].std(0)+1e-4) for s in np.unique(subjects)}
			self.stats[name] = (subjects, per)

	def __call__(self, eeg, split, start):
		subjects, per = self.stats[split]
		ids = subjects[start:start+eeg.shape[0]]
		return {"family": "gaussian", "params": [np.stack([per[s][0] for s in ids]), np.stack([per[s][1] for s in ids])]}


class StimulusLocked:
	"""Normal(mean, std) over training subjects of the volume at the same index t within the run (identical timing)."""
	wants_index = True

	def __init__(self, splits):
		fmri, subjects = splits["train"][1][..., 0], splits["train"][2]
		stack = np.stack([fmri[subjects == s] for s in np.unique(subjects)])  #(subjects, t, X, Y, Z)
		self.mu, self.sigma = stack.mean(0).astype(np.float32), (stack.std(0)+1e-4).astype(np.float32)
		self.t = {name: np.concatenate([np.arange((split[2] == s).sum()) for s in np.unique(split[2])]) for name, split in splits.items()
					if name in ("calib", "test") and split is not None}
		for name, split in splits.items():
			if(name in ("calib", "test") and split is not None):
				order = np.concatenate([np.where(split[2] == s)[0] for s in np.unique(split[2])])
				assert np.all(order == np.arange(len(order))), "instances must be grouped by subject in time order"

	def __call__(self, eeg, split, start):
		t = self.t[split][start:start+eeg.shape[0]]
		return {"family": "gaussian", "params": [self.mu[t][..., None], self.sigma[t][..., None]]}


fmri_train, sub_train = splits["train"][1][..., 0], splits["train"][2]
mu, var = fidelity_shared.gaussian_template([fmri_train[sub_train == s] for s in np.unique(sub_train)])
for name, predictor in [("eeg_free_template", Template(mu, np.sqrt(var))), ("oracle_own_mean", OwnMean(splits)), ("stimulus_locked", StimulusLocked(splits))]:
	opt.name, opt.seed = ("mni_" if opt.fmri_space == "mni" else "")+name+{"run_zscore": "", "fluct": "_fluct", "fluct_resid": "_fluct_resid"}[opt.target]+("_inject%g" % opt.inject_snr if opt.inject_snr > 0 else ""), 0
	out = experiment.run_dir(opt)
	os.makedirs(out, exist_ok=True)
	res = evaluate_uq.evaluate([predictor], splits, T=1, stochastic=False, seed=0)
	evaluate_uq.save(res, out)
	print("==", name); evaluate_uq.print_summary(res)
