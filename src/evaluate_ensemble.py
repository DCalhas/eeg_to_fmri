"""
Deep ensemble (Lakshminarayanan et al. 2017) from already trained seeds of one configuration:
the predictive distribution is the mixture of the members' Laplace(mu_m, b_m).

python evaluate_ensemble.py 05 -name calhas_aleatoric -fold 0 -members 2,3,5,7,11 -out_dir RUNS <same model flags as training>
"""
import argparse
import os

import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
parser.add_argument('-members', required=True, type=str, help="Comma separated seeds of the trained members")
parser.add_argument('-T', default=50, type=int, help="Total Monte Carlo samples (split across members if they are stochastic)")
opt = parser.parse_args()

from utils import process_utils
import tensorflow as tf
import evaluate_uq

members = list(map(int, opt.members.split(",")))
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=members[0])
splits, test, calib = experiment.load_split(opt)
eeg_train, fmri_train, _ = splits["train"]
eeg_shape, fmri_shape = (None,)+eeg_train.shape[1:], (None,)+fmri_train.shape[1:]

models = []
for seed in members:
	opt.seed = seed
	with tf.device('/CPU:0'):
		model, _ = experiment.build_model(opt, eeg_shape, fmri_shape, n_train=eeg_train.shape[0])
	model.load_weights(os.path.join(experiment.run_dir(opt), "weights")).expect_partial()
	models.append(model)

stochastic = experiment.is_stochastic(opt)
res = evaluate_uq.evaluate(models, splits, T=opt.T if stochastic else len(models), stochastic=True, seed=members[0])
opt.name, opt.seed = opt.name+"_ensemble", 0
out = experiment.run_dir(opt)
os.makedirs(out, exist_ok=True)
evaluate_uq.save(res, out)
evaluate_uq.print_summary(res)
#evaluation-time EEG nulls, as for every single model (train_cv.py)
for null in ["shift", "swap"]:
	res = evaluate_uq.evaluate(models, splits, T=max(len(models), 20) if stochastic else len(models), stochastic=True, seed=members[0], null=null)
	evaluate_uq.save(res, out, tag="_null_"+null)
