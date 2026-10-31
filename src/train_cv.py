"""
Train one configuration on one subject-level fold and seed, save weights, evaluate.

python train_cv.py 05 -name calhas_spectral -fold 0 -seed 2 -out_dir RUNS <model flags as in main.py>
"""
import argparse
import json
import os
import time

import numpy as np

import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
parser.add_argument('-T', default=50, type=int, help="Monte Carlo samples at evaluation")
parser.add_argument('-skip_eval', action="store_true")
parser.add_argument('-init_from', default=None, type=str, help="Load the weights of this configuration (same fold and seed) instead of training")
parser.add_argument('-swag_epochs', default=0, type=int, help="SWAG-diagonal epochs after training (0 disables)")
parser.add_argument('-swag_lr', default=1e-3, type=float)
parser.add_argument('-laplace', action="store_true", help="Last-layer diagonal Laplace after training")
parser.add_argument('-nulls', default="shift,swap", type=str, help="Evaluation-time EEG nulls (audit A9), comma separated; empty disables")
parser.add_argument('-T_null', default=20, type=int)
parser.add_argument('-debug_nan', action="store_true", help="check every training step and stop with a report at the first non-finite loss")
parser.add_argument('-debug_persistent', action="store_true", help="with -debug_nan: use a persistent GradientTape, as utils.train.apply_gradient")
opt = parser.parse_args()

from utils import process_utils, train
from regularizers import path_sgd
import tensorflow as tf
import evaluate_uq

out=experiment.run_dir(opt)
os.makedirs(out, exist_ok=True)
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=opt.seed)

splits, test, calib = experiment.load_split(opt)
eeg_train, fmri_train, _ = splits["train"]
print("I: train subjects", sorted(set(splits["train"][2].tolist())), "calib", calib, "test", test, flush=True)

with tf.device('/CPU:0'):
	eeg_shape, fmri_shape = (None,)+eeg_train.shape[1:], (None,)+fmri_train.shape[1:]
	#shuffled across subjects every epoch (the original code used .shuffle(1), i.e. subjects in order, which leaves the last iterate and the
	#BatchNorm statistics fitted to the last training subject)
	train_set = tf.data.Dataset.from_tensor_slices((eeg_train, fmri_train)).shuffle(eeg_train.shape[0], seed=opt.seed, reshuffle_each_iteration=True).batch(opt.batch_size)
	model, learning_rate = experiment.build_model(opt, eeg_shape, fmri_shape, n_train=eeg_train.shape[0])
	optimizer = path_sgd.optimizer("Adam", [(1,)+eeg_shape[1:], (1,)+fmri_shape[1:]], model, learning_rate)
	model.compile(optimizer=optimizer)
	loss_fn = experiment.loss_fn(opt)

if(opt.debug_nan):
	#inspect the first update of the real training path: weights before, loss, gradients (Inf vs NaN, which variables)
	batch0 = next(iter(train_set))
	fin = lambda t: bool(np.all(np.isfinite(t.numpy())))
	print("DBG weights finite before any update", all(fin(v) for v in model.trainable_variables), "n_vars", len(model.trainable_variables), flush=True)
	with tf.GradientTape() as tape:
		logits = model(*batch0, training=True)
		reg = tf.math.add_n(model.losses) if len(model.losses) else 0.
		loss0 = loss_fn(batch0[1], logits)+reg
	print("DBG step-0 loss finite", fin(loss0), "n model.losses", len(model.losses), "reg", float(tf.reduce_sum(reg)), flush=True)
	grads0 = tape.gradient(loss0, model.trainable_variables)
	for v, g in zip(model.trainable_variables, grads0):
		if(g is None):
			continue
		gn = g.numpy()
		if(not np.all(np.isfinite(gn))):
			print("DBG non-finite grad", v.name, tuple(v.shape), "nan", int(np.isnan(gn).sum()), "inf", int(np.isinf(gn).sum()), flush=True)
	print("DBG max |grad|", max(float(tf.reduce_max(tf.abs(g))) for g in grads0 if g is not None and fin(g)), flush=True)
	print("DBG duplicate variables in trainable_variables", len(model.trainable_variables)-len({id(v) for v in model.trainable_variables}), flush=True)
t0=time.time()
extra = {}

#compiled calls for the learning curves: eager calls of Flipout layers leak GPU memory (~20 MB per call, ~750 MB per
#epoch over the calibration set), so the validation pass uses the same compiled functions as the evaluation
model_call = tf.function(lambda e, f: model(e, f), reduce_retracing=True)

def mean_loss(eeg, fmri, batch=16):
	#loss on held-out subjects (calibration split, never test) for the learning curves
	total, n = 0., 0
	for i in range(0, eeg.shape[0], batch):
		total += float(tf.reduce_sum(loss_fn(fmri[i:i+batch], model_call(tf.constant(eeg[i:i+batch]), tf.constant(fmri[i:i+batch]))))); n += eeg[i:i+batch].shape[0]
	return total/n

#Flipout layers leak GPU memory on every eager call (~20-160 MB, see evaluate_uq._compiled_decoder); the training step is
#compiled for them (OOM at epoch ~8 otherwise). Other configurations keep the eager step they were run with.
if(opt.bnn_layer == "flipout"):
	step = tf.function(lambda batch: train.train_step(model, batch, optimizer, loss_fn, u_architecture=True))
else:
	step = lambda batch: train.train_step(model, batch, optimizer, loss_fn, u_architecture=True)
if(opt.debug_nan):
	def step(batch):
		#same update as utils.train.apply_gradient, with the gradients checked before they are applied
		with tf.GradientTape(persistent=opt.debug_persistent) as tape:
			logits = model(*batch, training=True)
			loss = loss_fn(batch[1], logits)+(tf.math.add_n(model.losses) if len(model.losses) else 0.)
		grads = tape.gradient(loss, model.trainable_variables)
		bad = [(v.name, int(np.isnan(g.numpy()).sum()), int(np.isinf(g.numpy()).sum())) for v, g in zip(model.trainable_variables, grads) if g is not None and not np.all(np.isfinite(g.numpy()))]
		if(bad):
			e = batch[0].numpy()
			print("DBG first non-finite gradients at this batch: loss finite", bool(np.all(np.isfinite(loss.numpy()))), "eeg absmax %.3g eeg per-instance absmax" % np.abs(e).max(), np.round(np.abs(e).reshape(e.shape[0], -1).max(1), 2).tolist(), flush=True)
			for b in bad[:12]:
				print("DBG   ", b, flush=True)
			layer = [l for l in model.decoder.layers if "variational_i_dct3d" in l.name]
			if(layer):
				sc = layer[0].scale.numpy(); print("DBG conc min %.3g n_zero %d" % (sc.min(), int((sc == 0).sum())), flush=True)
			out = logits[0]
			print("DBG b range", [float(tf.reduce_min(out[1])), float(tf.reduce_max(out[1]))], "mu absmax %.3g" % float(tf.reduce_max(tf.abs(out[0]))), flush=True)
			raise SystemExit("debug_nan: stopped at the first non-finite gradient")
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		return tf.reduce_mean(loss)

if(opt.init_from is None):
	curves = {"train": [], "val": [], "val_rmse": []}
	for epoch in range(opt.epochs):
		train_loss = []
		for k, batch in enumerate(train_set):
			train_loss.append(float(step(batch)))
			if(opt.debug_nan and not np.isfinite(train_loss[-1])):
				layer = [l for l in model.decoder.layers if "variational_i_dct3d" in l.name]
				print("NAN epoch", epoch, "step", k, "loss", train_loss[-1], "eeg absmax", float(tf.reduce_max(tf.abs(batch[0]))), "fmri absmax", float(tf.reduce_max(tf.abs(batch[1]))),
					"eeg finite", bool(np.all(np.isfinite(batch[0].numpy()))), "weights finite", all(np.all(np.isfinite(v.numpy())) for v in model.trainable_variables), flush=True)
				if(layer):
					sc = layer[0].scale.numpy(); print("NAN conc min %.3g n_zero %d loc absmax %.3g biases absmax %.3g" % (sc.min(), int((sc == 0).sum()), float(tf.reduce_max(tf.abs(layer[0].loc))), float(tf.reduce_max(tf.abs(layer[0].biases)))), flush=True)
				out = model.decoder(batch[0], training=False)
				print("NAN decoder output finite", [bool(np.all(np.isfinite(o.numpy()))) for o in (out if isinstance(out, (list, tuple)) else [out])], "loss terms finite", bool(np.all(np.isfinite(loss_fn(batch[1], model(*batch, training=True)).numpy()))), flush=True)
				bad = [v.name for v in model.trainable_variables if not np.all(np.isfinite(v.numpy()))]
				print("NAN non-finite weights", bad[:10], flush=True)
				raise SystemExit("debug_nan: stopped at the first non-finite loss")
			if(opt.debug_nan and k % 100 == 0):
				print("I: step", k, "loss", train_loss[-1], flush=True)
		eeg_cal, fmri_cal, _ = splits["calib"]
		out_cal = [evaluate_uq._compiled_decoder(model)(tf.constant(eeg_cal[i:i+16])) for i in range(0, eeg_cal.shape[0], 16)]
		mu_cal = np.concatenate([(o[0] if isinstance(o, (list, tuple)) else o).numpy() for o in out_cal])
		#a run with a non-finite training loss is a failure, never an evaluation (the spectral NaN went unnoticed before)
		assert np.all(np.isfinite(train_loss)), "non-finite training loss in epoch %d (first at step %d)" % (epoch, int(np.argmin(np.isfinite(train_loss))))
		curves["train"].append(float(np.mean(train_loss))); curves["val"].append(mean_loss(eeg_cal, fmri_cal))
		curves["val_rmse"].append(float(np.sqrt(np.mean((mu_cal-fmri_cal)**2))))
		print("I: epoch", epoch, {k: v[-1] for k, v in curves.items()}, "GPU current MB", round(tf.config.experimental.get_memory_info("GPU:0")["current"]/2**20), flush=True)
		with open(os.path.join(out, "curves.json"), "w") as f:
			json.dump(curves, f)
	model.save_weights(os.path.join(out, "weights"))
	extra["gpu_peak_mb_training"] = tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**20
	print("I: GPU peak memory during training (MB)", extra["gpu_peak_mb_training"], flush=True)
else:
	model.load_weights(os.path.join(experiment.run_dir(opt, opt.init_from), "weights")).expect_partial()

predictors, stochastic = [model], experiment.is_stochastic(opt)
if(opt.swag_epochs > 0):
	import posthoc
	predictors, stochastic = [posthoc.swag(model, train_set, optimizer, loss_fn, epochs=opt.swag_epochs, learning_rate=opt.swag_lr, seed=opt.seed, bn_eeg=eeg_train)], True
if(opt.laplace):
	import posthoc
	sampler, extra["laplace_prior_precision"] = posthoc.last_layer_laplace(model, splits["train"][:2], splits["calib"][:2], seed=opt.seed)
	predictors, stochastic = [sampler], True
with open(os.path.join(out, "config.json"), "w") as f:
	json.dump(dict(vars(opt), test_subjects=test, calib_subjects=calib, train_seconds=time.time()-t0, **extra), f, indent=1)

if(not opt.skip_eval):
	res=evaluate_uq.evaluate(predictors, splits, T=opt.T, stochastic=stochastic, seed=opt.seed)
	evaluate_uq.save(res, out)
	evaluate_uq.print_summary(res)
	for null in [n for n in opt.nulls.split(",") if n]:
		if(null == "swap" and len(np.unique(splits["test"][2])) < 2):
			print("I: null swap skipped (one test subject: no other subject's EEG to swap in)", flush=True); continue
		res=evaluate_uq.evaluate(predictors, splits, T=opt.T_null, stochastic=stochastic, seed=opt.seed, null=null)
		evaluate_uq.save(res, out, tag="_null_"+null)
		print("I: null", null); evaluate_uq.print_summary(res)
	print("I: GPU peak memory overall (MB)", tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**20, flush=True)
