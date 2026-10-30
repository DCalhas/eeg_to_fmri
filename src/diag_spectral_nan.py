"""Diagnostic: where the stochastic spectral decoder becomes NaN (all spectral grid runs have NaN loss from epoch 0)."""
import argparse
import numpy as np
import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
opt = parser.parse_args()
from utils import process_utils, train
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=opt.seed)
import tensorflow as tf
from regularizers import path_sgd

splits, _, _ = experiment.load_split(opt)
eeg, fmri, _ = splits["train"]
model, lr = experiment.build_model(opt, (None,)+eeg.shape[1:], (None,)+fmri.shape[1:], n_train=eeg.shape[0])
optimizer = path_sgd.optimizer("Adam", None, model, lr)
loss_fn = experiment.loss_fn(opt)
fin = lambda t: bool(np.all(np.isfinite(t.numpy())))
layer = [l for l in model.decoder.layers if "variational_i_dct3d" in l.name][0]
print("spectral layer", layer.name, "loc", fin(layer.loc), "scale min", float(tf.reduce_min(layer.scale)), "biases", fin(layer.biases), flush=True)

x, y = tf.constant(eeg[:4]), tf.constant(fmri[:4])
out = model.decoder(x, training=False)
outs = out if isinstance(out, (list, tuple)) else [out]
print("forward before training: finite", [fin(o) for o in outs], "max |mu|", float(tf.reduce_max(tf.abs(outs[0]))), "b range", [float(tf.reduce_min(outs[1])), float(tf.reduce_max(outs[1]))] if len(outs) > 1 else None, flush=True)
#which intermediate first becomes non-finite
probe = tf.keras.Model(model.decoder.inputs, [l.output for l in model.decoder.layers[1:]])
for l, t in zip(model.decoder.layers[1:], probe(x)):
	t = t[0] if isinstance(t, (list, tuple)) else t
	if(not fin(t)):
		print("first non-finite layer output:", l.name, type(l).__name__, flush=True); break
else:
	print("all intermediate outputs finite", flush=True)

for step in range(30):
	i = (step*4) % (eeg.shape[0]-4)
	batch = (tf.constant(eeg[i:i+4]), tf.constant(fmri[i:i+4]))
	with tf.GradientTape() as tape:
		logits = model(*batch, training=True)
		loss = loss_fn(batch[1], logits)+(tf.add_n(model.losses) if model.losses else 0.)
	grads = tape.gradient(loss, model.trainable_variables)
	bad = [v.name for v, g in zip(model.trainable_variables, grads) if g is not None and not fin(g)]
	print("step", step, "loss", np.round(loss.numpy(), 3).tolist()[:4], "nonfinite grads", bad[:6], flush=True)
	if(bad or not fin(loss)):
		print("scale min", float(tf.reduce_min(layer.scale)), "loc absmax", float(tf.reduce_max(tf.abs(layer.loc))), flush=True)
		break
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	if(not all(fin(v) for v in model.trainable_variables)):
		print("non-finite weights after step", step, [v.name for v in model.trainable_variables if not fin(v)][:6], flush=True); break
print("DIAG_DONE", flush=True)
