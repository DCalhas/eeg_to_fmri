"""Diagnostic 2: (a) tfp VonMises sample gradients vs concentration; (b) when/where spectral training turns NaN."""
import argparse
import numpy as np
import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
parser.add_argument('-steps', default=1500, type=int)
opt = parser.parse_args()
from utils import process_utils, train
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=opt.seed)
import tensorflow as tf
import tensorflow_probability as tfp
from regularizers import path_sgd
print("tfp", tfp.__version__, "tf", tf.__version__, flush=True)

#(a) reparameterization gradients of VonMises samples w.r.t. concentration and loc
for k in [0., 1e-8, 1e-4, 1e-2, 0.1, 1., 10., 100.]:
	conc = tf.Variable(tf.fill((2000,), k)); loc = tf.Variable(tf.zeros((2000,)))
	with tf.GradientTape() as tape:
		s = tf.cos(tfp.distributions.VonMises(loc, conc).sample(seed=1))
		obj = tf.reduce_sum(s)
	gk, gl = tape.gradient(obj, [conc, loc])
	print("VonMises conc %-6g samples finite %s grad conc finite %.3f grad loc finite %.3f" % (k, bool(np.all(np.isfinite(s.numpy()))), np.mean(np.isfinite(gk.numpy())), np.mean(np.isfinite(gl.numpy()))), flush=True)

#(b) training
splits, _, _ = experiment.load_split(opt)
eeg, fmri, _ = splits["train"]
model, lr = experiment.build_model(opt, (None,)+eeg.shape[1:], (None,)+fmri.shape[1:], n_train=eeg.shape[0])
optimizer = path_sgd.optimizer("Adam", None, model, lr)
loss_fn = experiment.loss_fn(opt)
fin = lambda t: bool(np.all(np.isfinite(t.numpy())))
layer = [l for l in model.decoder.layers if "variational_i_dct3d" in l.name][0]
rng = np.random.RandomState(0)
for step in range(opt.steps):
	i = rng.randint(0, eeg.shape[0]-4)
	batch = (tf.constant(eeg[i:i+4]), tf.constant(fmri[i:i+4]))
	with tf.GradientTape() as tape:
		logits = model(*batch, training=True)
		loss = loss_fn(batch[1], logits)+(tf.add_n(model.losses) if model.losses else 0.)
	grads = tape.gradient(loss, model.trainable_variables)
	sc = layer.scale.numpy()
	if(step % 100 == 0 or not fin(loss) or any(g is not None and not fin(g) for g in grads)):
		out = logits[0]
		print("step", step, "loss %.3f" % float(tf.reduce_mean(loss)), "conc min %.2e max %.2e n_zero %d" % (sc.min(), sc.max(), int((sc == 0).sum())),
			"b range", [float(tf.reduce_min(out[1])), float(tf.reduce_max(out[1]))], "mu absmax %.3g" % float(tf.reduce_max(tf.abs(out[0]))), flush=True)
	bad = [v.name for v, g in zip(model.trainable_variables, grads) if g is not None and not fin(g)]
	if(bad or not fin(loss)):
		print("FIRST NON-FINITE at step", step, "loss finite", fin(loss), "bad grads", bad[:8], flush=True)
		gs = dict(zip([v.name for v in model.trainable_variables], grads))
		g = gs.get(layer.scale.name)
		if(g is not None):
			g = g.numpy(); print("conc grad nonfinite where conc == 0:", int((~np.isfinite(g) & (sc == 0)).sum()), "where conc > 0:", int((~np.isfinite(g) & (sc > 0)).sum()), flush=True)
		break
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("DIAG2_DONE", flush=True)
