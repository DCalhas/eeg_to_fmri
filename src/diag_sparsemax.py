"""Diagnostic: does tfa sparsemax in Topographical_Attention produce NaN (forward or gradient) on the first training batch?"""
import argparse
import numpy as np
import experiment

parser = experiment.add_model_args(argparse.ArgumentParser())
opt = parser.parse_args()
from utils import process_utils
process_utils.process_setup_tensorflow(opt.gpu_mem, seed=opt.seed)
import tensorflow as tf
from tensorflow_addons.activations import sparsemax


def sparsemax_stable(z, axis=-1):
	"""Sparsemax (Martins & Astudillo 2016) by sorting; the support size is >= 1 by construction, no division by zero."""
	z = tf.convert_to_tensor(z)
	zs = tf.sort(z, axis=axis, direction="DESCENDING")
	k = tf.cast(tf.range(1, tf.shape(z)[axis]+1), z.dtype)
	cs = tf.cumsum(zs, axis=axis)
	support = tf.cast(1.+k*zs > cs, z.dtype)
	k_z = tf.reduce_sum(support, axis=axis, keepdims=True)
	tau = (tf.reduce_sum(zs*support, axis=axis, keepdims=True)-1.)/tf.maximum(k_z, 1.)
	return tf.maximum(z-tau, 0.)


splits, _, _ = experiment.load_split(opt)
eeg, fmri, _ = splits["train"]
train_set = tf.data.Dataset.from_tensor_slices((eeg, fmri)).shuffle(eeg.shape[0], seed=opt.seed, reshuffle_each_iteration=True).batch(opt.batch_size)
batch = next(iter(train_set))
model, _ = experiment.build_model(opt, (None,)+eeg.shape[1:], (None,)+fmri.shape[1:], n_train=eeg.shape[0])
att = [l for l in model.eeg_encoder.layers if "topographical" in l.name][0]
X = tf.reshape(batch[0], (batch[0].shape[0], att.channels, att.features))
with tf.GradientTape(persistent=True) as tape:
	tape.watch(att.A)
	c = tf.tensordot(X, att.A, axes=[[2], [1]])
	W_tfa, W_st = sparsemax(c, axis=-1), sparsemax_stable(c, axis=-1)
	o_tfa, o_st = tf.reduce_sum(tf.linalg.matmul(W_tfa, X)**2), tf.reduce_sum(tf.linalg.matmul(W_st, X)**2)
g_tfa, g_st = tape.gradient(o_tfa, att.A), tape.gradient(o_st, att.A)
fin = lambda t: bool(np.all(np.isfinite(t.numpy())))
cn = c.numpy()
print("config", opt.name, "c finite", fin(c), "c absmax %.3g" % np.abs(cn).max(), "row max-min min %.3g" % (cn.max(-1)-cn.min(-1)).min(), flush=True)
print("tfa sparsemax forward finite", fin(W_tfa), "grad finite", fin(g_tfa), "| stable forward finite", fin(W_st), "grad finite", fin(g_st), flush=True)
ok = np.isfinite(W_tfa.numpy())
print("max |tfa - stable| where tfa finite %.3g" % float(np.abs(W_tfa.numpy()[ok]-W_st.numpy()[ok]).max()), "rows with NaN in tfa", int((~ok).any(-1).sum()), flush=True)
print("A init absmax %.4g sum %.6g" % (float(tf.reduce_max(tf.abs(att.A))), float(tf.reduce_sum(att.A))), flush=True)
