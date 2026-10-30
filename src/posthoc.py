"""
Post-hoc epistemic uncertainty on a trained (heteroscedastic) decoder.

	* SWAG-diagonal (Maddox et al. 2019): after training, keep optimizing with a constant learning rate and fit a
	  diagonal Gaussian to the snapshots of the decoder weights; every prediction samples weights from it.
	* Last-layer Laplace, diagonal (Daxberger et al. 2021): Gaussian around the trained weights of the decoder dense
	  layer with precision = empirical Fisher (sum of squared per-volume gradients of the Laplace NLL, summed over voxels)
	  + prior precision; the prior precision is chosen by the NLL on calibration volumes.

Samplers are callables eeg -> decoder output, as expected by evaluate_uq.evaluate.
"""
import numpy as np
import tensorflow as tf

from utils import train


def batchnorm_layers(model):
	return [l for l in model.decoder.submodules if isinstance(l, tf.keras.layers.BatchNormalization)]


def _bn_step(model, layers, eeg, k):
	"""
	One compiled training-mode pass with BatchNorm momentum 1-1/(k+1). Keras reads the momentum when the graph is
	traced, so k is a Python argument: one graph per step k (n_batches graphs, traced once and reused for every sample).
	Eager calls are avoided because they leak GPU memory (see evaluate_uq._compiled_decoder).
	"""
	if(getattr(model, "_bn_fn", None) is None):
		def step(e, k):
			for l in layers:
				l.momentum = 1.-1./(k+1)
			return model.decoder(e, training=True)
		model._bn_fn = tf.function(step)
	model._bn_fn(eeg, k)


def bn_update(model, eeg, rng, n_batches=10, batch=16):
	"""Recompute the BatchNorm moving statistics for the current weights (Maddox et al. 2019, bn_update): the momentum of
	step k is 1-1/(k+1), so the moving statistics end as the exact average of n_batches training batch statistics."""
	layers = batchnorm_layers(model)
	if(not layers):
		return
	momentum = [l.momentum for l in layers]
	idx = rng.choice(eeg.shape[0], size=min(n_batches*batch, eeg.shape[0]), replace=False)
	for k, start in enumerate(range(0, len(idx), batch)):
		_bn_step(model, layers, tf.convert_to_tensor(eeg[np.sort(idx[start:start+batch])]), k)
	for l, m in zip(layers, momentum):
		l.momentum = m


class WeightSampler:
	"""
	Calls the decoder with weights drawn from N(mean, std^2) for the given variables, restoring them afterwards.
	With bn_eeg (training EEG) the BatchNorm statistics are recomputed for every sampled weight vector (needed when the
	sampled variables feed BatchNorm layers, as in SWAG over the whole decoder).
	"""
	def __init__(self, model, variables, mean, std, seed=0, bn_eeg=None):
		self.model, self.variables, self.mean, self.std, self.bn_eeg = model, variables, mean, std, bn_eeg
		self.rng = tf.random.Generator.from_seed(seed)
		self.np_rng = np.random.RandomState(seed)
		self.bn_vars = [v for l in batchnorm_layers(model) for v in (l.moving_mean, l.moving_variance)] if bn_eeg is not None else []
		self.bn_saved = [tf.identity(v) for v in self.bn_vars]

	def __call__(self, eeg):
		for v, m, s in zip(self.variables, self.mean, self.std):
			v.assign(m+s*self.rng.normal(m.shape))
		if(self.bn_eeg is not None):
			bn_update(self.model, self.bn_eeg, self.np_rng)
		import evaluate_uq
		out = evaluate_uq._compiled_decoder(self.model)(tf.convert_to_tensor(eeg))
		self.restore()
		return out

	def restore(self):
		for v, m in zip(self.variables, self.mean):
			v.assign(m)
		for v, m in zip(self.bn_vars, self.bn_saved):
			v.assign(m)


def swag(model, train_set, optimizer, loss_fn, epochs=3, learning_rate=None, snapshots_per_epoch=10, seed=0, bn_eeg=None):
	"""
	SWAG-diagonal with a constant learning rate. Uses the training optimizer (Adam) rather than SGD as in Maddox et al.;
	disclosed. bn_eeg: training EEG used to recompute the BatchNorm statistics of the SWA mean and of every sample.
	"""
	variables = model.decoder.trainable_variables
	if(learning_rate is not None):
		optimizer.learning_rate.assign(learning_rate)
	n_batches = sum(1 for _ in train_set)
	every = max(1, n_batches//snapshots_per_epoch)
	mean = [tf.zeros_like(v) for v in variables]; sq = [tf.zeros_like(v) for v in variables]; n = 0
	for epoch in range(epochs):
		for k, batch in enumerate(train_set):
			train.train_step(model, batch, optimizer, loss_fn, u_architecture=True)
			if((k+1) % every == 0):
				n += 1
				mean = [m+(v-m)/n for v, m in zip(variables, mean)]
				sq = [q+(v**2-q)/n for v, q in zip(variables, sq)]
	std = [tf.sqrt(tf.maximum(q-m**2, 0.)) for m, q in zip(mean, sq)]
	print("I: SWAG with", n, "snapshots, mean std", float(np.mean([tf.reduce_mean(s).numpy() for s in std])), flush=True)
	#the SWA mean becomes the reference weights, with BatchNorm statistics recomputed for it
	for v, m in zip(variables, mean):
		v.assign(m)
	if(bn_eeg is not None):
		bn_update(model, bn_eeg, np.random.RandomState(seed))
	return WeightSampler(model, variables, mean, std, seed=seed, bn_eeg=bn_eeg)


def _laplace_nll_sum(model, x, y):
	mu, b = model.decoder(x, training=False)[:2]
	return tf.reduce_sum(tf.math.abs(y-mu)/b+tf.math.log(2*b))


def last_layer_laplace(model, train_data, calib_data, layer_name="decoder_dense", priors=10.**np.arange(-2, 9),
						n_calib=64, T=20, seed=0):
	layer = model.decoder.get_layer(layer_name)
	variables = layer.trainable_variables
	fisher = [tf.zeros_like(v) for v in variables]
	eeg, fmri = train_data
	@tf.function
	def squared_gradients(x, y):
		with tf.GradientTape() as tape:
			nll = _laplace_nll_sum(model, x, y)
		return [g**2 for g in tape.gradient(nll, variables)]
	for i in range(eeg.shape[0]):
		fisher = [f+g for f, g in zip(fisher, squared_gradients(eeg[i:i+1], fmri[i:i+1]))]
	mean = [tf.identity(v) for v in variables]

	#prior precision by the calibration NLL of the Laplace mixture (on a subset of calibration volumes)
	import evaluate_uq
	eeg_cal, fmri_cal = calib_data
	idx = np.random.RandomState(seed).choice(eeg_cal.shape[0], size=min(n_calib, eeg_cal.shape[0]), replace=False)
	best, best_nll = None, np.inf
	for prior in priors:
		sampler = WeightSampler(model, variables, mean, [1./tf.sqrt(f+prior) for f in fisher], seed=seed)
		nll = []
		for i in idx:
			family, params = evaluate_uq.sample_predictions([sampler], eeg_cal[i:i+1], T, True)
			flat = [tf.reshape(p, (p.shape[0], 1, -1)) for p in params]
			y = tf.reshape(tf.constant(fmri_cal[i, ..., 0], tf.float32), (1, -1))
			nll.append(float(tf.reduce_mean(evaluate_uq.Predictive(family, flat).nll(y))))
		print("I: Laplace prior precision", prior, "calibration NLL", np.mean(nll), flush=True)
		if(np.mean(nll) < best_nll):
			best, best_nll = prior, np.mean(nll)
	print("I: Laplace selected prior precision", best, flush=True)
	return WeightSampler(model, variables, mean, [1./tf.sqrt(f+best) for f in fisher], seed=seed), best
