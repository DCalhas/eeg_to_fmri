import tensorflow as tf

import tensorflow_probability as tfp

class DenseVariational(tf.keras.layers.Layer):
	
	def __init__(
			self,
			units,
			activation=None,
			activity_regularizer=None,
			kernel_prior_initializer="GlorotUniform",
			kernel_posterior_initializer="GlorotUniform",
			bias_prior_initializer="GlorotUniform",
			bias_posterior_initializer="GlorotUniform",
			use_bias=True,
			trainable=True,
			seed=None,
			**kwargs):
		"""Construct layer.
		Args:
			${args}
			seed: Python scalar `int` which initializes the random number
				generator. Default value: `None` (i.e., use global seed).
		"""
		# pylint: enable=g-doc-args
		super(DenseVariational, self).__init__(
				activity_regularizer=activity_regularizer,
				**kwargs)

		self.units=units
		self.activation_fn=activation
		if(self.activation_fn is None):
			self.activation_fn="linear"
		self.kernel_prior_initializer=kernel_prior_initializer
		self.kernel_posterior_initializer=kernel_posterior_initializer
		self.bias_prior_initializer=bias_prior_initializer
		self.bias_posterior_initializer=bias_posterior_initializer
		self.use_bias=use_bias
		self.trainable=trainable
		self.seed = seed

	def build(self, input_shape):
		assert len(input_shape)==2#only for batch, features rank

		last_dim=input_shape[-1]

		input_shape = tf.TensorShape(input_shape)

		kernel_prior_initializer=self.kernel_prior_initializer
		kernel_posterior_initializer=self.kernel_posterior_initializer
		bias_prior_initializer=self.bias_prior_initializer
		bias_posterior_initializer=self.bias_posterior_initializer

		if(type(kernel_prior_initializer) is str):
			kernel_prior_initializer=getattr(tf.keras.initializers, kernel_prior_initializer)()
		if(type(kernel_posterior_initializer) is str):
			kernel_posterior_initializer=getattr(tf.keras.initializers, kernel_posterior_initializer)()
		if(type(bias_prior_initializer) is str):
			bias_prior_initializer=getattr(tf.keras.initializers, bias_prior_initializer)()
		if(type(bias_posterior_initializer) is str):
			bias_posterior_initializer=getattr(tf.keras.initializers, bias_posterior_initializer)()


		self.kernel_mu = self.add_weight(
				'kernel_mu',
				shape=[last_dim, self.units],
				initializer=kernel_prior_initializer,
				dtype=tf.float32,
				trainable=self.trainable)
		self.kernel_sigma = self.add_weight(
				'kernel_sigma',
				shape=[last_dim, self.units],
				initializer=kernel_posterior_initializer,
				dtype=tf.float32,
				trainable=self.trainable)

		if self.use_bias:
			self.bias_mu = self.add_weight(
					'bias_mu',
					shape=[self.units,],
					initializer=bias_prior_initializer,
					dtype=self.dtype,
					trainable=self.trainable)

			self.bias_sigma = self.add_weight(
					'bias_sigma',
					shape=[self.units,],
					initializer=bias_posterior_initializer,
					dtype=self.dtype,
					trainable=self.trainable)
		else:
			self.bias = None

		self.loc=0.0
		self.scale=1.0
		self.distribution="Normal"

		self.built = True

	@tf.function
	def call(self, X):

		epsilon_kernel = getattr(tfp.distributions, self.distribution)(self.loc, self.scale).sample()
		epsilon_bias = getattr(tfp.distributions, self.distribution)(self.loc, self.scale).sample()

		kernel=self.kernel_mu+self.kernel_sigma*epsilon_kernel
		if(self.use_bias):
			bias=self.bias_mu+self.bias_sigma*epsilon_bias
			
		output=tf.matmul(X, kernel)
		if(self.use_bias):
			output+=bias
			
		return getattr(tf.keras.activations, self.activation_fn)(output)

	def get_config(self):
		"""Returns the config of the layer.
		A layer config is a Python dictionary (serializable) containing the
		configuration of a layer. The same layer can be reinstantiated later
		(without its trained weights) from this configuration.
		Returns:
			config: A Python dictionary of class keyword arguments and their
				serialized values.
		"""
		return {'units': self.units,
				'activation': self.activation_fn,
				'kernel_posterior_initializer': self.kernel_posterior_initializer,
				'bias_posterior_initializer': self.bias_posterior_initializer,
				'kernel_prior_initializer': self.kernel_prior_initializer,
				'bias_prior_initializer': self.bias_prior_initializer,
				'use_bias': self.use_bias,
				'trainable': self.trainable,
				'seed': self.seed,}

	@classmethod
	def from_config(cls, config):
		return cls(**config)

"""
Layers for the journal experiments. DenseVariational above is kept untouched for reproducing the original runs:
it draws a single scalar epsilon shared by all weights and has no KL term, so it is not a Bayesian layer.
"""

class DenseFlipoutKL(tfp.layers.DenseFlipout):
	"""
	DenseFlipout whose KL(q||p) (kernel + bias) is registered as a callable, input-independent layer loss.
	tfp adds the KL inside call(), which fails ("out of scope") when the functional decoder is called from the
	tf.function-compiled EEG_to_fMRI.call; a callable loss is evaluated when model.losses is read.
	"""
	def __init__(self, units, kl_weight, **kwargs):
		super(DenseFlipoutKL, self).__init__(units, kernel_divergence_fn=None, bias_divergence_fn=None,
			bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(is_singular=False),
			bias_prior_fn=tfp.layers.default_multivariate_normal_fn, **kwargs)
		self.kl_weight=kl_weight

	def build(self, input_shape):
		super(DenseFlipoutKL, self).build(input_shape)
		kl = tfp.distributions.kl_divergence
		self.add_loss(lambda: self.kl_weight*(kl(self.kernel_posterior, self.kernel_prior)+kl(self.bias_posterior, self.bias_prior)))


def flipout_dense(units, kl_weight, seed=None):
	"""Mean-field Gaussian dense layer with Flipout (Wen et al. 2018), N(0,1) prior, KL scaled by kl_weight."""
	return DenseFlipoutKL(units, kl_weight, seed=seed)


class AlwaysDropout(tf.keras.layers.Layer):
	"""Dropout active at train and test time (MC dropout, Gal & Ghahramani 2016), independent of the training flag."""
	def __init__(self, rate, **kwargs):
		super(AlwaysDropout, self).__init__(**kwargs)
		self.rate=rate

	def call(self, x):
		return tf.nn.dropout(x, rate=self.rate)

	def get_config(self):
		return {"rate": self.rate}


class VoxelBias(tf.keras.layers.Layer):
	"""Trainable per-voxel offset, used for the log-scale of the aleatoric head."""
	def build(self, input_shape):
		self.bias=self.add_weight('voxel_bias', shape=input_shape[1:], initializer=tf.initializers.Zeros(), trainable=True)

	def call(self, x):
		return x+self.bias
