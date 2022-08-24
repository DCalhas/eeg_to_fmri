import tensorflow as tf

import tensorflow_probability as tfp

class DenseVariational(tf.keras.layers.Layer):
	
	def __init__(
			self,
			units,
			activation=None,
			activity_regularizer=None,
			kernel_initializer="GlorotUniform",
			bias_initializer="GlorotUniform",
			use_bias=False,
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
		self.activation=activation
		self.kernel_initializer=kernel_initializer
		self.bias_initializer=bias_initializer
		self.use_bias=use_bias
		self.trainable=trainable
		self.seed = seed

	def build(self, input_shape):
		assert len(input_shape)==2#only for batch, features rank

		last_dim=input_shape[-1]

		input_shape = tf.TensorShape(input_shape)

		self.kernel_mu = self.add_weight(
				'kernel_mu',
				shape=[last_dim, self.units],
				initializer=getattr(tf.keras.initializers, self.kernel_initializer)(),
				dtype=tf.float32,
				trainable=self.trainable)
		self.kernel_sigma = self.add_weight(
				'kernel_sigma',
				shape=[last_dim, self.units],
				initializer=getattr(tf.keras.initializers, self.kernel_initializer)(),
				dtype=tf.float32,
				trainable=self.trainable)

		if self.use_bias:
			self.bias_mu = self.add_weight(
					'bias_mu',
					shape=[self.units,],
					initializer=getattr(tf.keras.initializers, self.bias_initializer)(),
					dtype=self.dtype,
					trainable=self.trainable)

			self.bias_sigma = self.add_weight(
					'bias_sigma',
					shape=[self.units,],
					initializer=getattr(tf.keras.initializers, self.bias_initializer)(),
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
		bias=self.bias_mu+self.bias_sigma*epsilon_bias

		return tf.matmul(X, kernel)+bias

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
				'activation': self.activation,
				'kernel_initializer': self.kernel_initializer,
				'bias_initializer': self.bias_initializer,
				'use_bias': self.use_bias,
				'trainable': self.trainable,
				'seed': self.seed,}

	@classmethod
	def from_config(cls, config):
		return cls(**config)