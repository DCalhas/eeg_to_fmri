import tensorflow as tf

class DenseTemporal(tf.keras.layers.Layer):
	
	def __init__(
			self,
			units,
			activation=None,
			activity_regularizer=None,
			kernel_initializer="GlorotUniform",
			bias_initializer="GlorotUniform",
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
		super(DenseTemporal, self).__init__(
				activity_regularizer=activity_regularizer,
				**kwargs)

		self.units=units
		self.activation_fn=activation
		if(self.activation_fn is None):
			self.activation_fn="linear"
		self.kernel_initializer=kernel_initializer
		self.bias_initializer=bias_initializer
		self.use_bias=use_bias
		self.trainable=trainable
		self.seed = seed

	def build(self, input_shape):
		assert len(input_shape)==2#only for batch, features rank

		last_dim=input_shape[-1]

		input_shape = tf.TensorShape(input_shape)

		kernel_initializer=self.kernel_initializer
		bias_initializer=self.bias_initializer

		if(type(kernel_initializer) is str):
			kernel_initializer=getattr(tf.keras.initializers, kernel_initializer)()
		if(type(bias_initializer) is str):
			bias_initializer=getattr(tf.keras.initializers, bias_initializer)()


		self.kernel = self.add_weight(
				'kernel',
				shape=[last_dim, self.units],
				initializer=kernel_initializer,
				dtype=tf.float32,
				trainable=self.trainable)

		if self.use_bias:
			self.bias = self.add_weight(
					'bias',
					shape=[self.units,],
					initializer=bias_initializer,
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
		output=tf.matmul(tf.transpose(X, perm=[0,2,1]), self.kernel)
		if(self.use_bias):
			output+=self.bias
			
		return tf.transpose(getattr(tf.keras.activations, self.activation_fn)(output), perm=[0,2,1])

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
				'kernel_initializer': self.kernel_initializer,
				'bias_initializer': self.bias_initializer,
				'use_bias': self.use_bias,
				'trainable': self.trainable,
				'seed': self.seed,}

	@classmethod
	def from_config(cls, config):
		return cls(**config)