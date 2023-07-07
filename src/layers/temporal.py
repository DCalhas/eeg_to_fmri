import tensorflow as tf

from tensorflow_addons.activations import sparsemax

from layers.fourier_features import RandomFourierFeatures, Sinusoids

import numpy as np


class IntervalLengthError(Exception):
	
	def __init__(self, t, i):
		
		super(IntervalLengthError, self).__init__("Temporal dimension "+str(t)+" is lower than the time length "+str(i)+" given.")

class StridedTemporalLengthEEG(tf.keras.layers.Layer):

	def __init__(self, time_length, **kwargs):
		
		super(StridedTemporalLengthEEG, self).__init__(**kwargs)

		self.time_length=time_length


	def call(self,x):
		"""
		strided slice of jump 1, for all windowed time lengths
		"""
		if(x.shape[3] < self.time_length):
			raise IntervalLengthError(x.shape[3], self.time_length)

		return tf.concat([tf.expand_dims(tf.slice(x, [0, 0, 0, t, 0], [-1, x.shape[1], x.shape[2], self.time_length, x.shape[4]]), axis=1) for t in range(self.time_length)], axis=1)


	def get_config(self):
		"""Returns the config of the layer.
		A layer config is a Python dictionary (serializable) containing the
		configuration of a layer. The same layer can be reinstantiated later
		(without its trained weights) from this configuration.
		Returns:
			config: A Python dictionary of class keyword arguments and their
				serialized values.
		"""
		return {'time_length': self.time_length,}

	@classmethod
	def from_config(cls, config):
		return cls(**config)


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
		last_dim=input_shape[1]

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


class StyleTemporal(tf.keras.layers.Layer):
	
	def __init__(
			self,
			units,
			flat_shape_scores,
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
		super(StyleTemporal, self).__init__(
				activity_regularizer=activity_regularizer,
				**kwargs)

		self.units=units
		self.kernel_initializer=kernel_initializer
		self.bias_initializer=bias_initializer
		self.use_bias=use_bias
		self.trainable=trainable
		self.seed = seed

		self.reshape_style=tf.keras.layers.Reshape(flat_shape_scores)
		self.dense=tf.keras.layers.Dense(units, use_bias=use_bias)

	
	@tf.function
	def call(self, x, style):

		latent_style=self.dense(self.reshape_style(style))

		return tf.transpose(latent_style, perm=[0,2,1])*x

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
				'kernel_initializer': self.kernel_initializer,
				'bias_initializer': self.bias_initializer,
				'use_bias': self.use_bias,
				'trainable': self.trainable,
				'seed': self.seed,}

	@classmethod
	def from_config(cls, config):
		return cls(**config)


class LSTMFourierDecoder(tf.keras.layers.Layer):

	def __init__(self, in_dim, time_length, latent_dim=None,
						kernel_output_initializer="GlorotUniform",
						kernel_input_initializer="GlorotUniform",
						kernel_forget_initializer="GlorotUniform",
						kernel_context_initializer="GlorotUniform",
						bias_output_initializer="Zeros",
						bias_input_initializer="Zeros",
						bias_forget_initializer="Zeros",
						bias_context_initializer="Zeros",
						fourier_normalization="layer",
						trainable=True,
						seed=None,
					 	**kwargs):
		

		super(LSTMFourierDecoder, self).__init__(**kwargs)

		self.in_dim=in_dim
		self.time_length=time_length
		self.latent_dim=latent_dim
		if(self.latent_dim is None):
			self.latent_dim=in_dim

		self.trainable=trainable

		#kernel initializers
		self.kernel_output_initializer=kernel_output_initializer
		self.kernel_input_initializer=kernel_input_initializer
		self.kernel_forget_initializer=kernel_forget_initializer
		self.kernel_context_initializer=kernel_context_initializer
		self.bias_output_initializer=bias_output_initializer
		self.bias_input_initializer=bias_input_initializer
		self.bias_forget_initializer=bias_forget_initializer
		self.bias_context_initializer=bias_context_initializer

		self.fourier_projection=RandomFourierFeatures(self.latent_dim, normalization=fourier_normalization, trainable=True, seed=seed, name="random_fourier_features")

		self.sinusoids=Sinusoids()

	def build(self, input_shape):

		self.weight_input = self.add_weight('weight_input',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_input_initializer,dtype=tf.float32,trainable=self.trainable)
		#self.weight_output = self.add_weight('weight_output',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_output_initializer,dtype=tf.float32,trainable=self.trainable)
		self.weight_forget = self.add_weight('weight_forget',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_forget_initializer,dtype=tf.float32,trainable=self.trainable)
		self.weight_context = self.add_weight('weight_context',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_context_initializer,dtype=tf.float32,trainable=self.trainable)

		self.recurrent_input = self.add_weight('recurrent_input',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_input_initializer,dtype=tf.float32,trainable=self.trainable)
		#self.recurrent_output = self.add_weight('recurrent_output',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_output_initializer,dtype=tf.float32,trainable=self.trainable)
		self.recurrent_forget = self.add_weight('recurrent_forget',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_forget_initializer,dtype=tf.float32,trainable=self.trainable)
		self.recurrent_context = self.add_weight('recurrent_context',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_context_initializer,dtype=tf.float32,trainable=self.trainable)

		self.bias_input = self.add_weight('weight_input',shape=[self.latent_dim,],initializer=self.bias_input_initializer,dtype=tf.float32,trainable=self.trainable)
		#self.bias_output = self.add_weight('weight_output',shape=[self.latent_dim,],initializer=self.bias_output_initializer,dtype=tf.float32,trainable=self.trainable)
		self.bias_forget = self.add_weight('weight_forget',shape=[self.latent_dim,],initializer=self.bias_forget_initializer,dtype=tf.float32,trainable=self.trainable)
		self.bias_context = self.add_weight('weight_context',shape=[self.latent_dim,],initializer=self.bias_context_initializer,dtype=tf.float32,trainable=self.trainable)

		self.init_latent = self.add_weight('init_latent',shape=[1,1,self.latent_dim,],initializer=self.kernel_output_initializer,dtype=tf.float32,trainable=self.trainable)
		self.init_context = self.add_weight('init_context',shape=[1,1,self.latent_dim,],initializer=self.kernel_output_initializer,dtype=tf.float32,trainable=self.trainable)

		self.built=True

	def call(self, x):
		#place spatial dimension in the last channel
		x=tf.transpose(x, perm=[0,2,1])
		z=self.fourier_projection(x)

		hidden_states=np.empty( (self.time_length+1,), dtype=type(self.init_latent.value()))
		cos_hidden_states=np.empty( (self.time_length+1,), dtype=type(self.init_latent.value()))
		context_states=np.empty( (self.time_length+1,), dtype=type(self.init_context.value()))

		hidden_states[0]=self.init_latent.value()
		cos_hidden_states[0]=self.init_latent.value()
		context_states[0]=self.init_context.value()


		for t in range(self.time_length):
			z_t=tf.slice(z, [0, t, 0], [-1, 1, z.shape[2]])
			
			#output gate
			#o=self.sinusoids(tf.matmul(z_t, self.weight_output)+tf.matmul(hidden_states[t], self.recurrent_output)+self.bias_output)
			o=self.sinusoids(z_t)
			#input gate
			i=tf.keras.activations.sigmoid(tf.matmul(z_t, self.weight_input)+tf.matmul(hidden_states[t], self.recurrent_input)+self.bias_input)
			#forget gate
			f=tf.keras.activations.sigmoid(tf.matmul(z_t, self.weight_forget)+tf.matmul(hidden_states[t], self.recurrent_forget)+self.bias_forget)
			#context gate
			c=tf.keras.activations.tanh(tf.matmul(z_t, self.weight_context)+tf.matmul(hidden_states[t], self.recurrent_context)+self.bias_context)

			#apply forget and input gates
			c=f*context_states[t]+i*c
			h=o*c

			#add context and hidden state
			hidden_states[t+1]=z_t#h
			cos_hidden_states[t+1]=h#h
			context_states[t+1]=c

		return tf.transpose(tf.concat(cos_hidden_states[1:].tolist(), axis=1), perm=[0,2,1])

	def get_config(self):
		"""Returns the config of the layer.
		A layer config is a Python dictionary (serializable) containing the
		configuration of a layer. The same layer can be reinstantiated later
		(without its trained weights) from this configuration.
		Returns:
			config: A Python dictionary of class keyword arguments and their
				serialized values.
		"""
		return {'in_dim': self.in_dim,
				'time_length': self.time_length,
				'latent_dim': self.latent_dim,
				"kernel_output_initializer": self.kernel_output_initializer,
				"kernel_input_initializer": self.kernel_input_initializer,
				"kernel_forget_initializer": self.kernel_forget_initializer,
				"kernel_context_initializer": self.kernel_context_initializer,
				"bias_output_initializer": self.bias_output_initializer,
				"bias_input_initializer": self.bias_input_initializer,
				"bias_forget_initializer": self.bias_forget_initializer,
				"bias_context_initializer": self.bias_context_initializer,
				"fourier_normalization": self.fourier_normalization,
				"seed": self.seed,}

	@classmethod
	def from_config(cls, config):
		return cls(**config)


class LSTMFourierConnectivityDecoder(tf.keras.layers.Layer):


	def __init__(self, in_dim, time_length, latent_dim=None,
						kernel_input_initializer="GlorotUniform",
						kernel_forget_initializer="GlorotUniform",
						bias_input_initializer="Zeros",
						bias_forget_initializer="Zeros",
						fourier_normalization="layer",
						trainable=True,
						seed=None,
					 	**kwargs):
		

		super(LSTMFourierConnectivityDecoder, self).__init__(**kwargs)

		self.in_dim=in_dim
		self.time_length=time_length
		self.latent_dim=latent_dim
		if(self.latent_dim is None):
			self.latent_dim=in_dim

		self.trainable=trainable

		#kernel initializers
		self.kernel_input_initializer=kernel_input_initializer
		self.kernel_forget_initializer=kernel_forget_initializer
		self.bias_input_initializer=bias_input_initializer
		self.bias_forget_initializer=bias_forget_initializer

		self.fourier_projection=RandomFourierFeatures(self.latent_dim, normalization=fourier_normalization, trainable=True, seed=seed, name="random_fourier_features")

		self.connectivity_projection=tf.keras.layers.Dense(self.latent_dim)

		self.sinusoids=Sinusoids()

	def build(self, input_shape):

		self.weight_input = self.add_weight('weight_input',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_input_initializer,dtype=tf.float32,trainable=self.trainable)
		self.weight_forget = self.add_weight('weight_forget',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_forget_initializer,dtype=tf.float32,trainable=self.trainable)

		self.recurrent_input = self.add_weight('recurrent_input',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_input_initializer,dtype=tf.float32,trainable=self.trainable)
		self.recurrent_forget = self.add_weight('recurrent_forget',shape=[self.latent_dim, self.latent_dim],initializer=self.kernel_forget_initializer,dtype=tf.float32,trainable=self.trainable)

		self.bias_input = self.add_weight('weight_input',shape=[self.latent_dim,],initializer=self.bias_input_initializer,dtype=tf.float32,trainable=self.trainable)
		self.bias_forget = self.add_weight('weight_forget',shape=[self.latent_dim,],initializer=self.bias_forget_initializer,dtype=tf.float32,trainable=self.trainable)

		self.init_latent = self.add_weight('init_latent',shape=[1,1,self.latent_dim,],initializer=self.kernel_input_initializer,dtype=tf.float32,trainable=self.trainable)
		self.init_context = self.add_weight('init_context',shape=[1,1,self.latent_dim,],initializer=self.kernel_input_initializer,dtype=tf.float32,trainable=self.trainable)

		self.built=True

	def call(self, x, connectivity):
		#place spatial dimension in the last channel
		x=tf.transpose(x, perm=[0,2,1])
		z=self.fourier_projection(x)

		hidden_states=np.empty( (self.time_length+1,), dtype=type(self.init_latent.value()))
		cos_hidden_states=np.empty( (self.time_length+1,), dtype=type(self.init_latent.value()))
		context_states=np.empty( (self.time_length+1,), dtype=type(self.init_context.value()))

		hidden_states[0]=self.init_latent.value()
		cos_hidden_states[0]=self.init_latent.value()
		context_states[0]=self.init_context.value()

		for t in range(self.time_length):
			z_t=tf.slice(z, [0, t, 0], [-1, 1, z.shape[2]])
			connectivity_t=tf.slice(connectivity, [0, t, 0], [-1, 1, connectivity.shape[2]])
			
			#output gate
			o=self.sinusoids(z_t)
			#input gate
			i=tf.keras.activations.sigmoid(tf.matmul(z_t, self.weight_input)+tf.matmul(hidden_states[t], self.recurrent_input)+self.bias_input)
			#forget gate
			f=tf.keras.activations.sigmoid(tf.matmul(z_t, self.weight_forget)+tf.matmul(hidden_states[t], self.recurrent_forget)+self.bias_forget)
			
			#apply forget and input gates
			c=f*context_states[t]+i*self.connectivity_projection(connectivity_t)
			h=o*c

			#add context and hidden state
			hidden_states[t+1]=z_t#h
			cos_hidden_states[t+1]=h#h
			context_states[t+1]=c

		return tf.transpose(tf.concat(cos_hidden_states[1:].tolist(), axis=1), perm=[0,2,1])

	def get_config(self):
		"""Returns the config of the layer.
		A layer config is a Python dictionary (serializable) containing the
		configuration of a layer. The same layer can be reinstantiated later
		(without its trained weights) from this configuration.
		Returns:
			config: A Python dictionary of class keyword arguments and their
				serialized values.
		"""
		return {'in_dim': self.in_dim,
				'time_length': self.time_length,
				'latent_dim': self.latent_dim,
				"kernel_output_initializer": self.kernel_output_initializer,
				"kernel_input_initializer": self.kernel_input_initializer,
				"kernel_forget_initializer": self.kernel_forget_initializer,
				"kernel_context_initializer": self.kernel_context_initializer,
				"bias_output_initializer": self.bias_output_initializer,
				"bias_input_initializer": self.bias_input_initializer,
				"bias_forget_initializer": self.bias_forget_initializer,
				"bias_context_initializer": self.bias_context_initializer,
				"fourier_normalization": self.fourier_normalization,
				"seed": self.seed,}

	@classmethod
	def from_config(cls, config):
		return cls(**config)



class TemporalTopographicalAttention(tf.keras.layers.Layer):

	def __init__(self, channels, features, regularizer=None, seed=None, **kwargs):

		self.channels=channels
		self.features=features
		self.regularizer=regularizer
		self.seed=seed

		super(TemporalTopographicalAttention, self).__init__(**kwargs)

	def build(self, input_shape):
		self.A = self.add_weight('A',
								#shape=[self.channels,self.channels,self.features],
								shape=[self.channels,self.features],
								regularizer=self.regularizer,
								initializer=tf.initializers.GlorotUniform(seed=self.seed),
								dtype=tf.float32,
								trainable=True)
		
	def call(self, X):
		"""
		The defined topographical attention mechanism has an extra step:

			instead of performing the element wise product,
			one reduces the feature dimension,
			so instead of the attention weight matrix being of shape NxN
			it has shape NxNxF
			the F refers to the feature dimension that is reduced
		"""
		c = tf.matmul(X, tf.transpose(self.A, perm=[1,0]))
		
		W = sparsemax(c, axis=-1)
		self.attention_scores=W

		return tf.matmul(W, X), self.attention_scores


	def get_config(self):
		"""Returns the config of the layer.
		A layer config is a Python dictionary (serializable) containing the
		configuration of a layer. The same layer can be reinstantiated later
		(without its trained weights) from this configuration.
		Returns:
			config: A Python dictionary of class keyword arguments and their
				serialized values.
		"""
		return {'in_dim': self.in_dim,
				"seed": self.seed,}

	@classmethod
	def from_config(cls, config):
		return cls(**config)
