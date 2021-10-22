import tensorflow as tf

import numpy as np

_SUPPORTED_RBF_KERNEL_TYPES = ['gaussian']

def _get_default_scale(initializer, input_dim):
	if (isinstance(initializer, str) and
			initializer.lower() == 'gaussian'):
		return np.sqrt(input_dim / 2.0)
	return 1.0

def _get_random_features_initializer(initializer, shape):
	"""Returns Initializer object for random features."""

	def _get_cauchy_samples(loc, scale, shape):
		probs = np.random.uniform(low=0., high=1., size=shape)
		return loc + scale * np.tan(np.pi * (probs - 0.5))

	random_features_initializer = initializer
	if isinstance(initializer, str):
		if initializer.lower() == 'gaussian':
			random_features_initializer = tf.compat.v1.random_normal_initializer(
					stddev=1.0)
		elif initializer.lower() == 'laplacian':
			random_features_initializer = tf.compat.v1.constant_initializer(
					_get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))

		else:
			raise ValueError(
					'Unsupported kernel type: \'{}\'. Supported kernel types: {}.'.format(
							random_features_initializer, _SUPPORTED_RBF_KERNEL_TYPES))
	return random_features_initializer

class RandomFourierFeatures(tf.keras.layers.Layer):

	def __init__(self, output_dim, kernel_initializer='gaussian', scale=None, trainable=False, name=None, **kwargs):
		if output_dim <= 0:
			raise ValueError(
			'`output_dim` should be a positive integer. Given: {}.'.format(
			output_dim))
		if isinstance(kernel_initializer, str):
			if kernel_initializer.lower() not in _SUPPORTED_RBF_KERNEL_TYPES:
				raise ValueError(
				'Unsupported kernel type: \'{}\'. Supported kernel types: {}.'
				.format(kernel_initializer, _SUPPORTED_RBF_KERNEL_TYPES))
		if scale is not None and scale <= 0.0:
			raise ValueError('When provided, `scale` should be a positive float. '
			'Given: {}.'.format(scale))
		super(RandomFourierFeatures, self).__init__(name=name)
		self.output_dim = output_dim
		self.kernel_initializer = kernel_initializer
		self.scale = scale
		super(RandomFourierFeatures, self).__init__(trainable=trainable, **kwargs)

	def build(self, input_shape):
		input_shape = tf.TensorShape(input_shape)
		# TODO(pmol): Allow higher dimension inputs. Currently the input is expected
		# to have shape [batch_size, dimension].
		if input_shape.rank != 2:
			raise ValueError(
			'The rank of the input tensor should be 2. Got {} instead.'.format(
			input_shape.ndims))
		if input_shape.dims[1].value is None:
			raise ValueError(
			'The last dimension of the inputs to `RandomFourierFeatures` '
			'should be defined. Found `None`.')
		input_dim = input_shape.dims[1].value

		kernel_initializer = _get_random_features_initializer(self.kernel_initializer, shape=(input_dim, self.output_dim))

		self.unscaled_kernel = self.add_weight(name='unscaled_kernel',
											shape=(input_dim, self.output_dim),dtype=tf.float32,
											initializer=kernel_initializer,trainable=False)

		self.bias = self.add_weight(name='bias',shape=(self.output_dim,),
									dtype=tf.float32, initializer=tf.compat.v1.random_uniform_initializer(
									minval=0.0, maxval=2 * np.pi, dtype=tf.float32),
									trainable=False)

		if self.scale is None:
			self.scale = _get_default_scale(self.kernel_initializer, input_dim)
		self.kernel_scale = self.add_weight(name='kernel_scale',shape=(1,),
											dtype=tf.float32, initializer=tf.compat.v1.constant_initializer(self.scale),
											trainable=True, constraint='NonNeg')
		super(RandomFourierFeatures, self).build(input_shape)

	def call(self, inputs):
		inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
		inputs = tf.cast(inputs, tf.float32)
		kernel = (1.0 / self.kernel_scale) * self.unscaled_kernel
		outputs = tf.raw_ops.MatMul(a=inputs, b=kernel)
		outputs = tf.nn.bias_add(outputs, self.bias)
		return tf.cos(outputs)

	def compute_output_shape(self, input_shape):
		input_shape = tf.TensorShape(input_shape)
		input_shape = input_shape.with_rank(2)
		if input_shape.dims[-1].value is None:
			raise ValueError(
				'The innermost dimension of input shape must be defined. Given: %s' %
				input_shape)
		return input_shape[:-1].concatenate(self.output_dim)

	def get_config(self):
		kernel_initializer = self.kernel_initializer
		if not isinstance(kernel_initializer, str):
			kernel_initializer = initializers.serialize(kernel_initializer)
		config = {
			'output_dim': self.output_dim,
			'kernel_initializer': kernel_initializer,
			'scale': self.scale,
		}
		base_config = super(RandomFourierFeatures, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))



_get_default_scale

_get_random_features_initializer