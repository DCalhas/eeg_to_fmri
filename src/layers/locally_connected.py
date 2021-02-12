import tensorflow as tf

from utils import conv_utils

import numpy as np

class LocallyConnected3D(tf.keras.layers.Layer):

	def __init__(self,
				filters,
				kernel_size,
				strides=(1, 1, 1),
				padding='valid',
				data_format=None,
				activation=None,
				use_bias=True,
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				kernel_regularizer=None,
				bias_regularizer=None,
				activity_regularizer=None,
				kernel_constraint=None,
				bias_constraint=None,
				implementation=2,
				**kwargs):
		super(LocallyConnected3D, self).__init__(**kwargs)
		self.filters = filters
		self.kernel_size = conv_utils.normalize_tuple(kernel_size, 3, 'kernel_size')
		self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
		self.padding = conv_utils.normalize_padding(padding)
		if self.padding != 'valid' and implementation == 1:
			raise ValueError('Invalid border mode for LocallyConnected3D '
							'(only "valid" is supported if implementation is 1): ' +
							padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.activation = tf.keras.activations.get(activation)
		self.use_bias = use_bias
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
		self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
		self.bias_constraint = tf.keras.constraints.get(bias_constraint)
		self.implementation = implementation
		self.input_spec = tf.keras.layers.InputSpec(ndim=5)

	def build(self, input_shape):
		if self.data_format == 'channels_last':
			input_x, input_y, input_z = input_shape[1:-1]
			input_filter = input_shape[4]
		else:
			input_x, input_y, input_z = input_shape[2:]
			input_filter = input_shape[1]
		if input_x is None or input_y is None or input_z is None:
			raise ValueError('The spatial dimensions of the inputs to '
							' a LocallyConnected2D layer '
							'should be fully-defined, but layer received '
							'the inputs shape ' + str(input_shape))
		output_x = conv_utils.conv_output_length(input_x, self.kernel_size[0],
													self.padding, self.strides[0])
		output_y = conv_utils.conv_output_length(input_y, self.kernel_size[1],
													self.padding, self.strides[1])
		output_z = conv_utils.conv_output_length(input_z, self.kernel_size[2],
													self.padding, self.strides[2])
		self.output_x = output_x
		self.output_y = output_y
		self.output_z = output_z

		if self.implementation == 1:
			raise NotImplementedError

		elif self.implementation == 2:
			if self.data_format == 'channels_first':
				self.kernel_shape = (input_filter, input_x, input_y, input_z, self.filters,
									self.output_x, self.output_y, self.output_z)
			else:
				self.kernel_shape = (input_x, input_y, input_z, input_filter,
									self.output_x, self.output_y, self.output_z, self.filters)

			self.kernel = self.add_weight(
				shape=self.kernel_shape,
				initializer=self.kernel_initializer,
				name='kernel',
				regularizer=self.kernel_regularizer,
				constraint=self.kernel_constraint)

			self.kernel_mask = get_locallyconnected_mask(
				input_shape=(input_x, input_y, input_z),
				kernel_shape=self.kernel_size,
				strides=self.strides,
				padding=self.padding,
				data_format=self.data_format,
			)

		elif self.implementation == 3:
			raise NotImplementedError

		else:
			raise ValueError('Unrecognized implementation mode: %d.' %
							self.implementation)

		if self.use_bias:
			self.bias = self.add_weight(
				shape=(output_x, output_y, output_z, self.filters),
				initializer=self.bias_initializer,
				name='bias',
				regularizer=self.bias_regularizer,
				constraint=self.bias_constraint)
		else:
			self.bias = None
		if self.data_format == 'channels_first':
			self.input_spec = tf.keras.layers.InputSpec(ndim=5, axes={1: input_filter})
		else:
			self.input_spec = tf.keras.layers.InputSpec(ndim=5, axes={-1: input_filter})
		self.built = True

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			x = input_shape[2]
			y = input_shape[3]
			z = input_shape[4]
		elif self.data_format == 'channels_last':
			x = input_shape[1]
			y = input_shape[2]
			z = input_shape[3]

		x = conv_utils.conv_output_length(x, self.kernel_size[0],
											self.padding, self.strides[0])
		y = conv_utils.conv_output_length(y, self.kernel_size[1],
											self.padding, self.strides[1])
		z = conv_utils.conv_output_length(z, self.kernel_size[2],
											self.padding, self.strides[2])

		if self.data_format == 'channels_first':
			return (input_shape[0], self.filters, x, y, z)
		elif self.data_format == 'channels_last':
			return (input_shape[0], x, y, z, self.filters)

	def call(self, inputs):
		if self.implementation == 1:
			raise NotImplementedError

		elif self.implementation == 2:
			output = local_conv_matmul(inputs, self.kernel, self.kernel_mask,
									self.compute_output_shape(inputs.shape))
		elif self.implementation == 3:
			raise NotImplementedError
		else:
			raise ValueError('Unrecognized implementation mode: %d.' %
							self.implementation)

		if self.use_bias:
			output = tf.keras.backend.bias_add(output, self.bias, data_format=self.data_format)

		output = self.activation(output)
		return output


def get_locallyconnected_mask(input_shape, kernel_shape, strides, padding,
															data_format):
	mask = conv_utils.conv_kernel_mask(
			input_shape=input_shape,
			kernel_shape=kernel_shape,
			strides=strides,
			padding=padding)

	ndims = int(mask.ndim / 2)

	if data_format == 'channels_first':
		mask = np.expand_dims(mask, 0)
		mask = np.expand_dims(mask, -ndims - 1)

	elif data_format == 'channels_last':
		mask = np.expand_dims(mask, ndims)
		mask = np.expand_dims(mask, -1)

	else:
		raise ValueError('Unrecognized data_format: ' + str(data_format))

	return mask


def local_conv_matmul(inputs, kernel, kernel_mask, output_shape):
	inputs_flat = tf.reshape(inputs, (tf.shape(inputs)[0], -1))

	kernel = kernel_mask * kernel
	kernel = make_2d(kernel, split_dim=tf.keras.backend.ndim(kernel) // 2)

	output_flat = tf.linalg.matmul(inputs_flat, kernel, b_is_sparse=True)
	output = tf.reshape(output_flat, [
			tf.shape(output_flat)[0],
	] + list(output_shape)[1:])
	return output

def make_2d(tensor, split_dim):
	shape = tf.shape(tensor)
	in_dims = shape[:split_dim]
	out_dims = shape[split_dim:]

	in_size = tf.reduce_prod(in_dims)
	out_size = tf.reduce_prod(out_dims)

	return tf.reshape(tensor, (in_size, out_size))