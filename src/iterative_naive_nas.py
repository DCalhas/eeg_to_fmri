import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.python.keras.utils import conv_utils

import gen_dims_utils

layers = [tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose, 
								tf.keras.layers.Conv3D, tf.keras.layers.Conv3DTranspose]


class iterative_naive_nas:


	def __init__(self):
		self.iterations = 0


	######################################################################################################################################
	#
	#												Generation of Kernel and Stride Sizes
	#
	######################################################################################################################################


	def generate_kernel_stride_Conv1D(self, input_shape, output_shape):
		possible = gen_dims_utils.get_possible_kernel_size_conv(input_shape[0], output_shape[0])

		pos = list(range(len(possible)))
		generated_kernel_stride = possible[np.random.choice(pos)]

		return {'kernel': (generated_kernel_stride[0],),
				'stride': (generated_kernel_stride[1],)}


	def generate_kernel_stride_Conv1DTranspose(self, input_shape, output_shape):
		possible = gen_dims_utils.get_possible_kernel_size_deconv(input_shape[0], output_shape[0])

		pos = list(range(len(possible)))
		generated_kernel_stride = possible[np.random.choice(pos)]

		return {'kernel': (generated_kernel_stride[0],),
				'stride': (generated_kernel_stride[1],)}


	def generate_kernel_stride_Conv2D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv1D((input_shape[0],1), (output_shape[0], 1))

		return gen_dims_utils.add_generated_dim(generated, input_shape[1], output_shape[1], gen_dims_utils.get_possible_kernel_size_conv)


	def generate_kernel_stride_Conv2DTranspose(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv1DTranspose((input_shape[0],1), (output_shape[0], 1))

		return gen_dims_utils.add_generated_dim(generated, input_shape[1], output_shape[1], gen_dims_utils.get_possible_kernel_size_deconv)


	def generate_kernel_stride_Conv3D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv2D((input_shape[0], input_shape[1],1), (output_shape[0], output_shape[1], 1))

		return gen_dims_utils.add_generated_dim(generated, input_shape[2], output_shape[2], gen_dims_utils.get_possible_kernel_size_conv)


	def generate_kernel_stride_Conv3DTranspose(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv2DTranspose((input_shape[0], input_shape[1], 1), (output_shape[0], output_shape[1], 1))
		
		return gen_dims_utils.add_generated_dim(generated, input_shape[2], output_shape[2], gen_dims_utils.get_possible_kernel_size_deconv)


	######################################################################################################################################
	#
	#															Build Layer Functions
	#
	######################################################################################################################################

	#Dense layer is simple to generate
	def build_layer_Dense(self, input_shape, output_shape):
		if(type(output_shape) is not int):
			return None
		return layers[0](output_shape, input_shape=input_shape)


	def build_layer_Conv2D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv2D(input_shape, output_shape)

		return layers[1](1, kernel_size=generated['kernel'], strides=generated['stride'], input_shape=input_shape)


	def build_layer_Conv2DTranspose(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv2DTranspose(input_shape, output_shape)

		return layers[2](1, kernel_size=generated['kernel'],
							strides=generated['stride'], 
							padding='valid', 
							output_padding=None,
							dilation_rate=(1, 1),
							input_shape=input_shape)


	def build_layer_Conv3D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv3D(input_shape, output_shape)

		return layers[3](1, kernel_size=generated['kernel'], strides=generated['stride'], input_shape=input_shape)


	def build_layer_Conv3DTranspose(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv3DTranspose(input_shape, output_shape)

		return layers[4](1, kernel_size=generated['kernel'],
							strides=generated['stride'], 
							padding='valid', 
							output_padding=None,
							dilation_rate=(1, 1, 1),
							input_shape=input_shape)


	######################################################################################################################################
	#
	#													Iterative Neural Architecture Search
	#
	######################################################################################################################################


	def search(self):
		return []


if __name__ == "__main__":
	nas = iterative_naive_nas()

	#print(nas.generate_layer_Conv3DTranspose((10, 10, 10, 1), (30, 30, 30, 1)))
	#print(nas.generate_kernel_stride_Conv2D((10, 10, 1), (5, 5, 1)))

	model = tf.keras.Sequential()
	model.add(nas.build_layer_Conv3DTranspose((5, 5, 5, 1), (10, 10, 10, 1)))
	print(model.summary())