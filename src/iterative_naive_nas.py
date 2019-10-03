import tensorflow.compat.v1 as tf

import numpy as np

from tensorflow.python.keras.utils import conv_utils


layers = [tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose]


def get_possible_kernel_size_conv(input_shape, output_shape):
	#list of tuples where #1 is kernel size and #2 is stride size
	possible_combinations = []

	for stride in range(1, input_shape):
		for kernel in range(1, input_shape):

			new_dim = conv_utils.conv_output_length(input_shape, 
				kernel, 
				padding='valid', 
				stride=stride)

			#only accept dim = output desired and that are bigger than half the 
			if(new_dim == output_shape and kernel > stride):
				possible_combinations += [(kernel, stride)]

	return possible_combinations

def get_possible_kernel_size_deconv(input_shape, output_shape):
	#list of tuples where #1 is kernel size and #2 is stride size
	possible_combinations = []

	for stride in range(1, input_shape):
		for kernel in range(1, input_shape):

			new_dim = conv_utils.deconv_output_length(input_shape, 
				kernel, 
				padding='valid', 
				output_padding=None, 
				stride=stride, 
				dilation=0)

			#only accept dim = output desired and that are bigger than half the 
			if(new_dim == output_shape and kernel > stride):
				possible_combinations += [(kernel, stride)]

	return possible_combinations


class iterative_naive_nas:


	def __init__(self):
		self.iterations = 0


	#Dense layer is simple to generate
	def generate_layer_Dense(self, input_shape, output_shape):
		if(type(output_shape) is not int):
			return None
		return layers[0](output_shape, input_shape=input_shape)

	def generate_layer_Conv1D(self, input_shape, output_shape):
		possible = get_possible_kernel_size_conv(input_shape[0], output_shape[0])

		pos = list(range(len(possible)))
		generated_kernel_stride = possible[np.random.choice(pos)]

		return {'kernel': (generated_kernel_stride[0],),
				'stride': (generated_kernel_stride[1],)}

	def generate_layer_Conv1DTranspose(self, input_shape, output_shape):
		return None

	def generate_layer_Conv2D(self, input_shape, output_shape):
		generated = self.generate_layer_Conv1D((input_shape[0],1), (output_shape[0], 1))

		possible = get_possible_kernel_size_conv(input_shape[1], output_shape[1])

		pos = list(range(len(possible)))
		generated_kernel_stride = possible[np.random.choice(pos)]

		generated['kernel'] += (generated_kernel_stride[0],)
		generated['stride'] += (generated_kernel_stride[1],)

		return generated

	def generate_layer_Conv2DTranspose(self, input_shape, output_shape):
		return None

	def generate_layer_Conv3D(self, input_shape, output_shape):
		generated = self.generate_layer_Conv2D((input_shape[0], input_shape[1],1), (output_shape[0], output_shape[1], 1))

		possible = get_possible_kernel_size_conv(input_shape[2], output_shape[2])

		pos = list(range(len(possible)))
		generated_kernel_stride = possible[np.random.choice(pos)]

		generated['kernel'] += (generated_kernel_stride[0],)
		generated['stride'] += (generated_kernel_stride[1],)

		return generated

	def generate_layer_Conv3DTranspose(self, input_shape, output_shape):
		return None

	def search(self):
		return []


if __name__ == "__main__":
	nas = iterative_naive_nas()

	print(nas.generate_layer_Conv3D((10, 10, 10, 1), (2, 2, 3, 1)))