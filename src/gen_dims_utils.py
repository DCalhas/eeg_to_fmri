from tensorflow.python.keras.utils import conv_utils

import numpy as np

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
			if(new_dim == output_shape and kernel >= stride):
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
				dilation=1)

			#only accept dim = output desired and that are bigger than half the 
			if(new_dim == output_shape and kernel >= stride):
				possible_combinations += [(kernel, stride)]

	return possible_combinations

def add_generated_dim(generated, input_shape, output_shape, func):
	possible = func(input_shape, output_shape)

	pos = list(range(len(possible)))
	generated_kernel_stride = possible[np.random.choice(pos)]
	
	generated['kernel'] += (generated_kernel_stride[0],)
	generated['stride'] += (generated_kernel_stride[1],)

	return generated