from tensorflow.python.keras.utils import conv_utils

import numpy as np

def get_possible_kernel_size_conv(input_shape, output_shape):
	#list of tuples where #1 is kernel size and #2 is stride size
	possible_combinations = []

	for stride in range(1, input_shape):
		for kernel in range(1, input_shape):

			#new_dim = conv_utils.conv_output_length(input_shape, 
				#kernel, 
				#padding='valid', 
				#stride=stride)

			#only accept dim = output desired and that are bigger than half the 
			if( ((input_shape - kernel + stride) // stride) == output_shape and 
				kernel >= stride):
				possible_combinations += [(kernel, stride)]
				#fix this, it takes to much time trying it for all the points
				#wrapper to return first valid kernel and stride combination
				return possible_combinations

	return possible_combinations

def get_possible_kernel_size_deconv(input_shape, output_shape, next_input_shape=None):
	#list of tuples where #1 is kernel size and #2 is stride size
	possible_combinations = []


	##################################################################################################################
	#
	#							CASE WHERE WE DIMENSION SHRINKS BY ONE WITH RESHAPE AFTERWARDS
	#
	##################################################################################################################
	if(type(input_shape) is tuple and next_input_shape == None):
		for stride_0 in range(1, output_shape):
			for kernel_0 in range(1, output_shape):
				for stride_1 in range(1, output_shape):
					for kernel_1 in range(1, output_shape):
						#only accept dim = output desired and that are bigger than half the 
						if((input_shape[0] * stride_0 + max(kernel_0 - stride_0, 0))*(input_shape[1] * stride_1 + max(kernel_1 - stride_1, 0)) == output_shape and 
							(kernel_0 >= stride_0 and kernel_1 >= stride_1)):
							possible_combinations += [((kernel_0, kernel_1), (stride_0, stride_1))]
							#fix this, it takes to much time trying it for all the points
							#wrapper to return first valid kernel and stride combination
							return possible_combinations

	elif(type(input_shape) is tuple and next_input_shape != None):
		for stride_0 in range(1, output_shape):
			for kernel_0 in range(1, output_shape):
				if(input_shape[0] * stride_0 + max(kernel_0 - stride_0, 0) <= next_input_shape[0]):
					for stride_1 in range(1, output_shape):
						for kernel_1 in range(1, output_shape):
							#only accept dim = output desired and that are bigger than half the 
							if((input_shape[0] * stride_0 + max(kernel_0 - stride_0, 0))*(input_shape[1] * stride_1 + max(kernel_1 - stride_1, 0)) == output_shape and 
								#(kernel_0 >= stride_0 and kernel_1 >= stride_1) and 
								input_shape[1] * stride_1 + max(kernel_1 - stride_1, 0) <= next_input_shape[1]):
							
								possible_combinations += [((kernel_0, kernel_1), (stride_0, stride_1))]
								#fix this, it takes to much time trying it for all the points
								#wrapper to return first valid kernel and stride combination
								return possible_combinations

	else:
		for stride in range(1, output_shape):
			for kernel in range(1, output_shape):

				#only accept dim = output desired and that are bigger than half the 
				if(input_shape * stride + max(kernel - stride, 0) == output_shape and kernel >= stride):
					possible_combinations += [(kernel, stride)]
					#fix this, it takes to much time trying it for all the points
					#wrapper to return first valid kernel and stride combination
					return possible_combinations

	return possible_combinations

def add_generated_dim(generated, input_shape, output_shape, func):
	possible = func(input_shape, output_shape)

	pos = list(range(len(possible)))
	generated_kernel_stride = possible[np.random.choice(pos)]
	
	generated['kernel'] += (generated_kernel_stride[0],)
	generated['stride'] += (generated_kernel_stride[1],)

	return generated