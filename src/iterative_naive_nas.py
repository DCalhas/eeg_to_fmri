import numpy as np

import math

import tensorflow.compat.v1 as tf
from tensorflow.python.keras.utils import conv_utils

import gen_dims_utils

import bayesian_optimization


layers = [tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose, 
								tf.keras.layers.Conv3D, tf.keras.layers.Conv3DTranspose]



class Multi_Modal_Model:

	def __init__(self, eeg_encoder, bold_encoder, decoder):
		self.eeg_encoder = eeg_encoder
		self.bold_encoder = bold_encoder
		self.decoder = decoder

	def get_level(self):
		return np.amax(np.array([self.eeg_encoder.get_depth(), self.bold_encoder.get_depth(), self.decoder.get_depth()]))

	def build_eeg(self, input_shape, output_shape):
		print("BUIDLING EEG ENCODER", output_shape)

		return self.eeg_encoder.build_net(input_shape, output_shape, verbose=True)

	def build_bold(self, input_shape, output_shape):
		print("BUIDLING BOLD ENCODER", output_shape)

		return self.bold_encoder.build_net(input_shape, output_shape, verbose=True)

	def build_decoder(self, input_shape, output_shape):
		print("BUIDLING DECODER", output_shape)

		return self.decoder.build_net(input_shape, output_shape, verbose=True)

	#######################################################################################################################
	#
	#							OUTPUT DIMENSION OF NEW LAYERS ARE SUBJECT TO OPTIMIZATION HERE
	#
	#######################################################################################################################
	def BO(self):
		print(self.eeg_encoder.get_layers())
		print(self.bold_encoder.get_layers())
		print(self.decoder.get_layers())


		domain = []

		dilation_factor = 3

		print(self.eeg_encoder, self.bold_encoder)

		#DEFINE NEW SHAPE DOMAIN - FIRST LEVEL DOMAIN
		if(self.get_level() == 1):
			if(self.eeg_encoder.get_layers()[0].__name__ == "build_layer_Conv3DTranspose"):
				for i in range(int(64*5), int(64*5)*dilation_factor, 10):
					domain += [i]
			else:
				for i in range(10, int(64*5), 10):
					domain += [i]

			output_shape_domain = {'name': 'shape_domain', 'type': 'discrete',
			'domain': tuple(domain)}

		#DEFINE NEW SHAPE DOMAIN - SECOND LEVEL DOMAIN - DOMAIN FOR ENCODERS AND DECODER SEPARATE
		else:
			print("BUILD DOMAINS FOR EACH BRANCH OF THE NETWORK")

			eeg_domain = self.eeg_encoder.get_hidden_domain()
			bold_domain = self.bold_encoder.get_hidden_domain()
			decoder_domain = self.decoder.get_hidden_domain()

		new_output_shape, loss = bayesian_optimization.NAS_BO(self, [output_shape_domain])

		new_output_shape = (int(new_output_shape), 20, 1)

		print(self.eeg_encoder, self.bold_encoder)
		self.eeg_encoder.add_output_shape(new_output_shape)
		print(self.eeg_encoder.get_output_shapes())
		self.bold_encoder.add_output_shape(new_output_shape)
		print(self.bold_encoder.get_output_shapes())
		#self.decoder.add_output_shape(new_output_shape)

		print("RUNNING BO")
		return loss


class Neural_Architecture:

	def __init__(self, layers=[], output_shapes=[]):
		self.layers = layers
		self.output_shapes = output_shapes

	#add new layer to list of layers
	def add_layer(self, layer):
		return []

	def add_output_shape(self, output_shape):
		self.output_shapes += [output_shape]

	def get_output_shapes(self):
		return self.output_shapes

	def get_layers(self):
		return self.layers

	def get_last_layer(self):
		return self.layers[-1]

	def get_depth(self):
		return len(self.get_layers())

	def possible_next_layers(self):
		last_layer = self.get_last_layer()
		#0-Dense
		#1-Conv2D
		#2-Conv2DTranspose
		#3-Conv3D
		#4-Conv3DTranspose


		if(last_layer.__name__ == "build_layer_Conv3D"):
			return [1, 3]
		elif(last_layer.__name__ == "build_layer_Conv3DTranspose"):
			return [4]
		elif(last_layer.__name__ == "build_layer_Conv2DTranspose"):
			return [2]
		elif(last_layer.__name__ == "build_layer_Conv2D"):
			return [1]
		else:
			return [0]

	#remove last layer
	def remove_last_layer(self):
		return []


	#######################################################################################################################
	#
	#							GET DOMAIN TO BE EXPLORED BY BO FOR HIDDEN LAYER OUTPUT DIMENSION
	#
	#######################################################################################################################
	def get_hidden_domain(self, dilation_factor=3):
		domain = []

		print(self.output_shapes)
		print(self.get_layers())

		if(self.get_layers()[0].__name__ == "build_layer_Conv3DTranspose"):
			for i in range(int(64*5), int(64*5)*dilation_factor, 10):
				domain += [i]
		else:
			for i in range(10, int(64*5), 10):
				domain += [i]

		return {'name': 'eeg_shape_domain', 'type': 'discrete', 
								'domain': tuple(domain)}

	#build sequential model with the layers
	def build_net(self, input_shape, hidden_output_shape, verbose=False):
		model = tf.keras.Sequential()

		model.add(self.get_layers()[0](input_shape, hidden_output_shape))

		hidden_input_shape = hidden_output_shape

		for layer in range(len(self.get_layers()[1:])):
			model.add(self.get_layers()[layer](hidden_input_shape, self.output_shapes[layer]))
			hidden_input_shape = self.output_shapes[layer]

		#we know we want to go to 2D space, so if the EEG branch is still with 3D, reshape is needed
		if(len(hidden_input_shape) == 3 and len(input_shape) == 4):
			model.add(tf.keras.layers.Reshape(hidden_input_shape))


		model.build(input_shape=input_shape)

		if(verbose):
			print(model.summary())

		return model
		

class Iterative_Naive_NAS:


	def __init__(self):
		self.iterations = 0
		self.tested_architectures = {}
		self.best_loss = math.inf
		self.best_depth = 0
		self.improved = True


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
		if(len(output_shape) + 1 == len(input_shape)):
			possible = gen_dims_utils.get_possible_kernel_size_deconv((input_shape[0], input_shape[1]), output_shape[0])

			pos = list(range(len(possible)))
			
			generated_kernel_stride = possible[np.random.choice(pos)]

			return {'kernel': generated_kernel_stride[0],
					'stride': generated_kernel_stride[1]}

		else:
			possible = gen_dims_utils.get_possible_kernel_size_deconv(input_shape[0], output_shape[0])

			pos = list(range(len(possible)))
			
			generated_kernel_stride = possible[np.random.choice(pos)]

			return {'kernel': (generated_kernel_stride[0],),
					'stride': (generated_kernel_stride[1],)}


	def generate_kernel_stride_Conv2D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv1D((input_shape[0],1), (output_shape[0], 1))

		return gen_dims_utils.add_generated_dim(generated, input_shape[1], output_shape[1], gen_dims_utils.get_possible_kernel_size_conv)


	def generate_kernel_stride_Conv2DTranspose(self, input_shape, output_shape):
		if(len(output_shape) + 1 == len(input_shape)):
			return self.generate_kernel_stride_Conv1DTranspose(input_shape, output_shape)

		generated = self.generate_kernel_stride_Conv1DTranspose((input_shape[0],1), (output_shape[0], 1))

		return gen_dims_utils.add_generated_dim(generated, input_shape[1], output_shape[1], gen_dims_utils.get_possible_kernel_size_deconv)


	def generate_kernel_stride_Conv3D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv2D((input_shape[0], input_shape[1],1), (output_shape[0], output_shape[1], 1))

		return gen_dims_utils.add_generated_dim(generated, input_shape[2], output_shape[2], gen_dims_utils.get_possible_kernel_size_conv)


	def generate_kernel_stride_Conv3DTranspose(self, input_shape, output_shape):
		if(len(output_shape) + 1 == len(input_shape)):

			generated = self.generate_kernel_stride_Conv2DTranspose((input_shape[0], input_shape[1], 1), (output_shape[0], 1))
			
			return gen_dims_utils.add_generated_dim(generated, input_shape[2], output_shape[1], gen_dims_utils.get_possible_kernel_size_deconv)

		generated = self.generate_kernel_stride_Conv2DTranspose((input_shape[0], input_shape[1], 1), (output_shape[0], output_shape[1], 1))
		
		return gen_dims_utils.add_generated_dim(generated, input_shape[2], output_shape[2], gen_dims_utils.get_possible_kernel_size_deconv)


	######################################################################################################################################
	#
	#															Build Layer Functions
	#
	######################################################################################################################################

	#Dense layer is simple to generate
	def build_layer_Dense(self, input_shape, output_shape):
		if(type(output_shape) is not int and type(output_shape) is not tuple):
			return None

		shape = 1
		for dim in output_shape:
			shape *= dim

		print(shape)

		return layers[0](shape, input_shape=input_shape), tf.keras.layers.Reshape(output_shape)


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
	#									Types of layers are added and subjected to Bayesian Optimization
	#
	######################################################################################################################################


	def search(self, eeg_input_shape, bold_input_shape):

		#queue of Neural_Architecture instances || None is initial state
		eeg_queue = [None]
		bold_queue = [None]
		decoder_queue = [None]


		#for eeg but called core for global interpretation
		core_layers = [self.build_layer_Dense, self.build_layer_Conv2D, self.build_layer_Conv2DTranspose, 
						self.build_layer_Conv3D, self.build_layer_Conv3DTranspose]

		bold_layers = [self.build_layer_Conv2D]

		decoder_layers = [self.build_layer_Conv2DTranspose]

		while(len(eeg_queue) and len(bold_queue) and len(decoder_queue) and self.improved):

			#INITAL CASE
			if(eeg_queue[-1] == None and bold_queue[-1] == None and decoder_queue[-1] == None):
				#generate the first layers/level
				del eeg_queue[-1]
				del bold_queue[-1]
				del decoder_queue[-1]


				for layer in [core_layers[4]]:#[core_layers[3], core_layers[4]]:
					eeg_queue += [Neural_Architecture(layers=[layer])]

				for layer in bold_layers:
					bold_queue += [Neural_Architecture(layers=[layer])]

				for layer in decoder_layers:
					decoder_queue += [Neural_Architecture(layers=[layer])]

				self.best_depth = 1
				self.improved = True

			#stop search condition
			if(eeg_queue[-1].get_depth() > self.best_depth + 1):
				break

			#DEEP LEVEL CASE
			else:
				last_element_eeg = eeg_queue[-1]
				last_element_bold = bold_queue[-1]
				last_element_decoder = decoder_queue[-1]

				#optimize last_architecture
				synthesizer = Multi_Modal_Model(last_element_eeg, last_element_bold, last_element_decoder)
				val_loss = synthesizer.BO()
				self.tested_architectures[synthesizer] = val_loss
				if(val_loss < self.best_loss):
					self.best_loss = val_loss
					self.best_depth +=1
					self.improved=True



				possible_layers = eeg_queue[-1].possible_next_layers()

				del eeg_queue[-1]

				for layer in possible_layers:
					new_architecture = Neural_Architecture(layers=[core_layers[layer]] + last_element_eeg.get_layers(), 
						output_shapes=last_element_eeg.get_output_shapes())

				eeg_queue = eeg_queue + [new_architecture]



				possible_layers = bold_queue[-1].possible_next_layers()

				del bold_queue[-1]

				for layer in possible_layers:
					new_architecture = Neural_Architecture(layers=[core_layers[layer]] + last_element_bold.get_layers(), 
						output_shapes=last_element_bold.get_output_shapes())

				bold_queue = bold_queue + [new_architecture]



				possible_layers = decoder_queue[-1].possible_next_layers()

				del decoder_queue[-1]

				for layer in possible_layers:
					new_architecture = Neural_Architecture(layers=[core_layers[layer]] + last_element_decoder.get_layers(), 
						output_shapes=last_element_decoder.get_output_shapes())

				decoder_queue = decoder_queue + [new_architecture]

					
			#call BO to optimize the output_shape for each layer at this level
			print(self.improved)
			print(len(eeg_queue) and len(bold_queue) and len(decoder_queue) and self.improved)

		print("FINISHED NAS")

		return []


if __name__ == "__main__":
	nas = Iterative_Naive_NAS()

	eeg_input_shape = (64, 5, 20, 1)
	bold_input_shape = (14000, 20, 1)
	nas.search(eeg_input_shape, bold_input_shape)
	#print(nas.generate_layer_Conv3DTranspose((10, 10, 10, 1), (30, 30, 30, 1)))
	#print(nas.generate_kernel_stride_Conv2D((10, 10, 1), (5, 5, 1)))

	#model = tf.keras.Sequential()
	#model.add(nas.build_layer_Conv3DTranspose((5, 5, 20, 1), (35, 35, 20, 1)))
	#model.add(tf.keras.layers.Reshape((35, 20, 1)))
	#print(model.summary())