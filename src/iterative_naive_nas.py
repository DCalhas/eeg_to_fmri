import numpy as np

import math

import tensorflow.compat.v1 as tf
from tensorflow.python.keras.utils import conv_utils

import gen_dims_utils

import bayesian_optimization


layers = [tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose, 
								tf.keras.layers.Conv3D, tf.keras.layers.Conv3DTranspose, 
								tf.keras.layers.UpSampling2D, tf.keras.layers.UpSampling3D]



class Multi_Modal_Model:

	def __init__(self, eeg_encoder, bold_encoder, decoder, previous_eeg_network, previous_bold_network, previous_decoder_network):
		self.eeg_encoder = eeg_encoder
		self.bold_encoder = bold_encoder
		self.decoder = decoder

		self.previous_eeg_network = previous_eeg_network
		self.previous_bold_network = previous_bold_network
		self.previous_decoder_network = previous_decoder_network

	def get_level(self):
		return np.amax(np.array([self.eeg_encoder.get_depth(), self.bold_encoder.get_depth(), self.decoder.get_depth()]))

	def save_eeg(self, eeg_network):
		self.previous_eeg_network = eeg_network
		real_output_shape = self.previous_eeg_network.layers[0].output_shape
		self.eeg_encoder.add_real_output_shape((real_output_shape[1], 
												real_output_shape[2], 
												real_output_shape[3], 
												real_output_shape[4]))

	def save_bold(self, bold_network):
		self.previous_bold_network = bold_network
		real_output_shape = self.previous_bold_network.layers[0].output_shape
		self.bold_encoder.add_real_output_shape((real_output_shape[1], 
												real_output_shape[2], 
												real_output_shape[3]))

	def save_decoder(self, decoder_network):
		self.previous_decoder_network = decoder_network
		real_output_shape = self.previous_decoder_network.layers[0].output_shape
		self.decoder.add_real_output_shape((real_output_shape[1], 
												real_output_shape[2], 
												real_output_shape[3]))

	def build_eeg(self, input_shape, output_shape):

		return self.eeg_encoder.build_net(input_shape, output_shape, previous_model=self.previous_eeg_network, verbose=True)

	def build_bold(self, input_shape, output_shape):

		return self.bold_encoder.build_net(input_shape, output_shape, previous_model=self.previous_bold_network, verbose=True)

	def build_decoder(self, input_shape, output_shape):

		return self.decoder.build_net(input_shape, output_shape, previous_model=self.previous_decoder_network, verbose=True)

	#######################################################################################################################
	#
	#							OUTPUT DIMENSION OF NEW LAYERS ARE SUBJECT TO OPTIMIZATION HERE
	#
	#######################################################################################################################
	def BO(self):

		domain = []

		dilation_factor = 3


		#DEFINE NEW SHAPE DOMAIN - FIRST LEVEL DOMAIN
		if(self.get_level() == 1):
			for i in range(int(64*5)+100, 1000, 500):
				domain += [i]

			output_shape_domain = {'name': 'shape_domain', 'type': 'discrete',
			'domain': tuple(domain)}

			new_output_shape, loss = bayesian_optimization.NAS_BO(self, [output_shape_domain])

			new_output_shape = (int(new_output_shape), 20, 1)

			self.eeg_encoder.add_output_shape(new_output_shape)
			self.bold_encoder.add_output_shape(new_output_shape)
			self.decoder.add_output_shape(new_output_shape)


		#DEFINE NEW SHAPE DOMAIN - SECOND LEVEL DOMAIN - DOMAIN FOR ENCODERS AND DECODER SEPARATE
		else:
			print("BUILD DOMAINS FOR EACH BRANCH OF THE NETWORK")

			eeg_domain = self.eeg_encoder.get_hidden_domain()
			bold_domain = self.bold_encoder.get_hidden_domain()
			decoder_domain = self.decoder.get_hidden_domain()

			eeg_new_hidden_shape, bold_new_hidden_shape, decoder_new_hidden_shape, loss = bayesian_optimization.hidden_layer_NAS_BO(self, 
																								eeg_domain, bold_domain, decoder_domain)

			#optimization finishes if domains are empty
			if(not (eeg_new_hidden_shape and bold_new_hidden_shape and decoder_new_hidden_shape)):
				return math.inf

			eeg_new_hidden_shape = (int(eeg_new_hidden_shape), 20, 1)
			bold_new_hidden_shape = (int(bold_new_hidden_shape), 20, 1)
			decoder_new_hidden_shape = (int(decoder_new_hidden_shape), 20, 1)

			self.eeg_encoder.add_output_shape(eeg_new_hidden_shape)
			self.bold_encoder.add_output_shape(bold_new_hidden_shape)
			self.decoder.add_output_shape(decoder_new_hidden_shape)

		print("RUNNING BO")
		return loss


class Neural_Architecture:

	def __init__(self, layers=[], output_shapes=[], real_output_shapes=[]):
		self.layers = layers
		self.output_shapes = output_shapes
		self.real_output_shapes = real_output_shapes

	def get_output_shapes(self):
		return self.output_shapes

	def get_real_output_shapes(self):
		return self.real_output_shapes

	def get_layers(self):
		return self.layers

	def get_last_layer(self):
		return self.layers[-1]

	def get_depth(self):
		return len(self.get_layers())

	#add new layer to list of layers
	def add_layer(self, layer):
		return []

	def add_output_shape(self, output_shape):
		self.output_shapes += [output_shape]

	def add_real_output_shape(self, real_output_shape):
		self.real_output_shapes += [real_output_shape]

	def possible_next_layers(self):
		last_layer = self.get_last_layer()
		#0-Dense
		#1-Conv2D
		#2-Conv2DTranspose
		#3-Conv3D
		#4-Conv3DTranspose14164


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

		#dilation factor in order to recontruct midlayer
		if(self.get_layers()[0].__name__ == "build_layer_Conv2DTranspose"):
			print(self.get_real_output_shapes())
			print("\n\n\n\n\n\n")
			for i in range(self.get_output_shapes()[-1][0], self.get_real_output_shapes()[0][0], 10):
				domain += [i]
		elif(self.get_layers()[0].__name__ == "build_layer_Conv2D"):
			for i in range(self.get_output_shapes()[-1][0], 14164, 10):
				domain += [i]
		else:
			for i in range(int(64*5), self.get_output_shapes()[-1][0], 10):
				domain += [i]

		return {'name': 'eeg_shape_domain', 'type': 'discrete', 
								'domain': tuple(domain)}


	#######################################################################################################################
	#
	#											BUILDING TENSORFLOW INSTANCE NETWORK
	#
	#######################################################################################################################
	def build_net(self, input_shape, hidden_output_shape, previous_model=None, verbose=False):
		model = tf.keras.Sequential()

		try:

			_layers = []

			if(len(self.get_layers()) > 1 and self.get_layers()[0].__name__ == "build_layer_Conv3DTranspose"):
				_layers += self.get_layers()[0](input_shape, hidden_output_shape, next_input_shape=self.get_real_output_shapes()[-1])

				if(not _layers):
					return None

				hidden_input_shape = hidden_output_shape
				hidden_input_shape = (hidden_input_shape[1], hidden_input_shape[2], hidden_input_shape[3], hidden_input_shape[4])

			elif(len(self.get_layers()) > 1 and self.get_layers()[0].__name__ == "build_layer_Conv2DTranspose"):
				if(len(self.get_layers()) == 2):
					_layers += self.get_layers()[0](self.get_output_shapes()[0], input_shape)

					if(not _layers):
						return None

					hidden_input_shape = input_shape

				elif(len(self.get_layers()) > 2):
					_layers += self.get_layers()[0](self.get_output_shapes()[0], self.get_real_output_shapes()[-1])

					if(not _layers):
						return None

					hidden_input_shape = input_shape

			else:
				_layers += self.get_layers()[0](input_shape, hidden_output_shape)

				if(not _layers):
					return None

				hidden_input_shape = hidden_output_shape

			ros = -1

			if(len(self.get_layers()) > 1):
				for layer in self.get_layers()[1:]:
					if(layer.__name__ == "build_layer_Conv3DTranspose"):
						_layers += layer(hidden_input_shape, self.get_output_shapes()[ros], next_input_shape=self.get_real_output_shapes()[ros])

						ros -= 1

						if(not _layers):
							return None
						
						hidden_input_shape = self.get_output_shapes()[ros]
						hidden_input_shape = (hidden_input_shape[1], hidden_input_shape[2], hidden_input_shape[3], hidden_input_shape[4])

					elif(layer.__name__ == "build_layer_Conv2DTranspose" or layer.__name__ == "build_layer_UpSampling2D"):

						if(abs(ros) == len(self.get_real_output_shapes())-1 and len(self.get_real_output_shapes()) >= 2):
							_layers += layer(self.get_real_output_shapes()[ros], hidden_input_shape)

							if(not _layers):
								return None

							_layers += layer(hidden_input_shape, hidden_output_shape)

							if(not _layers):
								return None

							#stop model building
							break

						elif(len(self.get_real_output_shapes()) >= 2):
							_layers += layer(self.get_real_output_shapes()[ros], self.get_real_output_shapes()[ros-1])
							hidden_input_shape = self.get_real_output_shapes()[ros-1]
						else:
							_layers += layer(hidden_input_shape, self.get_real_output_shapes()[ros])

							hidden_input_shape = self.get_real_output_shapes()[ros]


						if(not layer):
							return None

						hidden_input_shape = self.get_real_output_shapes()[ros]
						ros -= 1

					else:
						_layers += layer(hidden_input_shape, self.get_output_shapes()[ros])

						if(not _layers):
							return None

						hidden_input_shape = self.get_output_shapes()[ros]
						ros -= 1

				hidden_input_shape = self.get_output_shapes()[ros+1]

			for l in _layers:
				model.add(l)

			if(len(hidden_input_shape) == 3 and len(input_shape) == 4):
				model.add(tf.keras.layers.Reshape(hidden_input_shape))

			model.build(input_shape=input_shape)

		#fix this try and except, so it runs how it is supposed
		except:
			print("An exception occured - Not specified which one - layer type: ", self.get_layers()[0].__name__, " with input - output",
					input_shape, "-", hidden_output_shape)
			return None

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
		
		if(not len(pos)):
			return None

		generated_kernel_stride = possible[np.random.choice(pos)]

		return {'kernel': (generated_kernel_stride[0],),
				'stride': (generated_kernel_stride[1],)}


	def generate_kernel_stride_Conv1DTranspose(self, input_shape, output_shape, next_input_shape=None):
		if(len(output_shape) + 1 == len(input_shape) and next_input_shape == None):
			possible = gen_dims_utils.get_possible_kernel_size_deconv((input_shape[0], input_shape[1]), output_shape[0])

			pos = list(range(len(possible)))

			if(not len(pos)):
				return None
			
			generated_kernel_stride = possible[np.random.choice(pos)]

			return {'kernel': generated_kernel_stride[0],
					'stride': generated_kernel_stride[1]}

		elif(len(output_shape) + 1 == len(input_shape) and next_input_shape != None):

			possible = gen_dims_utils.get_possible_kernel_size_deconv((input_shape[0], input_shape[1]), output_shape[0], next_input_shape=next_input_shape)

			pos = list(range(len(possible)))

			if(not len(pos)):
				return None
			
			generated_kernel_stride = possible[np.random.choice(pos)]

			return {'kernel': generated_kernel_stride[0],
					'stride': generated_kernel_stride[1]}

		else:
			possible = gen_dims_utils.get_possible_kernel_size_deconv(input_shape[0], output_shape[0])

			pos = list(range(len(possible)))

			if(not len(pos)):
				return None
			
			generated_kernel_stride = possible[np.random.choice(pos)]

			return {'kernel': (generated_kernel_stride[0],),
					'stride': (generated_kernel_stride[1],)}


	def generate_kernel_stride_Conv2D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv1D((input_shape[0],1), (output_shape[0], 1))

		if(not generated):
			return None

		return gen_dims_utils.add_generated_dim(generated, input_shape[1], output_shape[1], gen_dims_utils.get_possible_kernel_size_conv)


	def generate_kernel_stride_Conv2DTranspose(self, input_shape, output_shape, next_input_shape=None):
		if(len(output_shape) + 1 == len(input_shape) and next_input_shape == None):
			return self.generate_kernel_stride_Conv1DTranspose(input_shape, output_shape)

		if(len(output_shape) + 1 == len(input_shape) and next_input_shape != None):
			return self.generate_kernel_stride_Conv1DTranspose(input_shape, output_shape, next_input_shape=next_input_shape)

		generated = self.generate_kernel_stride_Conv1DTranspose((input_shape[0],1), (output_shape[0], 1))

		if(not generated):
			return None

		return gen_dims_utils.add_generated_dim(generated, input_shape[1], output_shape[1], gen_dims_utils.get_possible_kernel_size_deconv)


	def generate_kernel_stride_Conv3D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv2D((input_shape[0], input_shape[1],1), (output_shape[0], output_shape[1], 1))

		if(not generated):
			return None

		return gen_dims_utils.add_generated_dim(generated, input_shape[2], output_shape[2], gen_dims_utils.get_possible_kernel_size_conv)


	def generate_kernel_stride_Conv3DTranspose(self, input_shape, output_shape, next_input_shape=None):
		if(len(output_shape) + 1 == len(input_shape) and next_input_shape == None):

			generated = self.generate_kernel_stride_Conv2DTranspose((input_shape[0], input_shape[1], 1), (output_shape[0], 1))

			if(not generated):
				return None
			
			return gen_dims_utils.add_generated_dim(generated, input_shape[2], output_shape[1], gen_dims_utils.get_possible_kernel_size_deconv)

		elif(len(output_shape) + 1 == len(input_shape) and next_input_shape != None):
			generated = self.generate_kernel_stride_Conv2DTranspose((input_shape[0], input_shape[1], 1), (output_shape[0], 1), 
																	next_input_shape=next_input_shape)

			if(not generated):
				return None
			
			return gen_dims_utils.add_generated_dim(generated, input_shape[2], output_shape[1], gen_dims_utils.get_possible_kernel_size_deconv)

		generated = self.generate_kernel_stride_Conv2DTranspose((input_shape[0], input_shape[1], 1), (output_shape[0], output_shape[1], 1))
		
		if(not generated):
			return None

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

		return [layers[0](shape, input_shape=input_shape), tf.keras.layers.Reshape(output_shape)]


	def build_layer_Conv2D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv2D(input_shape, output_shape)

		if(not generated):
			return None

		return [layers[1](1, kernel_size=generated['kernel'], strides=generated['stride'], input_shape=input_shape)]


	def build_layer_Conv2DTranspose(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv2DTranspose(input_shape, output_shape)

		if(not generated):
			return None

		return [layers[2](1, kernel_size=generated['kernel'],
									strides=generated['stride'], 
									padding='valid', 
									output_padding=None,
									dilation_rate=(1, 1),
									input_shape=input_shape)]


	#returns a list of layers
	def build_layer_UpSampling2D(self, input_shape, output_shape):
		#augment dimension with Dense layer
		upsampling_layers = []		

		dilation_factor = (output_shape[0]*output_shape[1]) // (input_shape[0]*input_shape[1])

		dilation_factor_x = (output_shape[0]) // (input_shape[0])

		dilation_factor_y = (output_shape[1]) // (input_shape[1])

		upsampling_layers += [tf.keras.layers.Reshape((input_shape[0]*input_shape[1],), input_shape=input_shape)]

		upsampling_layers += [layers[0](input_shape[0]*input_shape[1] * dilation_factor)]

		upsampling_layers += [tf.keras.layers.Reshape( (input_shape[0], input_shape[1], dilation_factor) )]

		upsampling_layers += [tf.keras.layers.UpSampling2D( (dilation_factor_x, dilation_factor_y) )]
		
		#we need to perform padding to match the output shape desired
		#for now Zero_padding

		diff_x = output_shape[0] - input_shape[0] * dilation_factor_x
		diff_y = output_shape[1] - input_shape[1] * dilation_factor_y

		if(diff_x != 0 or diff_y != 0):
			upsampling_layers += [tf.keras.layers.ZeroPadding2D(padding=((0, diff_x), (0, diff_y)))]



		upsampling_layers += [tf.keras.layers.Conv2D(1, (3,3), padding='same')]

		return upsampling_layers


	def build_layer_Conv3D(self, input_shape, output_shape):
		generated = self.generate_kernel_stride_Conv3D(input_shape, output_shape)

		if(not generated):
			return None

		return [layers[3](1, kernel_size=generated['kernel'], strides=generated['stride'], input_shape=input_shape)]


	def build_layer_Conv3DTranspose(self, input_shape, output_shape, next_input_shape=None):
		generated = self.generate_kernel_stride_Conv3DTranspose(input_shape, output_shape, next_input_shape=next_input_shape)

		if(not generated):
			return None

		return [layers[4](1, kernel_size=generated['kernel'],
									strides=generated['stride'], 
									padding='valid', 
									output_padding=None,
									dilation_rate=(1, 1, 1),
									input_shape=input_shape)]


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
		#ecoder_layers = [self.build_layer_UpSampling2D]

		synthesizer = Multi_Modal_Model(None, None, None, None, None, None)

		while(len(eeg_queue) and len(bold_queue) and len(decoder_queue) and self.improved):

			#INITAL CASE
			if(eeg_queue[-1] == None and bold_queue[-1] == None and decoder_queue[-1] == None):
				#generate the first layers/level
				del eeg_queue[-1]
				del bold_queue[-1]
				del decoder_queue[-1]


				for layer in [core_layers[4]]:#[core_layers[3], core_layers[4]]:
					eeg_queue += [Neural_Architecture(layers=[layer], output_shapes=[], real_output_shapes=[])]

				for layer in bold_layers:
					bold_queue += [Neural_Architecture(layers=[layer], output_shapes=[], real_output_shapes=[])]

				for layer in decoder_layers:
					decoder_queue += [Neural_Architecture(layers=[layer], output_shapes=[], real_output_shapes=[])]

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
				synthesizer.eeg_encoder = last_element_eeg
				synthesizer.bold_encoder = last_element_bold
				synthesizer.decoder = last_element_decoder
				#synthesizer = Multi_Modal_Model(last_element_eeg, last_element_bold, last_element_decoder)
				val_loss = synthesizer.BO()
				self.tested_architectures[synthesizer] = val_loss
				if(val_loss <= self.best_loss):
					self.best_loss = val_loss
					self.best_depth +=1
					self.improved=True



				possible_layers = eeg_queue[-1].possible_next_layers()

				del eeg_queue[-1]

				for layer in possible_layers:
					new_architecture = Neural_Architecture(layers=[core_layers[layer]] + last_element_eeg.get_layers(), 
						output_shapes=last_element_eeg.get_output_shapes(), real_output_shapes=last_element_eeg.get_real_output_shapes())

				eeg_queue = eeg_queue + [new_architecture]



				possible_layers = bold_queue[-1].possible_next_layers()

				del bold_queue[-1]

				for layer in possible_layers:
					new_architecture = Neural_Architecture(layers=[core_layers[layer]] + last_element_bold.get_layers(), 
						output_shapes=last_element_bold.get_output_shapes(), real_output_shapes=last_element_bold.get_real_output_shapes())

				bold_queue = bold_queue + [new_architecture]



				possible_layers = decoder_queue[-1].possible_next_layers()

				del decoder_queue[-1]

				for layer in possible_layers:
					new_architecture = Neural_Architecture(layers=[core_layers[layer]] + last_element_decoder.get_layers(), 
						output_shapes=last_element_decoder.get_output_shapes(), real_output_shapes=last_element_decoder.get_real_output_shapes())

				decoder_queue = decoder_queue + [new_architecture]

					
			#call BO to optimize the output_shape for each layer at this level

		print("FINISHED NAS")

		return []


if __name__ == "__main__":
	nas = Iterative_Naive_NAS()

	eeg_input_shape = (64, 5, 20, 1)
	bold_input_shape = (14000, 20, 1)
	nas.search(eeg_input_shape, bold_input_shape)
	#print(nas.generate_layer_Conv3DTranspose((10, 10, 10, 1), (30, 30, 30, 1)))
	#print(nas.generate_kernel_stride_Conv2D((10, 10, 1), (5, 5, 1)))

	
	#nas.build_layer_UpSampling2D((5, 20, 1), (23, 20, 1))