import tensorflow as tf

"""
IEEE-Conference on Neural Engineering

This is a reproduction of the paper Liu et al. 2019, A Convolutional Neural Network for Transcoding Simultaneously Acquired EEG-fMRI Data

Disclaimer: This is not the official implementation and can contain operations that were not inteded by the authors of the paper
"""
class Liu_et_al(tf.keras.Model):


	"""

	Inputs:
		* eeg_shape - tuple (n_channels, time)
		* fmri_shape - tuple (X,Y,Z,T)

	max number of channels to run on the personal computer is 4
	standard number is 16
	"""
	def __init__(self, eeg_shape, fmri_shape, n_channels=4, latent_dim=256):
		super(Liu_et_al, self).__init__()

		self.eeg_shape = eeg_shape
		self.fmri_shape = fmri_shape

		self.time_dimension=fmri_shape[-1]
		self.spatial_dimension=fmri_shape[0]*fmri_shape[1]*fmri_shape[2]

		self.n_channels=n_channels
		self.latent_dim=latent_dim

		self.build_model()

	def build_model(self):

		input_shape = tf.keras.layers.Input(shape=(self.eeg_shape[1], self.eeg_shape[0]))
		
		x = tf.keras.layers.Dense(self.latent_dim)(input_shape)
		#8*8*4=256
		x = tf.keras.layers.Reshape((8,8,4,self.fmri_shape[-1]))(x)

		#kernel size to map to fmri spatial dimension sizes
		#x = tf.keras.layers.Conv3DTranspose(self.time_dimension, kernel_size=(9,9,4))(x)

		#16x16x7x300 go through 12 layers that maintain dimension so kernel and stride should be both 1
		#channel dimension becomes 16 so 300*16
		for i in range(11):
			x = tf.keras.layers.Conv3D(self.time_dimension*self.n_channels, kernel_size=1, strides=1)(x)
		#channel dimension goes back to 1
		x = tf.keras.layers.Conv3D(self.time_dimension, kernel_size=1, strides=1)(x)
		
		#reshape to perform convolution on the time dimension "Temporal Convolutional Layer"
		x = tf.keras.layers.Reshape((self.time_dimension,self.latent_dim))(x)
		#now goes to the time slicing part 16*16*7=1792
		for i in range(5):
			x = tf.keras.layers.Conv1D(self.latent_dim*self.n_channels, kernel_size=1, strides=1)(x)
		x = tf.keras.layers.Conv1D(self.spatial_dimension, kernel_size=1, strides=1)(x)

		x = tf.keras.layers.Reshape(self.fmri_shape)(x)
		
		self.nn=tf.keras.Model(input_shape, x)

	"""
	input_shape: (channels,time,1)
	output_shape: (X,Y,Z,time)
	"""
	def call(self, x):
		#channels last
		x = tf.transpose(x)
		return self.nn(x)