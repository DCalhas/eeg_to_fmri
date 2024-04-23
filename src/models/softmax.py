import tensorflow as tf


class Softmax(tf.keras.Model):
	
	def __init__(self, shape):

		super(Softmax, self).__init__()

		initializer = tf.initializers.GlorotUniform()
		#self.w = tf.Variable(initializer(shape=shape,dtype=tf.float32))


		self.w = self.add_weight('w',
								shape=[shape[0],],
								initializer=initializer,
								dtype=tf.float32,
								trainable=True)


	def call(self, X):
		"""
		n - number of networks
		b - number of batches
		x - x dimension of fMRI
		y - y dimension of fMRI
		z - z dimension of fMRI
		c - channel dimension = 1
		"""

		return tf.einsum('n,nbxyzc->bxyzc', tf.nn.softmax(self.w), X)