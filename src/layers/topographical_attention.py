import tensorflow as tf

"""
Topographical_Attention:

	Int channels
	Int features reduce over number of features

"""
class Topographical_Attention(tf.keras.layers.Layer):

	def __init__(self, channels, features):

		super(Topographical_Attention, self).__init__()


		self.channels=channels
		self.features=features
		
		self.A = self.add_weight('A',
								shape=[self.channels,self.channels,self.features],
								initializer=tf.initializers.GlorotUniform(),
								dtype=tf.float32,
								trainable=True)

	"""
	The defined topographical attention mechanism has an extra step:

		instead of performing the element wise product,
		one reduces the feature dimension,
		so instead of the attention weight matrix being of shape NxN
		it has shape NxNxF
		the F refers to the feature dimension that is reduced
	"""
	def call(self, X):
		
		c = tf.einsum('NCF,CMF->NCM', X, self.A)
		W = tf.nn.softmax(c, axis=-1)#dimension that is reduced in the next einsum, is the one that sums to one
		
		self.attention_scores = W
		
		#sum over M all M channels are multiplied by the attention scores over axis M that is normalized 
		return tf.einsum('NMF,NCM->NCF', X, W)