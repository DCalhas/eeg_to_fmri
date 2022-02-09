import tensorflow as tf



class LinearClassifier(tf.keras.Model):
	"""
	
	
	"""
	def __init__(self, n_classes=2):
		super(LinearClassifier, self).__init__()
		
		self.flatten = tf.keras.layers.Flatten()
		self.linear = tf.keras.layers.Dense(n_classes)
		
	def call(self, X):
		
		return self.linear(self.flatten(X))