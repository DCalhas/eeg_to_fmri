import tensorflow as tf

import numpy as np


"""
A simple mask that formes a 3D circle with an elipse in the z-axis


Example:

	>>> layer=CircleMask(y.shape)
"""
class MRICircleMask(tf.keras.layers.Layer):
	
	def __init__(self, input_shape, radius=25.):
		super(MRICircleMask, self).__init__()
		
		if(len(input_shape)==5):
			input_shape = input_shape[1:-1]
		elif(len(input_shape)==4 and input_shape[0]==1):
			input_shape = input_shape[:-1]
		elif(len(input_shape)==4 and input_shape[0]!=1):
			input_shape = input_shape[1:]
		
		h,w,d = input_shape
		center=[h//2, w//2, d//2]

		Y, X, Z = np.ogrid[:h, :w, :d]
		dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (h/d)*(Z-center[2])**2)
		
		self.mask = self.add_weight('mask',
								shape=list(input_shape+(1,)),
								initializer=tf.constant_initializer((dist_from_center <= radius).astype(np.float32)),
								dtype=tf.float32,
								trainable=False)
		
	def call(self, X):
		return X*self.mask