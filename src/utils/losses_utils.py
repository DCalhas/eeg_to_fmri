import numpy as np
from numpy import correlate

import tensorflow.compat.v1 as tf

import tensorflow.keras.backend as K



def contrastive_loss(y_true, y_pred):
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(1.0 - y_pred, 0))
	return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def cross_correlation(x, y):
	#how should the normalization be done??
	x = K.l2_normalize(x, axis=1)
	y = K.l2_normalize(y, axis=1)

	x = K.batch_flatten(x)
	y = K.batch_flatten(y)

	a = K.batch_dot(x, y, axes=1)

	b = K.batch_dot(x, x, axes=1)
	c = K.batch_dot(y, y, axes=1)

	return 1 - (a / (K.sqrt(b) * K.sqrt(c)))

def correlation(vects):
	#how should the normalization be done??
	x, y = vects
	x = K.l2_normalize(x, axis=1)
	y = K.l2_normalize(y, axis=1)

	#flatten because we are dealing with 16x20 matrices
	x = K.batch_flatten(x)
	y = K.batch_flatten(y)

	a = K.batch_dot(x, y, axes=1)

	b = K.batch_dot(x, x, axes=1)
	c = K.batch_dot(y, y, axes=1)

	return 1 - (a / (K.sqrt(b) * K.sqrt(c)))

def correlation_decoder_loss(x, y):
	x = K.l2_normalize(x, axis=1)
	y = K.l2_normalize(y, axis=1)

	x = K.batch_flatten(x)
	y = K.batch_flatten(y)

	x = K.cast(x, 'float32')

	a = K.batch_dot(x, y, axes=1)

	b = K.batch_dot(x, x, axes=1)
	c = K.batch_dot(y, y, axes=1)

	return 1 - (a / (K.sqrt(b) * K.sqrt(c)))

def cos_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)
