import numpy as np
from numpy import correlate

import tensorflow.compat.v1 as tf

import tensorflow.keras.backend as K


######################################################################################################################
#
#										CONSTASTIVE LOSSES
#
######################################################################################################################

def contrastive_loss(y_true, y_pred):
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(1.0 - y_pred, 0))
	return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


######################################################################################################################
#
#										CORRELATION LOSSES
#
######################################################################################################################


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

	return 1 - tf.abs(a / (K.sqrt(b) * K.sqrt(c)))

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



######################################################################################################################
#
#										ADVERSARIAL LOSSES
#
######################################################################################################################

def loss_minmax_generator(gen_pred):
    return -tf.reduce_mean(tf.log(gen_pred))

def loss_minmax_discriminator(real_pred, real_true, gen_pred):
    #need to separate positives from negatives

    #log(1) = 0
    positives = np.ones(real_pred.shape, dtype='float32')
    #log(1-1) = 0
    negatives = np.zeros(real_pred.shape, dtype='float32')
    for instance in range(real_true.shape[0]):
        if(real_true[instance] == 1.0):
            positives[instance] = real_pred[instance][0].numpy()
        else:
            negatives[instance] = real_pred[instance][0].numpy()

    positives = tf.convert_to_tensor(positives)
    negatives = tf.convert_to_tensor(negatives)

    return -tf.reduce_mean(tf.log(positives) + tf.log(1. - negatives) + tf.log(1. - gen_pred))




def loss_wasserstein_generator(gen_pred):
    return -tf.reduce_mean(gen_pred)

def loss_wasserstein_discriminator(real_pred, real_true, gen_pred):
    #need to separate positives from negatives

    #log(1) = 0
    positives = np.ones(real_pred.shape, dtype='float32')
    #log(1-1) = 0
    negatives = np.zeros(real_pred.shape, dtype='float32')
    for instance in range(real_true.shape[0]):
        if(real_true[instance] == 1.0):
            positives[instance] = real_pred[instance][0].numpy()
        else:
            negatives[instance] = real_pred[instance][0].numpy()

    positives = tf.convert_to_tensor(positives)
    negatives = tf.convert_to_tensor(negatives)

    return tf.reduce_mean(positives) - tf.reduce_mean(negatives) - tf.reduce_mean(gen_pred)

def get_reconstruction_loss(outputs, targets):
    reconstruction_loss = cross_correlation(outputs, targets)
    return K.mean(reconstruction_loss)