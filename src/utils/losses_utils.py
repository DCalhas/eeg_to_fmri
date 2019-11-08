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



######################################################################################################################
#
#										RANKING UTILS
#
######################################################################################################################

def get_ranked_bold(eeg, bold, corr_model=None, bold_network=None, top_k=5):
    #build training set for decoder
    #for each eeg instance - compare all the other bold instances
    ranked_bold = np.zeros((eeg.shape[0], ) + bold_network.output_shape[1:], dtype='float32')

    for eeg_idx in range(len(eeg)):
        eeg_instance = eeg[eeg_idx].reshape((1,) + eeg[eeg_idx].shape)

        ranking_corr = np.zeros(bold.shape[0])
        ranking_idx = list(range(bold.shape[0]))

        #check what is the correlation value with every single bold
        for bold_idx in range(len(bold)):
            bold_instance = bold[bold_idx].reshape((1,) + bold[bold_idx].shape)
            corr = corr_model.predict([eeg_instance, bold_instance])
            ranking_corr[bold_idx] = corr

        rankings = dict(zip(ranking_idx, list(ranking_corr)))

        top_ranked = []
        top_corr = []
        rank = 0
        for key, value in sorted(rankings.items(), key=lambda kv: kv[1], reverse=True):
            #stop condition, only gather the top_k correlated bold signals
            if(rank >= top_k):
                break

            top_ranked += [key]
            top_corr += [value]

            rank += 1

        top_corr = np.array(top_corr)
        top_corr = top_corr/np.sum(top_corr)

        #linear combination of the bold_network activations
        top_activations = bold_network(bold[top_ranked])
        for activation in range(len(top_activations)):
            ranked_bold[eeg_idx] = ranked_bold[eeg_idx] + top_corr[activation]*top_activations[activation]

    return ranked_bold.astype('float32')