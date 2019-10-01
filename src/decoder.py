#!/usr/bin/env python
# coding: utf-8

# # Decoder model

# ## This model decodes EEG embeddings from another network to fMRI signal. This other network learns a space shared between EEG and fMRI.

# In[1]:


import sys
sys.path.append("..")

import eeg_utils
import fmri_utils
import deep_cross_corr

import numpy as np
from numpy import correlate

import matplotlib.pyplot as plt

import mne
from nilearn.masking import apply_mask, compute_epi_mask

from sklearn.preprocessing import normalize

from scipy.signal import resample

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

import tensorflow.keras.backend as K

import sys


#############################################################################################################
#
#                                           LOAD DATA FUNCTION                                       
#
#############################################################################################################

def load_data(train_instances, test_instances):

    mask = fmri_utils.get_population_mask()

    #Load Data
    eeg_train, bold_train = deep_cross_corr.get_data(train_instances, masker=mask)
    eeg_test, bold_test = deep_cross_corr.get_data(test_instances, masker=mask)

    eeg_train = eeg_train.reshape(eeg_train.shape[0], eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
    eeg_test = eeg_test.reshape(eeg_test.shape[0], eeg_test.shape[1], eeg_test.shape[2], eeg_test.shape[3], 1)

    bold_train = bold_train.reshape(bold_train.shape[0], bold_train.shape[1], bold_train.shape[2], 1)
    bold_test = bold_test.reshape(bold_test.shape[0], bold_test.shape[1], bold_test.shape[2], 1)

    return eeg_train, bold_train, eeg_test, bold_test

#############################################################################################################
#
#                                           NETWORK ARCHITECTURE FUNCTIONS                                       
#
#############################################################################################################


def decoding_network(input_shape):
    decoder_model = tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(1, kernel_size=(100, 1),
                              activation='selu', strides=(3,1), input_shape=input_shape[1:]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(1, kernel_size=(100, 1), 
                              activation='selu', strides=(50,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ZeroPadding2D(padding=(57,0))
    ], name="bold_synthesizer")

    decoder_model.build(input_shape)

    return decoder_model

def multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network):
    input_eeg = tf.keras.layers.Input(shape=eeg_input_shape)
    input_bold = tf.keras.layers.Input(shape=bold_input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_eeg = eeg_network(input_eeg)
    processed_bold = bold_network(input_bold)

    correlation = tf.keras.layers.Lambda(deep_cross_corr.correlation, 
                         output_shape=deep_cross_corr.cos_dist_output_shape, name="correlation_layer")([processed_eeg, processed_bold])

    return tf.keras.Model([input_eeg, input_bold], correlation)

#############################################################################################################
#
#                                           TRAINING FUNCTIONS                                       
#
#############################################################################################################

def loss_decoder(model, inputs, targets):
    reconstruction_loss = deep_cross_corr.cross_correlation(outputs, targets)
    return K.mean(reconstruction_loss)


def grad_decoder(model, inputs, targets):
    with tf.GradientTape() as tape:    
        tape.watch(inputs)
        outputs = model(inputs)
        
        #loss
        reconstruction_loss = deep_cross_corr.cross_correlation(outputs, targets)
        reconstruction_loss = K.mean(reconstruction_loss)
        return -reconstruction_loss,  tape.gradient(-reconstruction_loss, model.trainable_weights)

def grad_multi_encoder(model, inputs, targets, reconstruction_loss):
    with tf.GradientTape() as tape:    
        tape.watch(inputs)
        outputs = model(inputs)
        
        #loss
        encoder_loss = abs(deep_cross_corr.contrastive_loss(outputs, targets)) + abs(reconstruction_loss)
        return -encoder_loss,  tape.gradient(-encoder_loss, 
                                            model.trainable_weights)

class custom_training_loss:
    def __init__(self):
        self.encoder_loss = 0
        self.decoder_loss = 0
        self.batch = 0
        
    def update_batch_decoder_loss_avg(self, loss):
        self.decoder_loss += loss
        self.batch += 1
    
    def get_batch_decoder_loss_avg(self):
        return self.decoder_loss/self.batch
    
    def update_batch_encoder_loss_avg(self, loss):
        self.encoder_loss += loss
        self.batch += 1
    
    def get_batch_encoder_loss_avg(self):
        return self.encoder_loss/self.batch


def run_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_model, multi_modal_model, epochs=10, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), batch_size=128):
    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    global_step = tf.Variable(0)


    for epoch in range(epochs):
        
        losses = custom_training_loss()

        for batch_init in range(0, len(X_train_eeg), batch_size):
            batch_start = batch_init
            if(batch_start + batch_size >= len(X_train_eeg)):
                batch_stop = len(X_train_eeg)
            else:
                batch_stop = batch_start + batch_size
            
            shared_eeg = eeg_network(X_train_eeg[batch_start:batch_stop])
            
            # Optimize the synthesizer model
            decoder_loss, decoder_grads = grad_decoder(decoder_model, shared_eeg, X_train_bold[batch_start:batch_stop])
            optimizer.apply_gradients(zip(decoder_grads, decoder_model.trainable_variables), 
                                      global_step)

            
            #now train the compression by correlation model
            encoder_loss, encoder_grads = grad_multi_encoder(multi_modal_model, 
                                                             [X_train_eeg[batch_start:batch_stop], 
                                                                                 X_train_bold[batch_start:batch_stop]], 
                                                             tr_y[batch_start:batch_stop], decoder_loss)
            optimizer.apply_gradients(zip(encoder_grads, multi_modal_model.trainable_variables), 
                                      global_step)
            # Track progress
            losses.update_batch_decoder_loss_avg(decoder_loss)
            losses.update_batch_encoder_loss_avg(encoder_loss)

        # end epoch
        decoder_loss = losses.get_batch_decoder_loss_avg()
        encoder_loss = losses.get_batch_encoder_loss_avg()
        
        print("Encoder Loss: ", tf.keras.backend.eval(encoder_loss), " || Decoder Loss: ", tf.keras.backend.eval(decoder_loss))

eeg_input_shape = (64, 5, 20, 1)
kernel_size = (eeg_train.shape[1], eeg_train.shape[2], 1)
eeg_network = deep_cross_corr.eeg_network(eeg_input_shape, kernel_size)

bold_input_shape = (14164, 20, 1)
kernel_size = (bold_train.shape[1], 1)
bold_network = deep_cross_corr.bold_network(bold_input_shape, kernel_size)

shared_eeg_train = eeg_network.predict(eeg_train)
input_shape = (None, shared_eeg_train.shape[1], shared_eeg_train.shape[2], 1)
decoder_model = decoding_network(input_shape)

multi_modal_model = multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network)

if __name__ == "__main__":
    #############################################################################################################
    #
    #                                       LOAD DATA
    #
    #############################################################################################################

    eeg_train, bold_train, eeg_test, bold_test = load_data(list(range(14)), list(range(14, 16)))

    #############################################################################################################
    #
    #                                       EEG NETWORK BRANCH
    #
    #############################################################################################################

    #EEG network branch
    eeg_input_shape = (eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
    kernel_size = (eeg_train.shape[1], eeg_train.shape[2], 1)
    eeg_network = deep_cross_corr.eeg_network(eeg_input_shape, kernel_size)
    print(eeg_network.summary())

    #############################################################################################################
    #
    #                                       BOLD NETWORK BRANCH
    #
    #############################################################################################################

    #BOLD network branch
    bold_input_shape = (bold_train.shape[1], bold_train.shape[2], 1)
    kernel_size = (bold_train.shape[1], 1)
    bold_network = deep_cross_corr.bold_network(bold_input_shape, kernel_size)
    print(bold_network.summary())


    #############################################################################################################
    #
    #                                       DECODER NETWORK
    #
    #############################################################################################################

    shared_eeg_train = eeg_network.predict(eeg_train)


    #Decoder Network
    input_shape = (None, shared_eeg_train.shape[1], shared_eeg_train.shape[2], 1)

    decoder_model = decoding_network(input_shape)
    print(decoder_model.summary())


    #############################################################################################################
    #
    #                                       MULTI MODAL NETWORK
    #
    #############################################################################################################

    multi_modal_model = multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network)


    X_train_eeg, X_train_bold, tr_y = deep_cross_corr.create_eeg_bold_pairs(eeg_train, bold_train)
    X_test_eeg, X_test_bold, te_y = deep_cross_corr.create_eeg_bold_pairs(eeg_test, bold_test)

    #convert to tensors, for the networks to accept it as input
    X_train_eeg = tf.convert_to_tensor(X_train_eeg, dtype=np.float32)
    X_train_bold = tf.convert_to_tensor(X_train_bold, dtype=np.float32)
    tr_y = tf.convert_to_tensor(tr_y, dtype=np.float32)

    run_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_model, multi_modal_model)