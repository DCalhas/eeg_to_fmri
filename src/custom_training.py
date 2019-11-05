#!/usr/bin/env python
# coding: utf-8

# # Decoder model

# ## This model decodes EEG embeddings from another network to fMRI signal. This other network learns a space shared between EEG and fMRI.

# In[1]:


import sys
sys.path.append("..")

from utils import eeg_utils, fmri_utils, losses_utils

import numpy as np

import tensorflow.compat.v1 as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config = tf.ConfigProto(allow_soft_placement=True,
                        gpu_options=gpu_options)
config.gpu_options.allow_growth=True


tf.enable_eager_execution(config=config)

import tensorflow.keras.backend as K

import sys

#############################################################################################################
#
#                                           NETWORK ARCHITECTURE FUNCTIONS                                       
#
#############################################################################################################

def multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network):
    input_eeg = tf.keras.layers.Input(shape=eeg_input_shape)
    input_bold = tf.keras.layers.Input(shape=bold_input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_eeg = eeg_network(input_eeg)
    processed_bold = bold_network(input_bold)

    correlation = tf.keras.layers.Lambda(losses_utils.correlation, 
                         output_shape=losses_utils.cos_dist_output_shape, name="correlation_layer")([processed_eeg, processed_bold])

    return tf.keras.Model([input_eeg, input_bold], correlation)

#############################################################################################################
#
#                                           TRAINING FUNCTIONS                                       
#
#############################################################################################################

def loss_decoder(outputs, targets):
    reconstruction_loss = losses_utils.cross_correlation(outputs, targets)
    return K.mean(reconstruction_loss)

def grad_decoder(model, inputs, targets):
    with tf.GradientTape() as tape:    
        tape.watch(inputs)

        outputs = model(inputs)

        reconstruction_loss = losses_utils.cross_correlation(outputs, targets)
        reconstruction_loss = K.mean(reconstruction_loss)

        return reconstruction_loss,  tape.gradient(reconstruction_loss, model.trainable_weights)

def grad_multi_encoder(model, inputs, targets, reconstruction_loss, linear_combination):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.variables)

        outputs = model(inputs)
        
        encoder_loss = linear_combination*abs(losses_utils.contrastive_loss(outputs, targets)) + (1-linear_combination)*abs(reconstruction_loss)
        return encoder_loss,  tape.gradient(encoder_loss, 
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



#################################################################################################################
#
#                                       CUSTOM TRAINING FUNCTIONS
#
#################################################################################################################

"""
linear_combination_training

trains a encoder and decoder
it gives the decoder a reconstruction loss (cosine loss)
it gives the encoder a linear combination loss of the reconstruction loss and the contrastive loss
"""
def linear_combination_training(X_train_eeg, X_train_bold, tr_y, eeg_network, 
    decoder_model, multi_modal_model, epochs=10, 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    linear_combination=0.5, 
    batch_size=128,
    X_val_eeg=None, X_val_bold=None, tv_y=None, session=None):
    # keep results for plotting

    validation = False
    if(X_val_eeg is not None and X_val_bold is not None and tv_y is not None):
        validation = True

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
            
            # Optimize the synthesizer mode
            decoder_loss, decoder_grads = grad_decoder(decoder_model, shared_eeg, X_train_bold[batch_start:batch_stop])
            with tf.name_scope("gradient_decoder") as scope:
                optimizer.apply_gradients(zip(decoder_grads, decoder_model.trainable_variables), name=scope)

            #now train the compression by correlation model
            encoder_loss, encoder_grads = grad_multi_encoder(multi_modal_model, 
                                                             [X_train_eeg[batch_start:batch_stop], 
                                                                                 X_train_bold[batch_start:batch_stop]], 
                                                             tr_y[batch_start:batch_stop], decoder_loss, linear_combination)
            with tf.name_scope("gradient_encoders") as scope:
                optimizer.apply_gradients(zip(encoder_grads, multi_modal_model.trainable_variables), name=scope)

            # Track progress
            losses.update_batch_decoder_loss_avg(decoder_loss)
            losses.update_batch_encoder_loss_avg(encoder_loss)

        # end epoch
        decoder_loss = losses.get_batch_decoder_loss_avg()
        encoder_loss = losses.get_batch_encoder_loss_avg()

        #get validation analyses
        shared_eeg_val = eeg_network(X_val_eeg)
        val_loss = loss_decoder(decoder_model(shared_eeg_val), X_val_bold)
        
        print("Encoder Loss: ", tf.keras.backend.eval(encoder_loss), " || Decoder Loss: ", tf.keras.backend.eval(decoder_loss),
            "Validation Decoder Loss: ", tf.keras.backend.eval(val_loss))
        sys.stdout.flush()

    shared_eeg_val = eeg_network(X_val_eeg)
    return tf.keras.backend.eval(loss_decoder(decoder_model(shared_eeg_val), X_val_bold))
