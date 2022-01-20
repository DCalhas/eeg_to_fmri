import tensorflow as tf

from models import fmri_ae

from utils import state_utils

from layers.fourier_features import RandomFourierFeatures, FourierFeatures
from layers.fft import padded_iDCT3D, DCT3D, variational_iDCT3D
from layers.topographical_attention import Topographical_Attention
from layers.resnet_block import ResBlock

from pathlib import Path
import shutil
import os
import pickle

search_space = [{'name': 'learning_rate', 'type': 'continuous',
					'domain': (1e-5, 1e-2)},
					{'name': 'reg', 'type': 'continuous',
					'domain': (1e-6, 1e-1)},
				   #{'name': 'channels', 'type': 'discrete',
					#'domain': (4,8,16)},
				   {'name': 'batch_norm', 'type': 'discrete',
					'domain': (0,1)},
					{'name': 'eeg_architecture', 'type': 'discrete',
					'domain': tuple(range(20))},
					{'name': 'fmri_decoder_architecture', 'type': 'discrete',
					'domain': (0,1)},
				   {'name': 'dropout', 'type': 'discrete',
					'domain': (0.0, 0.2, 0.3, 0.4, 0.5)},
				   #{'name': 'skip_connections', 'type': 'discrete',
					#'domain': (0,1)},
				   {'name': 'epochs', 'type': 'discrete',
					'domain': (5,10,15,20,25,30)},
					{'name': 'batch_size', 'type': 'discrete',
					'domain': (2, 4, 8, 16, 32)}]


@tf.function
def call(obj, x1, x2):
    z1 = obj.eeg_encoder(x1)

    z2 = obj.fmri_encoder(x2)
    return [obj.decoder(z1), z1, z2]

"""
    Random behaviour of GPU with tf functions does not reproduce the same results
    Call this function when getting results
"""
def _call(self, x1, x2):
    z1 = self.eeg_encoder(x1)

    if(self.training):
        z2 = self.fmri_encoder(x2)
        return [self.decoder(z1), z1, z2]

    return self.decoder(z1)

def build(*kwargs):
	return EEG_to_fMRI()


"""
This class implements an architecture for EEG to fMRI transcription

encode: architecture that encodes the EEG signal to a space where an instance of fMRI is also represented

decode: architecture that maps the encoded representation to the fMRI space representation

call: encode and decode

"""

class EEG_to_fMRI(tf.keras.Model):


    """
        NA_specification - tuple - (list1, list2, bool, tuple1, tuple2)
                                    * list1 - kernel sizes
                                    * list2 - stride sizes
                                    * bool - maxpool
                                    * tuple1 - kernel size of maxpool
                                    * tuple2 - stride size of maxpool
                                    Example:
                                    na = ([(2,2,2), (2,2,2)], [(1,1,1), (1,1,1)], True, (2,2,2), (1,1,1))
                                    na is a neural architecture with 2 layers, kernel of size 2 for all 3 dimensions
                                    stride of size 1 for all dimensions, between each layer a max pooling operation 
                                    is applied with kernel size 2 for all dimensions and stride size 1 for all dimensions

    """
    def __init__(self, latent_shape, input_shape, na_spec, n_channels,
                weight_decay=0.000, skip_connections=False, batch_norm=True,
                dropout=False, local=True, fourier_features=False, 
                conditional_attention_style=False, random_fourier=False,
                conditional_attention_style_prior=False,
                inverse_DFT=False, DFT=False, 
                variational_iDFT=False, variational_coefs=None,
                resolution_decoder=None, low_resolution_decoder=False,
                topographical_attention=False, seed=None, fmri_args=None):
        super(EEG_to_fMRI, self).__init__()

        self.training=True
        self.latent_shape=latent_shape
        self._input_shape=input_shape
        self.na_spec=na_spec
        self.n_channels=n_channels
        self.weight_decay=weight_decay
        self.skip_connections=skip_connections
        self.batch_norm=batch_norm
        self.dropout=dropout
        self.local=local
        self.fourier_features=fourier_features
        self.conditional_attention_style=conditional_attention_style
        self.random_fourier=random_fourier
        self.inverse_DFT=inverse_DFT
        self.DFT=DFT
        self.variational_iDFT=variational_iDFT
        self.variational_coefs=variational_coefs
        self.resolution_decoder=resolution_decoder
        self.low_resolution_decoder=low_resolution_decoder
        self.topographical_attention=topographical_attention
        self.seed=seed
        self.fmri_args=fmri_args
        
        if(len(fmri_args)==17):#needs to be update if 
            self.fmri_ae = fmri_ae.fMRI_AE(*fmri_args)
        else:
            raise NotImplementedError

        input_shape, x, attention_scores = self.build_encoder(latent_shape, input_shape, na_spec, n_channels, 
                            dropout=dropout, weight_decay=weight_decay, 
                            skip_connections=skip_connections, local=local, 
                            batch_norm=batch_norm, 
                            topographical_attention=topographical_attention,
                            seed=seed)
        self.build_decoder(input_shape, x, latent_shape, inverse_DFT=inverse_DFT, DFT=DFT,
                            attention_scores=attention_scores, 
                            conditional_attention_style=conditional_attention_style,
                            conditional_attention_style_prior=conditional_attention_style_prior,
                            random_fourier=random_fourier,
                            fourier_features=fourier_features,
                            resolution_decoder=resolution_decoder,
                            low_resolution_decoder=low_resolution_decoder,
                            variational_iDFT=variational_iDFT, 
                            variational_coefs=variational_coefs,
                            outfilter=self.fmri_ae.outfilter, seed=seed)

    def build_encoder(self, latent_shape, input_shape, na_spec, n_channels, 
                            dropout=False, weight_decay=0.000, 
                            skip_connections=False, batch_norm=True, 
                            local=True, topographical_attention=False,
                            seed=None):

        attention_scores=None

        input_shape = tf.keras.layers.Input(shape=input_shape)

        if(topographical_attention):
            x = input_shape
            #reshape to flattened features to apply attention mechanism
            x = tf.keras.layers.Reshape((self._input_shape[0], self._input_shape[1]*self._input_shape[2]))(x)
            #topographical attention
            x, attention_scores = Topographical_Attention(self._input_shape[0], self._input_shape[1]*self._input_shape[2])(x)
            #reshape back to original shape
            x = tf.keras.layers.Reshape(self._input_shape)(x)
            previous_block_x = x
        else:
            x = input_shape
            previous_block_x = input_shape

        for i in range(len(na_spec[0])):
            x = ResBlock(tf.keras.layers.Conv3D, 
                        na_spec[0][i], na_spec[1][i], n_channels,
                        maxpool=na_spec[2], batch_norm=batch_norm, weight_decay=weight_decay, 
                        maxpool_k=na_spec[3], maxpool_s=na_spec[4],
                        skip_connections=skip_connections, seed=seed)(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)#placeholder
        x = tf.keras.layers.Reshape(latent_shape)(x)

        self.eeg_encoder = tf.keras.Model(input_shape, x)
        self.fmri_encoder = self.fmri_ae.encoder

        return input_shape, x, attention_scores

    def build_decoder(self, input_shape, output_encoder, latent_shape, fourier_features=False, random_fourier=False, 
                            attention_scores=None, conditional_attention_style=False, conditional_attention_style_prior=False,
                            inverse_DFT=False, DFT=False, 
                            low_resolution_decoder=False, resolution_decoder=None, 
                            variational_iDFT=False, variational_coefs=None, 
                            dropout=False, outfilter=0, seed=None):

        x = tf.keras.layers.Flatten()(output_encoder)

        if(fourier_features):
            if(random_fourier):
                self.latent_resolution = RandomFourierFeatures(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                                                  trainable=True, seed=seed, name="random_fourier_features")
            else:
                self.latent_resolution = FourierFeatures(latent_shape[0]*latent_shape[1]*latent_shape[2], 
                                                                    trainable=True, name="fourier_features")
        else:
            self.latent_resolution = tf.keras.layers.Dense(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                                                            name="dense")
        if(conditional_attention_style):
            if(conditional_attention_style_prior):
                self.latent_style = self.add_weight(name='style_prior',
                                                      shape=(latent_shape[0]*latent_shape[1]*latent_shape[2],),
                                                      initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                                                      trainable=True)
            else:
                attention_scores = tf.keras.layers.Flatten(name="conditional_attention_style_flatten")(attention_scores)
                self.latent_style = tf.keras.layers.Dense(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                                        use_bias=False,
                                                        name="conditional_attention_style_dense",
                                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(attention_scores)

        x = self.latent_resolution(x)
        
        if(conditional_attention_style):
            x = x*self.latent_style

        if(dropout):
            x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Reshape(latent_shape)(x)

        #placeholder
        if(resolution_decoder is None):
            resolution_decoder=latent_shape

        if(low_resolution_decoder):
            x = tf.keras.layers.Flatten()(x)

            assert type(resolution_decoder) is tuple and len(resolution_decoder) == 3
            latent_shape = resolution_decoder

            #upsampling
            x = tf.keras.layers.Dense(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
            x = tf.keras.layers.Reshape(latent_shape)(x)
        if(DFT):
            #convert to Discrete cosine transform low resolution coefficients
            x = DCT3D(latent_shape[0], latent_shape[1], latent_shape[2])(x)
        if(inverse_DFT):
            if(variational_iDFT):
                assert type(variational_coefs) is tuple
                x = variational_iDCT3D(*(latent_shape + self.fmri_ae.in_shape[:3] + variational_coefs))(x)
            else:
                x = padded_iDCT3D(latent_shape[0], latent_shape[1], latent_shape[2],
                            out1=self.fmri_ae.in_shape[0], out2=self.fmri_ae.in_shape[1], out3=self.fmri_ae.in_shape[2])(x)
        else:
            x = tf.keras.layers.Flatten()(x)
            #upsampling
            x = tf.keras.layers.Dense(self.fmri_ae.in_shape[0]*self.fmri_ae.in_shape[1]*self.fmri_ae.in_shape[2],
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        
        x = tf.keras.layers.Reshape(self.fmri_ae.in_shape)(x)
        #filter
        if(outfilter == 1):
            x = tf.keras.layers.Conv3D(filters=1, kernel_size=1, strides=1,
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        elif(outfilter == 2):
            x = LocallyConnected3D(filters=1, kernel_size=1, strides=1, implementation=3,
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)

        self.decoder = tf.keras.Model(input_shape, x)   

    def build(self, input_shape1, input_shape2):
        self.eeg_encoder.build(input_shape=input_shape1)
        self.decoder.build(input_shape=self.eeg_encoder.output_shape)

        self.fmri_encoder.build(input_shape=input_shape2)

        self.built=True
        
        self.trainable_variables.append(self.fmri_encoder.trainable_variables)


    """
        Random behaviour of GPU with tf functions does not reproduce the same results
        Call this function when getting results
    """
    #@tf.function(input_signature=[tf.TensorSpec([None,64,134,10,1], tf.float32), tf.TensorSpec([None,64,64,30,1], tf.float32)])
    def call(self, x1, x2):
        if(self.training):
            return [self.decoder(x1), 
                    self.eeg_encoder(x1), 
                    self.fmri_encoder(x2)]

        return self.decoder(x1)

    def saved_call(self, x1, x2):
        if(self.training):
            return [self.decoder(x1), 
                    self.eeg_encoder(x1), 
                    self.fmri_encoder(x2)]

        return self.decoder(x1)

    def get_config(self):

        return {"latent_shape": self.latent_shape,
                "input_shape": self._input_shape,
                "na_spec": self.na_spec,
                "n_channels": self.n_channels,
                "weight_decay": self.weight_decay,
                "skip_connections": self.skip_connections,
                "batch_norm": self.batch_norm,
                "dropout": self.dropout,
                "local": self.local,
                "fourier_features": self.fourier_features,
                "conditional_attention_style": self.conditional_attention_style,
                "random_fourier": self.random_fourier,
                "inverse_DFT": self.inverse_DFT,
                "DFT": self.DFT,
                "variational_iDFT": self.variational_iDFT,
                "variational_coefs": self.variational_coefs,
                "resolution_decoder": self.resolution_decoder,
                "low_resolution_decoder": self.low_resolution_decoder,
                "topographical_attention": self.topographical_attention,
                "seed": self.seed,
                "fmri_args": self.fmri_args}

    @classmethod
    def from_config(cls, config):
        return cls(**config)