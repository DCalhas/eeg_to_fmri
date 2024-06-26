import tensorflow as tf

import tensorflow_probability as tfp

from models import fmri_ae

from utils import state_utils

from tensorflow.keras.layers import Dense#globals get attr

from regularizers.activity_regularizers import InOfDistribution, MaxBatchNorm

from layers.temporal import DenseTemporal
from layers.fourier_features import RandomFourierFeatures, FourierFeatures, Sinusoids, MaxNormalization
from layers.fft import padded_iDCT3D, DCT3D, variational_iDCT3D, iDCT3D
from layers.topographical_attention import Topographical_Attention, Topographical_Attention_Scores_Regularization, Topographical_Attention_Reduction
from layers.resnet_block import ResBlock, pretrained_ResBlock
from layers.bayesian import DenseVariational
from layers.mask import MRICircleMask
from layers.latent_attention import Latent_EEG_Spatial_Attention, Latent_fMRI_Spatial_Attention
from layers.style import Style

from pathlib import Path
import shutil
import os
import pickle

parameters=(0.002980911194116198, 0.0, (9, 9, 4), (1, 1, 1), 4, (7, 7, 7), 4, True, True, True, True, 3, 1, True)


#parameters={"learning_rate": 0.002980911194116198, "weight_decay": 0.0, "kernel_size": (9,9,4), "stride_size": (1,1,1), "batch_size": 4, "latent_dimension": (7,7,7), "n_channels": 4, "max_pool": True, "batch_norm": True, "skip_connections": True, "dropout": True, "n_stacks": 3, "outfilter": 1, "local": True,}

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



class EEG_to_fMRI(tf.keras.Model):
    """
    This class implements an architecture for EEG to fMRI transcription

    encode: architecture that encodes the EEG signal to a space where an instance of fMRI is also represented

    decode: architecture that maps the encoded representation to the fMRI space representation

    call: encode and decode

    """


    
    def __init__(self, latent_shape, input_shape, na_spec, n_channels,
                weight_decay=0.000, skip_connections=False, batch_norm=True,
                dropout=False, local=True, fourier_features=False, time_length=1,
                conditional_attention_style=False, random_fourier=False,
                conditional_attention_style_prior=False, batch_norm_reg=False,
                inverse_DFT=False, DFT=False, aleatoric_uncertainty=False,
                variational_iDFT=False, variational_coefs=None, variational_dist=None,
                variational_iDFT_dependent=False, variational_iDFT_dependent_dim=1,
                variational_random_padding=False, fourier_normalization="layer",
                resolution_decoder=None, low_resolution_decoder=False,
                topographical_attention=False, organize_channels=False,
                consistency=False, seed=None, fmri_args=None):
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
        super(EEG_to_fMRI, self).__init__()

        self.training=True
        self.latent_shape=latent_shape
        self._input_shape=input_shape
        self.na_spec=na_spec
        self.n_channels=n_channels
        self.time_length=time_length
        self.weight_decay=weight_decay
        self.skip_connections=skip_connections
        self.batch_norm=batch_norm
        self.batch_norm_reg=batch_norm_reg
        self.dropout=dropout
        self.local=local
        self.fourier_features=fourier_features
        self.conditional_attention_style=conditional_attention_style
        self.conditional_attention_style_prior=conditional_attention_style_prior
        self.random_fourier=random_fourier
        self.fourier_normalization=fourier_normalization
        self.inverse_DFT=inverse_DFT
        self.DFT=DFT
        self.variational_iDFT=variational_iDFT
        self.variational_coefs=variational_coefs
        self.variational_iDFT_dependent=variational_iDFT_dependent
        self.variational_iDFT_dependent_dim=variational_iDFT_dependent_dim
        self.variational_dist=variational_dist
        self.variational_random_padding=variational_random_padding
        self.resolution_decoder=resolution_decoder
        self.low_resolution_decoder=low_resolution_decoder
        self.topographical_attention=topographical_attention
        self.organize_channels=organize_channels
        self.aleatoric_uncertainty=aleatoric_uncertainty
        self.consistency=consistency
        self.seed=seed
        self.fmri_args=fmri_args
        
        if(len(fmri_args)==17):#needs to be update if 
            self.fmri_ae = fmri_ae.fMRI_AE(*fmri_args, time_length=time_length)
        else:
            raise NotImplementedError

        input_shape, x, attention_scores = self.build_encoder(latent_shape, input_shape, na_spec, n_channels, 
                            dropout=dropout, weight_decay=weight_decay, time_length=time_length,
                            skip_connections=skip_connections, local=local, 
                            batch_norm=batch_norm, 
                            topographical_attention=topographical_attention,
                            organize_channels=organize_channels,
                            seed=seed)
        self.build_decoder(input_shape, x, latent_shape, inverse_DFT=inverse_DFT, DFT=DFT,
                            attention_scores=attention_scores, time_length=time_length,
                            conditional_attention_style=conditional_attention_style,
                            conditional_attention_style_prior=conditional_attention_style_prior,
                            random_fourier=random_fourier,
                            batch_norm_reg=batch_norm_reg,
                            fourier_features=fourier_features,
                            fourier_normalization=fourier_normalization,
                            resolution_decoder=resolution_decoder,
                            low_resolution_decoder=low_resolution_decoder,
                            variational_iDFT=variational_iDFT, 
                            variational_coefs=variational_coefs,
                            variational_iDFT_dependent=variational_iDFT_dependent,
                            variational_iDFT_dependent_dim=variational_iDFT_dependent_dim,
                            variational_dist=variational_dist,
                            consistency=consistency,
                            variational_random_padding=variational_random_padding,
                            outfilter=self.fmri_ae.outfilter, weight_decay=weight_decay, seed=seed)

    def build_encoder(self, latent_shape, input_shape, na_spec, n_channels, 
                            dropout=False, weight_decay=0.000, time_length=1,
                            skip_connections=False, batch_norm=True, 
                            local=True, topographical_attention=False,
                            organize_channels=False, seed=None):

        attention_scores=None

        input_shape = tf.keras.layers.Input(shape=input_shape)

        if(topographical_attention):
            x = input_shape
            #reshape to flattened features to apply attention mechanism
            x = tf.keras.layers.Reshape((self._input_shape[0], self._input_shape[1]*self._input_shape[2]))(x)
            #topographical attention
            x, attention_scores = Topographical_Attention(self._input_shape[0], self._input_shape[1]*self._input_shape[2])(x)
            if(organize_channels):
                attention_scores = Topographical_Attention_Scores_Regularization()(attention_scores)
            #x = Topographical_Attention_Reduction()(x, attention_scores)
            #x, attention_scores = Topographical_Attif(organize_channels):
            #reshape back to original shape
            x = tf.keras.layers.Reshape(self._input_shape)(x)
            previous_block_x = x
        else:
            x = input_shape
            previous_block_x = input_shape

        for i in range(len(na_spec[0])):
            x = ResBlock("Conv3D", 
                        na_spec[0][i], na_spec[1][i], n_channels,
                        maxpool=na_spec[2], batch_norm=batch_norm, weight_decay=weight_decay, 
                        maxpool_k=na_spec[3], maxpool_s=na_spec[4],
                        skip_connections=skip_connections, seed=seed)(x)


        if(time_length>1):
            """
            Here we can not avoid but manipulate the dimensions in order for the shape to be correct for the regularization
            """
            x=tf.keras.layers.Reshape(target_shape=(x.shape[1]*x.shape[2]*x.shape[4], x.shape[3]))(x)
            x=DenseTemporal(latent_shape[0]*latent_shape[1]*latent_shape[2], kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
            x=tf.keras.layers.Permute((2,1))(x)
            x=DenseTemporal(time_length, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
            x=tf.keras.layers.Permute((2,1))(x)
            x=tf.keras.layers.Reshape(latent_shape+(time_length,))(x)
        else:
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
            x = tf.keras.layers.Reshape(latent_shape)(x)

        self.eeg_encoder = tf.keras.Model(input_shape, x)
        self.fmri_encoder = self.fmri_ae.encoder

        return input_shape, x, attention_scores

    def build_decoder(self, input_shape, output_encoder, latent_shape, time_length=1, fourier_features=False, random_fourier=False, 
                            attention_scores=None, conditional_attention_style=False, conditional_attention_style_prior=False,
                            inverse_DFT=False, DFT=False, fourier_normalization="layer", batch_norm_reg=False,
                            low_resolution_decoder=False, resolution_decoder=None, 
                            variational_iDFT=False, variational_coefs=None, variational_dist=None,
                            variational_iDFT_dependent=False, variational_iDFT_dependent_dim=1,
                            variational_random_padding=False, consistency=False,
                            dropout=False, outfilter=0, weight_decay=0., seed=None):

        x = tf.keras.layers.Flatten()(output_encoder)#TODO: does it make sense in TRs>1?

        if(fourier_features):
            if(random_fourier):
                x = RandomFourierFeatures(latent_shape[0]*latent_shape[1]*latent_shape[2], batch_norm_reg=batch_norm_reg, consistency=consistency,
                                                                normalization=fourier_normalization, trainable=True, seed=seed, name="random_fourier_features")(x)
            else:
                x = FourierFeatures(latent_shape[0]*latent_shape[1]*latent_shape[2], 
                                                                    trainable=True, name="fourier_features")(x)
            x=Sinusoids()(x)
        else:
            x = tf.keras.layers.Dense(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                                                            name="dense")(x)#TODO: does it make sense in TRs>1?
        
        if(conditional_attention_style):
            if(conditional_attention_style_prior):
                x = Style(initializer="glorot_uniform", trainable=True, seed=seed, name='style_prior')(x)
            else:
                attention_scores = tf.keras.layers.Flatten(name="conditional_attention_style_flatten")(attention_scores)
                self.latent_style = tf.keras.layers.Dense(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                                        use_bias=False,
                                                        name="conditional_attention_style_dense",
                                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(attention_scores)
                x = x*self.latent_style

        if(dropout):
            x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Reshape(latent_shape)(x)

        #placeholder
        if(resolution_decoder is None):
            resolution_decoder=latent_shape

        if(low_resolution_decoder):
            assert not time_length>1, "This layers are not set for time length > 1"
            x = tf.keras.layers.Flatten()(x)

            assert type(resolution_decoder) is tuple and len(resolution_decoder) == 3
            latent_shape = resolution_decoder

            #upsampling
            if(self.aleatoric_uncertainty and not variational_iDFT):#insert a new flag! instead of combination of flags
                x = DenseVariational(latent_shape[0]*latent_shape[1]*latent_shape[2])(x)
                #x = tfp.layers.DenseFlipout(latent_shape[0]*latent_shape[1]*latent_shape[2],
                #                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
            else:
                x = tf.keras.layers.Dense(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
            x = tf.keras.layers.Reshape(latent_shape)(x)
        if(DFT and not variational_random_padding):#if random padding of frequencies there should not be special waves defined
            assert not time_length>1, "This layers are not set for time length > 1"
            #convert to Discrete cosine transform low resolution coefficients
            x = DCT3D(latent_shape[0], latent_shape[1], latent_shape[2])(x)
        if(inverse_DFT):
            assert not time_length>1, "This layers are not set for time length > 1"
            if(variational_iDFT):
                assert type(variational_coefs) is tuple
                x = variational_iDCT3D(*(latent_shape + self.fmri_ae.in_shape[:3] + variational_coefs), 
                                        coefs_perturb=True, dependent=variational_iDFT_dependent, 
                                        posterior_dimension=variational_iDFT_dependent_dim, distribution=variational_dist,
                                        random_padding=variational_random_padding)(x)
            else:
                x = padded_iDCT3D(latent_shape[0], latent_shape[1], latent_shape[2],
                            out1=self.fmri_ae.in_shape[0], out2=self.fmri_ae.in_shape[1], out3=self.fmri_ae.in_shape[2])(x)
        elif(self.aleatoric_uncertainty):
            x = tf.keras.layers.Flatten()(x)
            x =  DenseVariational(self.fmri_ae.in_shape[0]*self.fmri_ae.in_shape[1]*self.fmri_ae.in_shape[2])(x)
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

        output=x
        if(self.aleatoric_uncertainty):
            output=[output]+[tf.keras.layers.Dense(1, activation=tf.keras.activations.exponential)(x)]

        self.decoder = tf.keras.Model(input_shape, output)

    def build(self, input_shape1, input_shape2):
        self.eeg_encoder.build(input_shape=input_shape1)
        self.decoder.build(input_shape=input_shape1)

        self.fmri_encoder.build(input_shape=input_shape2)

        self.built=True
        
        self.trainable_variables.append(self.fmri_encoder.trainable_variables)


    
    @tf.function(input_signature=[tf.TensorSpec([None,64,134,10,1], tf.float32), tf.TensorSpec([None,64,64,30,1], tf.float32)], reduce_retracing=True)
    def call(self, x1, x2):
        """
        Random behaviour of GPU with tf functions does not reproduce the same results
        Call this function when getting results
        """
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
                "batch_norm_reg": self.batch_norm_reg,
                "dropout": self.dropout,
                "time_length": self.time_length,
                "local": self.local,
                "fourier_features": self.fourier_features,
                "fourier_normalization": self.fourier_normalization,
                "conditional_attention_style": self.conditional_attention_style,
                "conditional_attention_style_prior": self.conditional_attention_style_prior,
                "random_fourier": self.random_fourier,
                "inverse_DFT": self.inverse_DFT,
                "DFT": self.DFT,
                "variational_iDFT": self.variational_iDFT,
                "variational_coefs": self.variational_coefs,
                "variational_iDFT_dependent": self.variational_iDFT_dependent,
                "variational_iDFT_dependent_dim": self.variational_iDFT_dependent_dim,
                "variational_dist": self.variational_dist,
                "variational_random_padding": self.variational_random_padding,
                "resolution_decoder": self.resolution_decoder,
                "consistency": self.consistency,
                "low_resolution_decoder": self.low_resolution_decoder,
                "topographical_attention": self.topographical_attention,
                "organize_channels": self.organize_channels,
                "aleatoric_uncertainty": self.aleatoric_uncertainty,
                "seed": self.seed,
                "fmri_args": self.fmri_args}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

custom_objects={"Topographical_Attention": Topographical_Attention,
                "EEG_to_fMRI": EEG_to_fMRI,
                "ResBlock": ResBlock,
                "fMRI_AE": fmri_ae.fMRI_AE,
                "padded_iDCT3D": padded_iDCT3D, 
                "DCT3D": DCT3D, 
                "variational_iDCT3D": variational_iDCT3D, 
                "iDCT3D": iDCT3D,
                "RandomFourierFeatures": RandomFourierFeatures,
                "Style": Style,
                "Latent_EEG_Spatial_Attention": Latent_EEG_Spatial_Attention,
                "Latent_fMRI_Spatial_Attention": Latent_fMRI_Spatial_Attention,
                "DenseTemporal": DenseTemporal,
                "DenseVariational": DenseVariational,
                "InOfDistribution": InOfDistribution,
                "MaxBatchNorm": MaxBatchNorm,
                "MaxNormalization": MaxNormalization,
                "Sinusoids": Sinusoids,}


class pretrained_EEG_to_fMRI(tf.keras.Model):
    """
    pretrained_EEG_to_fMRI
    """

    def __init__(self, model, input_shape, activation=tf.keras.activations.linear, regularizer=None, feature_selection=False, segmentation_mask=False, latent_contrastive=False, organize_channels=False, seed=None):
        """
        init method
        """

        super(pretrained_EEG_to_fMRI, self).__init__()

        if(feature_selection):
            print("WARNING: Feature selection is deprecated for "+pretrained_EEG_to_fMRI.__name__)
        if(segmentation_mask):
            print("WARNING: Segmentation mask is deprecated for "+pretrained_EEG_to_fMRI.__name__)

        self._input_shape=input_shape
        self.feature_selection=feature_selection
        self.segmentation_mask=segmentation_mask
        self.latent_contrastive=latent_contrastive
        self.organize_channels=organize_channels

        input_shape, x, attention_scores = self.build_encoder(model, activation=activation, regularizer=regularizer, organize_channels=organize_channels, seed=seed)
        
        self.build_decoder(model, input_shape, x, activation=activation, attention_scores=attention_scores, regularizer=regularizer, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed)
        

    def build_encoder(self, pretrained_model, activation=None, regularizer=None, organize_channels=False, seed=None):

        attention_scores=None
        input_shape = tf.keras.layers.Input(shape=self._input_shape)

        x = input_shape
        #reshape to flattened features to apply attention mechanism
        x = tf.keras.layers.Reshape((self._input_shape[0], self._input_shape[1]*self._input_shape[2]))(x)
        #topographical attention
        #x, attention_scores = Topographical_Attention(self._input_shape[0], self._input_shape[1]*self._input_shape[2], regularizer=regularizer)(x)
        #if(organize_channels):
        #    attention_scores = Topographical_Attention_Scores_Regularization()(attention_scores)

        #reshape back to original shape
        x = tf.keras.layers.Reshape(self._input_shape)(x)
        previous_block_x = x
        
        #set the rest of the layers, but untrainable
        resblocks = pretrained_model.layers[1].layers[4:-3]

        if(self._input_shape[0]==132):
            resblocks[0].left_layers[0].strides=(3,)+resblocks[0].left_layers[0].strides[1:]
            resblocks[0].right_layers[0].strides=(3,)+resblocks[0].right_layers[0].strides[1:]
            resblocks[1].left_layers[0].strides=(5,)+resblocks[1].left_layers[0].strides[1:]
            resblocks[1].right_layers[0].strides=(5,)+resblocks[1].right_layers[0].strides[1:]
        for i in range(len(resblocks)):
            #change stride size according to number of channels
            if(self._input_shape[0]==41):#for dataset 11
                resblocks[i].left_layers[0].strides=(2,)+resblocks[i].left_layers[0].strides[1:]
                resblocks[i].right_layers[0].strides=(2,)+resblocks[i].right_layers[0].strides[1:]
                x = tf.keras.layers.ZeroPadding3D(padding=((0,2), (0,0), (0,0)))(x)
            elif(self._input_shape[0]==19):#for dataset 15
                resblocks[i].left_layers[0].strides=(2,)+resblocks[i].left_layers[0].strides[1:]
                resblocks[i].right_layers[0].strides=(2,)+resblocks[i].right_layers[0].strides[1:]
                resblocks[i].left_layers[0].kernel_size=(3,)+resblocks[i].left_layers[0].kernel_size[1:]
                resblocks[i].right_layers[0].kernel_size=(3,)+resblocks[i].right_layers[0].kernel_size[1:]
            x = pretrained_ResBlock(resblocks[i], trainable=True, activation=activation, regularizer=regularizer, seed=seed)(x)
            x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(pretrained_model.layers[1].layers[-2].units,
                                activation=activation,#adapt to distribution learned by random fourier
                                kernel_regularizer=regularizer,
                                bias_regularizer=regularizer,
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),#tf.constant_initializer(pretrained_model.layers[1].layers[-2].kernel.numpy()),
                                bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed),#tf.constant_initializer(pretrained_model.layers[1].layers[-2].bias.numpy()),
                                trainable=True)(x)#placeholder
        #x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Reshape(pretrained_model.layers[1].layers[-1].target_shape)(x)

        x = tf.keras.layers.Flatten()(x)

        if("Fourier" in type(pretrained_model.layers[3]).__name__):
            self.index_model=4
            if("Fourier" in type(pretrained_model.layers[self.index_model].layers[10]).__name__):
                index=10
        elif("Fourier" in type(pretrained_model.layers[3].layers[10]).__name__):
            self.index_model=3
            index=10
        elif("Dense" in type(pretrained_model.layers[3].layers[10]).__name__):
            self.index_model=3
            index=10
        else:
            raise NotImplementedError

        if("Fourier" in type(pretrained_model.layers[self.index_model].layers[index]).__name__):
            x = globals()[type(pretrained_model.layers[self.index_model].layers[index]).__name__](
                                            pretrained_model.layers[self.index_model].layers[index].units,
                                            normalization=pretrained_model.layers[self.index_model].layers[index].normalization,
                                            scale=pretrained_model.layers[self.index_model].layers[index].kernel_scale.numpy(),
                                            batch_norm_reg=pretrained_model.layers[self.index_model].layers[index].batch_norm_reg,
                                            consistency=pretrained_model.layers[self.index_model].layers[index].consistency,
                                            trainable=False, name="latent_projection")(x)
        else:
            x = globals()[type(pretrained_model.layers[self.index_model].layers[index]).__name__](
                                                pretrained_model.layers[self.index_model].layers[index].units,
                                                kernel_regularizer=regularizer,
                                                bias_regularizer=regularizer,
                                                trainable=False, name="latent_projection")(x)

        self.eeg_encoder = tf.keras.Model(input_shape, x)

        return input_shape, x, attention_scores
    
    def build_decoder(self, pretrained_model, input_shape, output_encoder, activation=None, attention_scores=None, regularizer=None, feature_selection=False, segmentation_mask=False, seed=None):
        x = output_encoder
        
        if("Fourier" in type(pretrained_model.layers[self.index_model].layers[10]).__name__):
            index=11
        elif("Dense" in type(pretrained_model.layers[self.index_model].layers[10]).__name__):
            index=11
        else:
            raise NotImplementedError

        #project sinusoids
        if("Sinusoids" in type(pretrained_model.layers[self.index_model].layers[index]).__name__):
            x=Sinusoids()(x)
            index+=1
        
        if(pretrained_model.layers[self.index_model].layers[index].name=="conditional_attention_style_dense"):
            attention_scores = tf.keras.layers.Flatten(name="conditional_attention_style_flatten")(attention_scores)
            self.latent_style = getattr(tf.keras.layers, type(pretrained_model.layers[self.index_model].layers[index]).__name__)(
                                        pretrained_model.layers[self.index_model].layers[index].units,
                                        activation=tf.keras.activations.linear,
                                        kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer,
                                        use_bias=pretrained_model.layers[self.index_model].layers[index].use_bias,
                                        name="conditional_attention_style_dense",
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                                        trainable=False)(attention_scores)
            #add style features from attention graph or prior learned style
            x = x*self.latent_style
        elif(pretrained_model.layers[self.index_model].layers[index].name=="style_prior"):
            x = Style(initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].latent_style.numpy()), trainable=False, seed=None, name='style_prior')(x)

        index+=1
        
        x = getattr(tf.keras.layers, type(pretrained_model.layers[self.index_model].layers[index]).__name__)(
                    pretrained_model.layers[self.index_model].layers[index].target_shape)(x)
        index+=1

        x = getattr(tf.keras.layers, type(pretrained_model.layers[self.index_model].layers[index]).__name__)()(x)
        index+=1

        #upsampling layer
        if(type(pretrained_model.layers[self.index_model].layers[index]).__name__=="Dense"):
            x = getattr(tf.keras.layers, type(pretrained_model.layers[self.index_model].layers[index]).__name__)(
                    pretrained_model.layers[self.index_model].layers[index].units,
                    activation=tf.keras.activations.linear,
                    kernel_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].kernel.numpy()),
                    bias_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].bias.numpy()),
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer,
                    trainable=False)(x)
        elif(type(pretrained_model.layers[self.index_model].layers[index]).__name__=="DenseVariational"):
            x = DenseVariational(pretrained_model.layers[self.index_model].layers[index].units,
                                activation=pretrained_model.layers[self.index_model].layers[index].activation_fn,
                                kernel_prior_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].kernel_mu.numpy()),
                                kernel_posterior_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].kernel_sigma.numpy()),
                                bias_prior_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].bias_mu.numpy()),
                                bias_posterior_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].bias_sigma.numpy()),
                                trainable=False)(x)
        else:
            raise NotImplementedError
        
        index+=1

        if(type(pretrained_model.layers[self.index_model].layers[index+1]).__name__=="DCT3D"):
            #reshape
            x = getattr(tf.keras.layers, type(pretrained_model.layers[self.index_model].layers[index]).__name__)(
                        pretrained_model.layers[self.index_model].layers[index].target_shape)(x)
            index+=1

            #initialize DCT3D layer
            x = DCT3D(**pretrained_model.layers[self.index_model].layers[index].get_config())(x)
            #remove this
            index+=1

            if(type(pretrained_model.layers[self.index_model].layers[index]).__name__=="variational_iDCT3D"):
                x = variational_iDCT3D(**pretrained_model.layers[self.index_model].layers[index].get_config(), 
                                        normal_loc_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].normal.distribution.loc.numpy()),
                                        normal_scale_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].normal.distribution.scale.numpy()),
                                        w1_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].w1.numpy()),
                                        w2_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].w2.numpy()),
                                        w3_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].w3.numpy()),
                                        loc_posterior_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].loc.numpy()),
                                        scale_posterior_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].scale.numpy()),
                                        biases_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].biases.numpy()),
                                        trainable=False)(x)
            elif(type(pretrained_model.layers[self.index_model].layers[index]).__name__=="padded_iDCT3D"):
                x = padded_iDCT3D(**pretrained_model.layers[self.index_model].layers[index].get_config())(x)
            index+=1



        x = getattr(tf.keras.layers, type(pretrained_model.layers[self.index_model].layers[index]).__name__)(
            pretrained_model.layers[self.index_model].layers[index].target_shape)(x)

        #feature selection occurs here
        z = None
        if(feature_selection):
            z = tf.keras.layers.ReLU()(x)

        #remove this
        index+=1
        x = getattr(tf.keras.layers, type(pretrained_model.layers[self.index_model].layers[index]).__name__)(
                                        filters=pretrained_model.layers[self.index_model].layers[index].filters, 
                                        kernel_size=pretrained_model.layers[self.index_model].layers[index].kernel_size, 
                                        strides=pretrained_model.layers[self.index_model].layers[index].strides,
                                        kernel_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].kernel.numpy()),
                                        bias_initializer=tf.constant_initializer(pretrained_model.layers[self.index_model].layers[index].bias.numpy()),
                                        padding=pretrained_model.layers[self.index_model].layers[index].padding,
                                        trainable=False)(x)

        self.q_decoder = tf.keras.Model(input_shape, x)
        

    def build(self, input_shape):
        self.eeg_encoder.build(input_shape=input_shape)
        self.q_decoder.build(input_shape=input_shape)
        self.built=True

    #@tf.function(input_signature=[tf.TensorSpec([None,64,134,10,1], tf.float32), tf.TensorSpec([None,64,64,30,1], tf.float32)])
    def call(self, x1, **kwargs):
        """
            Random behaviour of GPU with tf functions does not reproduce the same results
            Call this function when getting results
        """
        
        z = [self.q_decoder(x1)]

        if(self.latent_contrastive):
            z+=[self.eeg_encoder(x1)]

        return z