import tensorflow as tf

from models import fmri_ae


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


def build(*kwargs):
	return EEG_to_fMRI()


"""
This class implements an architecture for EEG to fMRI transcription

encode: architecture that encodes the EEG signal to a space where an instance of fMRI is also represented

decode: architecture that maps the encoded representation to the fMRI space representation

call: encode and decode

"""

class EEG_to_fMRI(tf.keras.Model):
    
    def __init__(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                maxpool=True, weight_decay=0.000, 
                skip_connections=False, batch_norm=True,
                dropout=False, n_stacks=2, local=True, 
                seed=None, fmri_args=None):
        super(EEG_to_fMRI, self).__init__()
        
        self.fmri_ae = fmri_ae.fMRI_AE(*fmri_args)
        
        self.build_encoder(latent_shape, input_shape, kernel_size, 
                            stride_size, n_channels, 
                            maxpool=maxpool, dropout=dropout,
                            weight_decay=weight_decay, skip_connections=skip_connections,
                            n_stacks=n_stacks, local=local, 
                            batch_norm=batch_norm, seed=seed)
        self.build_decoder()
        
    def build_encoder(self, latent_shape, input_shape, kernel_size, 
                            stride_size, n_channels, 
                            maxpool=True, dropout=False,
                            weight_decay=0.000, skip_connections=False,
                            batch_norm=True, n_stacks=2, 
                            local=True, seed=None):
        
        input_shape = tf.keras.layers.Input(shape=input_shape)
        
        x = input_shape
        previous_block_x = input_shape

        for i in range(n_stacks):
            x = fmri_ae.stack(x, previous_block_x, tf.keras.layers.Conv3D, 
                        kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, 
                        skip_connections=skip_connections, seed=seed)
            previous_block_x=x

        if(local):
            operation=tf.keras.layers.Conv3D
        else:
            operation=LocallyConnected3D

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.experimental.RandomFourierFeatures(latent_shape[0]*latent_shape[1]*latent_shape[2],
                                                              trainable=True)(x)
        
        if(dropout):
            x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Reshape(latent_shape)(x)
        
        self.eeg_encoder = tf.keras.Model(input_shape, x)
        self.fmri_encoder = self.fmri_ae.encoder
        
    def build_decoder(self):
        self.decoder = self.fmri_ae.decoder
        
    def build(self, input_shape1, input_shape2):
        self.eeg_encoder.build(input_shape=input_shape1)
        
        self.fmri_ae.build(input_shape=input_shape2)        
        self.fmri_encoder.build(input_shape=input_shape2)
        
        self.built=True
    
    def call(self, X, training=True):
        x1, x2 = X
        
        z1 = self.eeg_encoder(x1)
        z2 = self.fmri_encoder(x2)
        
        if(training):
            return [self.decoder(z1), z1, z2]
        return self.decoder(z1)