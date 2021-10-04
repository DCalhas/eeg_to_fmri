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


def block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=True, batch_norm=True, weight_decay=0.000,  padding="valid",
            maxpool_k=None, maxpool_s=None,
            seed=None):

    x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.L2(weight_decay),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                    padding=padding)(x)
    if(maxpool):
        x = tf.keras.layers.MaxPool3D(pool_size=maxpool_k, strides=maxpool_s)(x)
    if(batch_norm):
        x = tf.keras.layers.BatchNormalization()(x)

    return tf.keras.layers.ReLU()(x)


def skip_block(x, skip_x, operation, kernel_size, stride_size, n_channels,
                maxpool=True, batch_norm=True, weight_decay=0.000, padding="valid",
                maxpool_k=None, maxpool_s=None,
                seed=None):

    skip_x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.L2(weight_decay),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                    padding=padding)(skip_x)

    if(maxpool):
        skip_x = tf.keras.layers.MaxPool3D(pool_size=maxpool_k, strides=maxpool_s)(skip_x)
    if(batch_norm):
        skip_x = tf.keras.layers.BatchNormalization()(skip_x)

    x = tf.keras.layers.Add()([x, skip_x])

    return tf.keras.layers.ReLU()(x)

def stack(x, previous_block_x, operation, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, 
                        weight_decay=0.000, skip_connections=False,
                        maxpool_k=None, maxpool_s=None,
                        seed=None):
    #downsampling block    
    x = block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=maxpool, batch_norm=batch_norm, 
            maxpool_k=maxpool_k, maxpool_s=maxpool_s,
            weight_decay=weight_decay, padding="valid",
            seed=seed)

    #non downsampling block
    x = block(x, operation, 3, 1, n_channels,
            maxpool=False, batch_norm=batch_norm, 
            weight_decay=weight_decay, padding="same",
            seed=seed)

    #skip connection
    if(skip_connections):
        x = skip_block(x, previous_block_x, operation, 
                        kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm,
                        maxpool_k=maxpool_k, maxpool_s=maxpool_s,
                        weight_decay=weight_decay, padding="valid",
                        seed=seed)

    return x



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
                dropout=False, local=True, seed=None, fmri_args=None):
        super(EEG_to_fMRI, self).__init__()

        self.fmri_ae = fmri_ae.fMRI_AE(*fmri_args)

        self.build_encoder(latent_shape, input_shape, na_spec, n_channels, 
                            dropout=dropout, weight_decay=weight_decay, 
                            skip_connections=skip_connections, local=local, 
                            batch_norm=batch_norm, seed=seed)
        self.build_decoder()

    def build_encoder(self, latent_shape, input_shape, na_spec, n_channels, 
                            dropout=False, weight_decay=0.000, 
                            skip_connections=False, batch_norm=True, 
                            local=True, seed=None):

        input_shape = tf.keras.layers.Input(shape=input_shape)

        x = input_shape
        previous_block_x = input_shape

        for i in range(len(na_spec[0])):
            x = stack(x, previous_block_x, tf.keras.layers.Conv3D, 
                        na_spec[0][i], na_spec[1][i], n_channels,
                        maxpool=na_spec[2], batch_norm=batch_norm, weight_decay=weight_decay, 
                        maxpool_k=na_spec[3], maxpool_s=na_spec[4],
                        skip_connections=skip_connections, seed=seed)
            previous_block_x=x

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
        print(self.fmri_encoder.trainable_variables)
        print(self.decoder.trainable_variables)
        self.trainable_variables = self.trainable_variables.append(self.fmri_encoder.trainable_variables).append(self.decoder.trainable_variables)

    def call(self, X, training=True):
        x1, x2 = X

        z1 = self.eeg_encoder(x1)
        z2 = self.fmri_encoder(x2)

        if(training):
            return [self.decoder(z1), z1, z2]
        return self.decoder(z1)