import tensorflow as tf

from layers.locally_connected import LocallyConnected3D

search_space = [{'name': 'learning_rate', 'type': 'continuous',
                    'domain': (1e-5, 1e-2)},
                    {'name': 'reg', 'type': 'continuous',
                    'domain': (1e-6, 1e-1)},
                   {'name': 'kernel_size', 'type': 'discrete',
                    'domain': (2,3)},
                   {'name': 'stride_size', 'type': 'discrete',
                    'domain': (1,)},
                   {'name': 'channels', 'type': 'discrete',
                    'domain': (4,8,16)},
                   {'name': 'latent_dimension', 'type': 'discrete',
                    'domain': (2,3,4,5)},
                   {'name': 'batch_norm', 'type': 'discrete',
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
    input_shape = kwargs[0]
    reg = float(kwargs[2])
    kernel_size = int(kwargs[3])
    stride_size = int(kwargs[4])
    n_channels = int(kwargs[5])
    latent_dimension = int(kwargs[6])
    batch_norm = bool(kwargs[7])
    dropout = float(kwargs[8])

    return fMRI_AE((latent_dimension,)*3, input_shape, kernel_size, stride_size, n_channels)



def block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=True, batch_norm=True, weight_decay=0.000):

    x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.L2(weight_decay))(x)
    if(maxpool):
        x = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(1,1,1))(x)
    if(batch_norm):
        x = tf.keras.layers.BatchNormalization()(x)

    return tf.keras.layers.ReLU()(x)

def skip_block(x, skip_x, operation, kernel_size, stride_size, n_channels,
                maxpool=True, batch_norm=True, weight_decay=0.000):

    skip_x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.L2(weight_decay))(skip_x)
    if(batch_norm):
        skip_x = tf.keras.layers.BatchNormalization()(skip_x)

    x = tf.keras.layers.Add()([x, skip_x])

    return tf.keras.layers.ReLU()(x)

def stack(x, previous_block_x, operation, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False):
    
    x = block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay)

    x = block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay)

    #skip connection
    if(skip_connections):
        if(maxpool):
            skip_kernel = (kernel_size[0]*2+1, kernel_size[1]*2+1, kernel_size[2]*2+1)
        else:
            skip_kernel = (kernel_size[0]*2-1, kernel_size[1]*2-1, kernel_size[2]*2-1)
        x = skip_block(x, previous_block_x, operation, 
                        skip_kernel, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay)

    return x


class fMRI_AE(tf.keras.Model):
    
    def __init__(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False,
                        n_stacks=2, local=True, local_attention=False):
        
        
        super(fMRI_AE, self).__init__()
        
        self.build_encoder(latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, skip_connections=skip_connections,
                        n_stacks=n_stacks, local=local, local_attention=local_attention)

        self.build_decoder()
    
    def build_encoder(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False,
                        n_stacks=2, local=True, local_attention=False):

        self.latent_shape = latent_shape
        self.in_shape = input_shape


        input_shape = tf.keras.layers.Input(shape=input_shape)
        
        x = input_shape
        previous_block_x = input_shape

        for i in range(n_stacks):
            x = stack(x, previous_block_x, tf.keras.layers.Conv3D, 
                        kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, 
                        skip_connections=skip_connections)
            previous_block_x=x

        if(local):
            operation=tf.keras.layers.Conv3D
        else:
            operation=LocallyConnected3D

        x = block(x, operation, kernel_size, stride_size, n_channels,
                maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.latent_shape[0]*self.latent_shape[1]*self.latent_shape[2])(x)
        x = tf.keras.layers.Reshape(self.latent_shape)(x)

        if(local_attention):
            x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1, 2, 3))(x,x)
        
        self.encoder = tf.keras.Model(input_shape, x)

    def build_decoder(self):
        self.decoder = tf.keras.Sequential([
                                            tf.keras.layers.InputLayer(input_shape=self.latent_shape),
                                            tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(self.in_shape[0]*self.in_shape[1]*self.in_shape[2]),
                                            tf.keras.layers.Reshape(self.in_shape)
                                        ])

    def encode(self, X):
        return self.encoder(X)
    
    def decode(self, Z):
        return self.decoder(Z)
    
    def call(self, X):
        if(not self.encoder.built):
            self.encoder.build(X.shape)

        return self.decode(self.encode(X))
