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
            maxpool=True, batch_norm=True, weight_decay=0.000,
            seed=None):

    x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.L2(weight_decay),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    if(maxpool):
        x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1), strides=(1,1,1))(x)
    if(batch_norm):
        x = tf.keras.layers.BatchNormalization()(x)

    return tf.keras.layers.ReLU()(x)

def skip_block(x, skip_x, operation, kernel_size, stride_size, n_channels,
                maxpool=True, batch_norm=True, weight_decay=0.000,
                seed=None):

    skip_x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.L2(weight_decay),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(skip_x)
    if(batch_norm):
        skip_x = tf.keras.layers.BatchNormalization()(skip_x)

    x = tf.keras.layers.Add()([x, skip_x])

    return tf.keras.layers.ReLU()(x)

def stack(x, previous_block_x, operation, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False,
                        seed=None):
    
    x = block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay,
            seed=seed)

    x = block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay,
            seed=seed)

    #skip connection
    if(skip_connections):
        if(maxpool):
            skip_kernel = (kernel_size[0]*2+1, kernel_size[1]*2+1, kernel_size[2]*2-1)
        else:
            skip_kernel = (kernel_size[0]*2-1, kernel_size[1]*2-1, kernel_size[2]*2-1)
        x = skip_block(x, previous_block_x, operation, 
                        skip_kernel, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay,
                        seed=seed)
        
    return x


class fMRI_AE(tf.keras.Model):
    
    def __init__(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False,
                        n_stacks=2, local=True, local_attention=False, outfilter=0, seed=None):
        
        
        super(fMRI_AE, self).__init__()
        
        self.build_encoder(latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, skip_connections=skip_connections,
                        n_stacks=n_stacks, local=local, local_attention=local_attention, seed=seed)

        self.build_decoder(outfilter=outfilter, seed=seed)
    
    def build_encoder(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False,
                        n_stacks=2, local=True, local_attention=False, seed=None):

        self.latent_shape = latent_shape
        self.in_shape = input_shape


        input_shape = tf.keras.layers.Input(shape=input_shape)
        
        x = input_shape
        previous_block_x = input_shape

        for i in range(n_stacks):
            x = stack(x, previous_block_x, tf.keras.layers.Conv3D, 
                        kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, 
                        skip_connections=skip_connections, seed=seed)
            previous_block_x=x

        if(local):
            operation=tf.keras.layers.Conv3D
        else:
            operation=LocallyConnected3D

        x = block(x, operation, (3,3,3), (1,1,1), n_channels,
                maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, seed=seed)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.latent_shape[0]*self.latent_shape[1]*self.latent_shape[2], 
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        x = tf.keras.layers.Reshape(self.latent_shape)(x)

        if(local_attention):
            #x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1, 2, 3))(x,x)
            x = tf.keras.layers.MultiHeadAttention(num_heads=n_channels, key_dim=x.shape[1]*x.shape[2]*x.shape[3], attention_axes=(1, 2, 3),
                                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x,x)
        
        self.encoder = tf.keras.Model(input_shape, x)

    def build_decoder(self, outfilter=0, seed=None):
        input_shape = tf.keras.layers.Input(shape=self.latent_shape)

        x = tf.keras.layers.Flatten()(input_shape)

        #upsampling
        x = tf.keras.layers.Dense(self.in_shape[0]*self.in_shape[1]*self.in_shape[2],
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        x = tf.keras.layers.Reshape(self.in_shape)(x)

        #filter
        if(outfilter == 1):
            x = tf.keras.layers.Conv3D(filters=1, kernel_size=1, strides=1,
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        elif(outfilter == 2):
            x = LocallyConnected3D(filters=1, kernel_size=1, strides=1, implementation=3,
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        
        self.decoder = tf.keras.Model(input_shape, x)        

    def encode(self, X):
        return self.encoder(X)
    
    def decode(self, Z):
        return self.decoder(Z)
    
    def call(self, X):
        if(not self.encoder.built):
            self.encoder.build(X.shape)

        return self.decode(self.encode(X))
