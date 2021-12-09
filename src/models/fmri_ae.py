import tensorflow as tf
import tensorflow_probability as tfp

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

def block18(x, operation, kernel_size, stride_size, n_channels,
            maxpool=True, batch_norm=True, weight_decay=0.000,  padding="valid",
            seed=None):

    x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.L2(weight_decay),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                    padding=padding)(x)
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


def skip_block18(x, skip_x, operation, kernel_size, stride_size, n_channels,
                maxpool=True, batch_norm=True, weight_decay=0.000, padding="valid",
                seed=None):

    skip_x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
                    kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                    bias_regularizer=tf.keras.regularizers.L2(weight_decay),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                    padding=padding)(skip_x)

    if(maxpool):
        skip_x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1), strides=(1,1,1))(skip_x)
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


def stack18(x, previous_block_x, operation, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False,
                        seed=None):
    #downsampling block    
    x = block18(x, operation, kernel_size, stride_size, n_channels,
            maxpool=maxpool, batch_norm=batch_norm, 
            weight_decay=weight_decay, padding="valid",
            seed=seed)

    #non downsampling block
    x = block18(x, operation, 3, 1, n_channels,
            maxpool=False, batch_norm=batch_norm, 
            weight_decay=weight_decay, padding="same",
            seed=seed)
    
    #skip connection
    if(skip_connections):
        x = skip_block18(x, previous_block_x, operation, 
                        kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, 
                        weight_decay=weight_decay, padding="valid",
                        seed=seed)
        
    return x


class fMRI_AE(tf.keras.Model):
    
    def __init__(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False,
                        n_stacks=2, local=True, local_attention=False, outfilter=0, dropout=False, seed=None, _build_decoder=True):
        
        
        super(fMRI_AE, self).__init__()
        
        self.latent_shape=latent_shape
        self._input_shape=input_shape
        self.kernel_size=kernel_size
        self.stride_size=stride_size
        self.n_channels=n_channels
        self.maxpool=maxpool
        self.batch_norm=batch_norm
        self.weight_decay=weight_decay
        self.skip_connections=skip_connections
        self.n_stacks=n_stacks
        self.local=local
        self.local_attention=local_attention
        self.outfilter=outfilter
        self.dropout=dropout
        self.seed=seed
        self.latent_shape = latent_shape
        self.in_shape = input_shape
        
        self.build_encoder(latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, skip_connections=skip_connections,
                        n_stacks=n_stacks, local=local, local_attention=local_attention, dropout=dropout, seed=seed)
        if(_build_decoder):
            self.build_decoder(outfilter=outfilter, seed=seed)
    
    def build_encoder(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False,
                        n_stacks=2, local=True, local_attention=False, dropout=False, seed=None):

        input_shape = tf.keras.layers.Input(shape=input_shape)
        
        x = input_shape
        previous_block_x = input_shape

        for i in range(n_stacks):
            #x = stack(x, previous_block_x, tf.keras.layers.Conv3D, 
            x = stack(x, previous_block_x, tf.keras.layers.Conv3D, 
                        kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, 
                        skip_connections=skip_connections, seed=seed)
            previous_block_x=x

        if(local):
            operation=tf.keras.layers.Conv3D
        else:
            operation=LocallyConnected3D

        #x = block(x, operation, (7,7,7), stride_size, n_channels,
        #x = block(x, operation, (7,7,7), stride_size, n_channels,
        #        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, seed=seed)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.latent_shape[0]*self.latent_shape[1]*self.latent_shape[2], 
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        if(dropout):
            x = tf.keras.layers.Dropout(0.5)(x)
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

    def get_config(self):
        return {"latent_shape": self.latent_shape,
                "input_shape": self._input_shape,
                "kernel_size": self.kernel_size,
                "stride_size": self.stride_size,
                "n_channels": self.n_channels,
                "maxpool": self.maxpool,
                "batch_norm": self.batch_norm,
                "weight_decay": self.weight_decay,
                "skip_connections": self.skip_connections,
                "n_stacks": self.n_stacks,
                "local": self.local,
                "local_attention": self.local_attention,
                "outfilter": self.outfilter,
                "dropout": self.dropout,
                "seed": self.seed}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BNN_fMRI_AE(tf.keras.Model):
    
    def __init__(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, weight_decay=0.000, skip_connections=False,
                        n_stacks=2, local=True, local_attention=False, outfilter=0, seed=None):
        
        
        super(BNN_fMRI_AE, self).__init__()
        
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

        self._input_tensor = input_shape
        
        x = input_shape
        previous_block_x = input_shape

        for i in range(n_stacks):
            x = stack(x, previous_block_x, tfp.layers.Convolution3DFlipout, 
                        kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, 
                        skip_connections=skip_connections, seed=seed)
            previous_block_x=x

        if(local):
            operation=tfp.layers.Convolution3DFlipout
        else:
            operation=LocallyConnected3D

        x = block(x, operation, (7,7,7), stride_size, n_channels,
                maxpool=maxpool, batch_norm=batch_norm, weight_decay=weight_decay, seed=seed)

        x = tf.keras.layers.Flatten()(x)
        x = tfp.layers.DenseFlipout(self.latent_shape[0]*self.latent_shape[1]*self.latent_shape[2])(x)
        x = tf.keras.layers.Reshape(self.latent_shape)(x)

        if(local_attention):
            #x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1, 2, 3))(x,x)
            x = tf.keras.layers.MultiHeadAttention(num_heads=n_channels, key_dim=x.shape[1]*x.shape[2]*x.shape[3], attention_axes=(1, 2, 3),
                                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x,x)
        
        self.output_encoder = x

    def build_decoder(self, outfilter=0, seed=None):

        x = tf.keras.layers.Flatten()(self.output_encoder)

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
        

        #variance computation along with regression
        variance_pre = tf.keras.layers.Dense(1)(x)
        variance = tf.keras.layers.Activation('softplus', name='variance')(variance_pre)

        self.model = tf.keras.Model(inputs=self._input_tensor, outputs=[x,variance])

    
    def call(self, X):
        if(not self.model.built):
            self.model.build(X.shape)

        return self.model(X)
