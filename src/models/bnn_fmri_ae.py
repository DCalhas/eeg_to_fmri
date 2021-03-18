import tensorflow as tf
import tensorflow_probability as tfp

from layers.locally_connected import LocallyConnected3D



def block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=True, batch_norm=True):

    x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size)(x)
    if(maxpool):
        x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1), strides=(1,1,1))(x)
    if(batch_norm):
        x = tf.keras.layers.BatchNormalization()(x)

    return tf.keras.layers.ReLU()(x)

def skip_block(x, skip_x, operation, kernel_size, stride_size, n_channels,
                maxpool=True, batch_norm=True):

    skip_x = operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size)(skip_x)
    if(batch_norm):
        skip_x = tf.keras.layers.BatchNormalization()(skip_x)

    x = tf.keras.layers.Add()([x, skip_x])

    return tf.keras.layers.ReLU()(x)

def stack(x, previous_block_x, operation, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, skip_connections=False):
    
    x = block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=maxpool, batch_norm=batch_norm)

    x = block(x, operation, kernel_size, stride_size, n_channels,
            maxpool=maxpool, batch_norm=batch_norm)

    #skip connection
    if(skip_connections):
        if(maxpool):
            skip_kernel = (kernel_size[0]*2+1, kernel_size[1]*2+1, kernel_size[2]*2-1)
        else:
            skip_kernel = (kernel_size[0]*2-1, kernel_size[1]*2-1, kernel_size[2]*2-1)
        x = skip_block(x, previous_block_x, operation, 
                        skip_kernel, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm)
        
    return x




class BNN_fMRI_AE(tf.keras.Model):
    
    def __init__(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, skip_connections=False,
                        n_stacks=2, local=True, local_attention=False, outfilter=0):
        
        
        super(BNN_fMRI_AE, self).__init__()
        
        self.build_encoder(latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm, skip_connections=skip_connections,
                        n_stacks=n_stacks, local=local, local_attention=local_attention)

        self.build_decoder(outfilter=outfilter)
    
    def build_encoder(self, latent_shape, input_shape, kernel_size, stride_size, n_channels,
                        maxpool=True, batch_norm=True, skip_connections=False,
                        n_stacks=2, local=True, local_attention=False):

        self.latent_shape = latent_shape
        self.in_shape = input_shape


        input_shape = tf.keras.layers.Input(shape=input_shape)

        self._input_tensor = input_shape
        
        x = input_shape
        previous_block_x = input_shape

        for i in range(n_stacks):
            x = stack(x, previous_block_x, tfp.layers.Convolution3D,#Flipout, 
                        kernel_size, stride_size, n_channels,
                        maxpool=maxpool, batch_norm=batch_norm,
                        skip_connections=skip_connections)
            previous_block_x=x

        if(local):
            #operation=tfp.layers.Convolution3DFlipout
            operation=tfp.layers.Convolution3D
        else:
            operation=LocallyConnected3D

        x = block(x, operation, (7,7,7), stride_size, n_channels,
                maxpool=maxpool, batch_norm=batch_norm)

        x = tf.keras.layers.Flatten()(x)
        #x = tfp.layers.DenseFlipout(self.latent_shape[0]*self.latent_shape[1]*self.latent_shape[2])(x)
        x = tfp.layers.Dense(self.latent_shape[0]*self.latent_shape[1]*self.latent_shape[2])(x)
        x = tf.keras.layers.Reshape(self.latent_shape)(x)

        if(local_attention):
            #x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1, 2, 3))(x,x)
            x = tf.keras.layers.MultiHeadAttention(num_heads=n_channels, key_dim=x.shape[1]*x.shape[2]*x.shape[3], attention_axes=(1, 2, 3))(x,x)
        
        self.output_encoder = x

    def build_decoder(self, outfilter=0):

        x = tf.keras.layers.Flatten()(self.output_encoder)

        #upsampling
        x = tfp.layers.DenseFlipout(self.in_shape[0]*self.in_shape[1]*self.in_shape[2])(x)
        x = tf.keras.layers.Reshape(self.in_shape)(x)

        #filter
        if(outfilter == 1):
            x = tfp.layers.Convolution3DFlipout(filters=1, kernel_size=1, strides=1)(x)
        elif(outfilter == 2):
            x = LocallyConnected3D(filters=1, kernel_size=1, strides=1, implementation=3)(x)
        

        #variance computation along with regression
        variance = tf.keras.layers.Dense(1)(x)

        self.model = tf.keras.Model(inputs=self._input_tensor, outputs=[x,variance])

    
    def call(self, X, training=True, T=10):
        if(not self.model.built):
            self.model.build(X.shape)

        if(not training):
            self.monte_carlo_prediction(X, T=10)

        return self.model(X)

    def monte_carlo_prediction(self, X, T=10):
        y_hat = tf.zeros(tf.shape(X))
        
        for i in range(T):
            y_hat = y_hat + self.model(X)[0]

        return y_hat/T
