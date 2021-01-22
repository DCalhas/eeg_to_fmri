import tensorflow as tf

search_space = [{'name': 'learning_rate', 'type': 'continuous',
                    'domain': (1e-5, 1e-2)},
                    {'name': 'reg', 'type': 'continuous',
                    'domain': (1e-6, 1e-1)},
                   {'name': 'kernel_size', 'type': 'discrete',
                    'domain': (2,3,4)},
                   {'name': 'stride_size', 'type': 'discrete',
                    'domain': (1,2)},
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

class fMRI_AE(tf.keras.Model):
    
    def __init__(self, latent_shape, input_shape, kernel_size, stride_size, n_channels):
        
        
        super(fMRI_AE, self).__init__()
        
        self.latent_shape = latent_shape
        self.in_shape = input_shape
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv3D(filters=n_channels, kernel_size=kernel_size, strides=stride_size, activation="relu"),
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(1,1,1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=n_channels, kernel_size=kernel_size, strides=stride_size, activation="relu"),
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(1,1,1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=n_channels, kernel_size=kernel_size, strides=stride_size, activation="relu"),
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(1,1,1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_shape[0]*self.latent_shape[1]*self.latent_shape[2]),
            tf.keras.layers.Reshape(self.latent_shape)
        ])
        
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
        return self.decode(self.encode(X))
