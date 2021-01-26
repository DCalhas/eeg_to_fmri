import tensorflow as tf


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


	def __init__(self, encoder, decoder):
		super(EEG_to_fMRI, self).__init__()

		self.encoder = encoder
		self.decoder = decoder

	def encode(self, X):
		return self.encoder(X)
	
	def decode(self, Z):
		return self.decoder(Z)

	def call(self, X):
		return self.decode(self.encode(X))