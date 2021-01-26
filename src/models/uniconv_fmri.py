import tensorflow as tf

from models.eeg_to_fmri import EEG_to_fMRI


search_space = [{'name': 'learning_rate', 'type': 'continuous',
					'domain': (1e-5, 1e-2)},
					{'name': 'reg', 'type': 'continuous',
					'domain': (1e-6, 1e-1)},
					{'name': 'eeg_architecture', 'type': 'discrete',
					'domain': tuple(range(20))},
				   	{'name': 'epochs', 'type': 'discrete',
					'domain': (5,10,15,20,25,30)},
					{'name': 'batch_size', 'type': 'discrete',
					'domain': (2, 4, 8, 16, 32)}]


def build(*kwargs):
	print(kwargs)
	encoder = None
	decoder = None
	return EEGUniConv_to_fMRI(encoder, decoder)


"""
This class implements an architecture for EEG to fMRI transcription

encode: architecture that encodes the EEG signal to a space where an instance of fMRI is also represented

decode: architecture that maps the encoded representation to the fMRI space representation

call: encode and decode

"""
class EEGUniConv_to_fMRI(EEG_to_fMRI):

	def __init__(self, encoder, decoder):
		super(EEGUniConv_to_fMRI, self).__init__(encoder, decoder)