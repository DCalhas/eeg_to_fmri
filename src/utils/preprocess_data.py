from utils import data_utils, eeg_utils, fmri_utils

import sys

import os

import numpy as np

from sklearn.utils import shuffle

import tensorflow as tf

import gc

#should eeg_limit be true??
CHANNELS = {"all": None, "eeg": ("eeg",), "ecg": ("ecg",)}

def dataset_subjects(dataset, test_individuals, calib_individuals=(), n_individuals=None, interval_eeg=10, cache_dir=None,
						channels="eeg", eeg_norm="time", eeg_fmax=50., target="run_zscore", fmri_space="native", inject_snr=0., verbose=False):
	"""
	Subject-level split for cross-validation. fMRI: each subject's 4D run z-scored with one scalar mean/std (no IQR,
	no per-volume fit), as main.py. EEG: magnitude STFT with 1-TR non-overlapping windows, normalized per subject with
		eeg_norm="time": every channel and frequency z-scored over the recording
		eeg_norm="channels": legacy setting, every time-frequency bin z-scored across channels (scipy zscore axis=0)
	channels: "eeg" (EEG electrodes only), "ecg" (the ECG channel only, confound control) or "all" (legacy setting, includes ECG).
	eeg_fmax: highest STFT frequency kept, in Hz (bins of 1/TR Hz, DC dropped). None keeps the legacy 134 bins, which on
		ds002338 (1-s windows at 200 Hz) run past Nyquist into mirrored bins and the 50 Hz low-pass stopband.

	target: "run_zscore" (the z-scored run, dominated by the static anatomy) or "fluct" (the dynamics only: every
		subject's voxels minus their temporal mean, divided by their temporal std, zero outside the subject's anatomical
		Otsu mask; normalized with the subject's own run, like the run z-score). With "fluct" the output also has
		"mask", the Otsu mask of the training subjects' mean anatomy, for the evaluation.
		"fluct_resid": "fluct" minus the stimulus-locked average of the training subjects at every volume index (audit
		A17: ds002338 has identical block timing in every subject), i.e. the part of the dynamics task timing cannot explain.

	fmri_space: "native" (the original geometry: each subject's EPI resized to 64x64x30, no registration) or "mni" (review #1:
		the motion-corrected volumes registered to MNI152 at 3 mm by experiments/register_xp2_bold.py, x padded 60->64,
		y cropped 72->64 (4 slices each side, outside the brain), z averaged in pairs 60->30, i.e. 3x3x6 mm voxels),
		then the same run z-score / fluct targets. Same volumes (bold_shift onwards) and the same EEG.
	inject_snr: positive control (review #2b), with -target fluct/fluct_resid only. A known EEG-coupled signal, the
		negated C3/C4 8-30 Hz STFT power (mu/beta desynchronisation) convolved with a canonical HRF and z-scored per
		subject, is added with amplitude sqrt(inject_snr) to the unit-variance fluctuations of a fixed 8x8x4-voxel box
		(inside each subject's mask) in every subject; out["inject_mask"] marks the box for the evaluation.

	The per-subject STFT/fMRI arrays are cached in cache_dir, since loading is the slow part of each run.
	Inputs:
		* test_individuals, calib_individuals - lists of subject indices (0-based, sorted folder order)
	Outputs:
		* dict with keys "train", "calib", "test", each (eeg, fmri, subject_ids) where subject_ids index every instance
	"""
	if(n_individuals is None):
		n_individuals=getattr(data_utils, "n_individuals_"+dataset)
	n_volumes=getattr(fmri_utils, "n_volumes_"+dataset)
	if(dataset=="04"):
		#CHUR-XP1 replication: motor localizer, EEG windows aligned to the scanner pulses, registered volumes only
		assert fmri_space=="mni", "CHUR-XP1 is used on registered volumes only (-fmri_space mni)"
		n_volumes, n_individuals=XP1_N_VOLUMES, XP1_N_SUBJECTS

	assert not (channels=="ecg" and eeg_norm=="channels"), "the tiled ECG has zero variance across channels"
	f_limit_h=135 if eeg_fmax is None else int(round(eeg_fmax*getattr(fmri_utils, "TR_"+dataset)))+1
	#eeg_fmax=None reproduces the legacy 134 bins on purpose (beyond Nyquist on ds002338); a requested cut-off must not be
	assert eeg_fmax is None or f_limit_h-1 <= getattr(eeg_utils, "fs_"+dataset)*getattr(fmri_utils, "TR_"+dataset)/2, "STFT bins beyond Nyquist"
	cache_file=None if cache_dir is None else os.path.join(cache_dir, dataset+"_n"+str(n_individuals)+"_"+channels+"_"+eeg_norm+"_f"+str(f_limit_h-1)+".npz")
	if(dataset=="04"):
		eeg_all, fmri_all = xp1_motorloc(channels, eeg_norm, eeg_fmax, cache_dir)
	elif(cache_file is not None and os.path.exists(cache_file)):
		with np.load(cache_file) as f:
			eeg_all, fmri_all = f["eeg"], f["fmri"]
	else:
		eeg_limit=dataset in ["02","03","04","05"] or eeg_fmax is not None
		eeg_all, fmri_all, _ = data_utils.load_data(list(range(n_individuals)), raw_eeg=False, n_voxels=None,
														bold_shift=getattr(fmri_utils, "bold_shift_"+dataset), n_partitions=25,
														mutate_bands=False, by_partitions=False, partition_length=14,
														f_resample=getattr(fmri_utils, "TR_"+dataset), fmri_resolution_factor=1,
														standardize_eeg=True, standardize_fmri=True, ind_volume_fit=False, iqr_outlier=False,
														eeg_limit=eeg_limit, eeg_f_limit_h=f_limit_h, eeg_f_limit_l=0, dataset=dataset,
														channel_types=CHANNELS[channels], eeg_norm=eeg_norm)
		if(cache_file is not None):
			os.makedirs(cache_dir, exist_ok=True)
			np.savez(cache_file, eeg=eeg_all.astype('float32'), fmri=fmri_all.astype('float32'))

	#the searched EEG encoder (na_specification_2) down-samples a 134-bin frequency axis: the kept bins (normalized) are
	#zero-padded back to 134, so the bins above eeg_fmax carry no signal instead of mirrored/stopband noise
	if(eeg_all.shape[2] < 134):
		eeg_all = np.pad(eeg_all, ((0, 0), (0, 0), (0, 134-eeg_all.shape[2])))

	#subjects whose files are missing are skipped by the readers, so the block length is inferred, not assumed
	assert eeg_all.shape[0]==fmri_all.shape[0] and eeg_all.shape[0]%n_volumes==0, "unexpected number of volumes per subject"
	n_loaded=eeg_all.shape[0]//n_volumes
	if(verbose):
		print("I: Loaded", n_loaded, "subjects with", n_volumes, "volumes each")

	if(fmri_space=="mni" and dataset!="04"):
		fmri_all=mni_fmri(dataset, n_loaded, n_volumes)

	test_individuals, calib_individuals = list(test_individuals), list(calib_individuals)
	splits={"test": test_individuals, "calib": calib_individuals,
			"train": [i for i in range(n_loaded) if i not in test_individuals+calib_individuals]}
	out={}
	assert target in ("run_zscore", "fluct", "fluct_resid")
	if(target in ("fluct", "fluct_resid")):
		from evaluate_uq import otsu_mask
		anatomy=[]
		for i in range(n_loaded):
			block=fmri_all[i*n_volumes:(i+1)*n_volumes]
			mean=block.mean(0); anatomy.append(mean)
			fmri_all[i*n_volumes:(i+1)*n_volumes]=(block-mean)/(block.std(0)+1e-6)*otsu_mask(mean)
		out["mask"]=otsu_mask(np.mean([anatomy[i] for i in splits["train"]], 0))
		if(target=="fluct_resid"):
			locked=np.mean([fmri_all[i*n_volumes:(i+1)*n_volumes] for i in splits["train"]], 0)
			for i in range(n_loaded):
				fmri_all[i*n_volumes:(i+1)*n_volumes]-=locked
	if(inject_snr>0):
		assert target in ("fluct", "fluct_resid"), "the injected signal is defined on the unit-variance fluctuations"
		box=np.zeros(fmri_all.shape[1:4], bool); c=np.array(box.shape)//2
		box[c[0]-4:c[0]+4, c[1]-4:c[1]+4, int(0.7*box.shape[2])-2:int(0.7*box.shape[2])+2]=True
		out["inject_mask"]=box&out["mask"]
		ch=[getattr(eeg_utils, "channels_"+dataset).index(c3) for c3 in ("C3", "C4")]
		t=np.arange(0, 32, getattr(fmri_utils, "TR_"+dataset))
		from scipy.stats import gamma
		hrf=gamma.pdf(t, 6)-gamma.pdf(t, 16)/6.; hrf/=hrf.sum()
		for i in range(n_loaded):
			rows=slice(i*n_volumes, (i+1)*n_volumes)
			tr=getattr(fmri_utils, "TR_"+dataset)
			mu=-eeg_all[rows][:, ch, int(8*tr)-1:int(30*tr)].reshape(n_volumes, -1).mean(1)  #STFT bin k is (k+1)/TR Hz: 8-30 Hz
			sig=np.convolve(mu-mu.mean(), hrf)[:n_volumes]; sig=(sig-sig.mean())/(sig.std()+1e-6)
			assert fmri_all.ndim==4, fmri_all.shape
			block=fmri_all[rows]  #a view: the in-place update writes fmri_all
			sel=box&(block.std(0)>0)
			block[:, sel]+=np.sqrt(inject_snr)*sig[:, None]
	for name, individuals in splits.items():
		if(not len(individuals)):
			out[name]=None
			continue
		eeg=np.concatenate([eeg_all[i*n_volumes:(i+1)*n_volumes] for i in individuals])
		fmri=np.concatenate([fmri_all[i*n_volumes:(i+1)*n_volumes] for i in individuals])
		eeg, fmri = data_utils.create_eeg_bold_pairs(eeg, fmri, raw_eeg=False, fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),
													fs_sample_fmri=getattr(fmri_utils, "TR_"+dataset), interval_eeg=interval_eeg,
													n_volumes=n_volumes, n_individuals=len(individuals), instances_per_individual=25)
		subject_ids=np.repeat(individuals, n_volumes-interval_eeg)
		out[name]=(np.expand_dims(eeg, -1).astype('float32'), np.expand_dims(fmri, -1).astype('float32'), subject_ids)
	return out


def dataset(dataset, n_individuals=8, interval_eeg=6, time_length=1, ind_volume_fit=True, raw_eeg=False, standardize_fmri=True, standardize_eeg=True, iqr=True, file_output=None, verbose=False):
	"""
	time_length - refers to how big are the temporal segments considered. Default is 1, which indicates prediction of only one volume
	"""

	if(verbose):
		if(file_output == None):
			print("I: Starting to Load Data")	
		else:
			print("I: Starting to Load Data", file=file_output)

	#TR of fmri and window size of STFT
	f_resample=getattr(fmri_utils, "TR_"+dataset)

	if(dataset in ["02","03","04","05"]):
		eeg_limit=True
		eeg_f_limit_h=135
		eeg_f_limit_l=0
	else:
		eeg_limit=False
		eeg_f_limit_h=135
		eeg_f_limit_l=0

	eeg_train, fmri_train, scalers = data_utils.load_data(list(range(n_individuals)), raw_eeg=raw_eeg, n_voxels=None, 
															bold_shift=getattr(fmri_utils, "bold_shift_"+dataset), n_partitions=25, 
															mutate_bands=False,
															by_partitions=False, partition_length=14, 
															f_resample=f_resample, fmri_resolution_factor=1, 
															standardize_eeg=standardize_eeg, standardize_fmri=standardize_fmri,
															ind_volume_fit=ind_volume_fit, iqr_outlier=iqr,
															eeg_limit=eeg_limit, eeg_f_limit_h=eeg_f_limit_h,
															eeg_f_limit_l=eeg_f_limit_l, dataset=dataset)

	eeg_channels=eeg_train.shape[1]

	n_individuals_train = getattr(data_utils, "n_individuals_train_"+dataset)
	n_individuals_test = getattr(data_utils, "n_individuals_test_"+dataset)
	n_volumes = getattr(fmri_utils, "n_volumes_"+dataset)

	if(raw_eeg):
		eeg_test = eeg_train[n_individuals_train*(n_volumes)*int(f_resample*getattr(eeg_utils, "fs_"+dataset)):(n_individuals_train+n_individuals_test)*n_volumes*int(f_resample*getattr(eeg_utils, "fs_"+dataset))]
		eeg_train = eeg_train[:n_individuals_train*n_volumes*int(f_resample*getattr(eeg_utils, "fs_"+dataset))]
	else:
		eeg_test = eeg_train[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_test)*n_volumes]
		eeg_train = eeg_train[:n_individuals_train*n_volumes]

	fmri_test = fmri_train[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_test)*n_volumes]
	fmri_train = fmri_train[:n_individuals_train*n_volumes]

	if(verbose):
		if(file_output==None):
			print("I: Finished Loading Data")
		else:
			print("I: Finished Loading Data", file=file_output)

	if(time_length>1):
		eeg_train, fmri_train = data_utils.create_eeg_bold_temporal_pairs(eeg_train, fmri_train,time_length=time_length,raw_eeg=raw_eeg,fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),fs_sample_fmri=f_resample,interval_eeg=interval_eeg, n_volumes=n_volumes, n_individuals=n_individuals_train,instances_per_individual=25)
		eeg_test, fmri_test = data_utils.create_eeg_bold_temporal_pairs(eeg_test, fmri_test,time_length=time_length,raw_eeg=raw_eeg,fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),fs_sample_fmri=f_resample,interval_eeg=interval_eeg, n_volumes=n_volumes, n_individuals=n_individuals_test,instances_per_individual=25)
	else:
		eeg_train, fmri_train = data_utils.create_eeg_bold_pairs(eeg_train, fmri_train, raw_eeg=raw_eeg,fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),fs_sample_fmri=f_resample,interval_eeg=interval_eeg, n_volumes=n_volumes, n_individuals=n_individuals_train,instances_per_individual=25)
		eeg_test, fmri_test = data_utils.create_eeg_bold_pairs(eeg_test, fmri_test, raw_eeg=raw_eeg,fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),fs_sample_fmri=f_resample,interval_eeg=interval_eeg, n_volumes=n_volumes, n_individuals=n_individuals_test,instances_per_individual=25)

	eeg_train = np.expand_dims(eeg_train, axis=-1)
	eeg_test = np.expand_dims(eeg_test, axis=-1)
	if(time_length==1):
		fmri_train = np.expand_dims(fmri_train, axis=-1)
		fmri_test = np.expand_dims(fmri_test, axis=-1)

	eeg_train = eeg_train.astype('float32')
	fmri_train = fmri_train.astype('float32')
	eeg_test = eeg_test.astype('float32')
	fmri_test = fmri_test.astype('float32')

	if(verbose):
		if(file_output == None):
			print("I: Pairs Created")
		else:
			print("I: Pairs Created", file=file_output)

	return (eeg_train, fmri_train), (eeg_test, fmri_test)


def dataset_clf(dataset, n_individuals=8, mutate_bands=False, f_resample=2, raw_eeg=False, raw_eeg_resample=False, eeg_limit=False, eeg_f_limit_h=135, eeg_f_limit_l=0, file_output=None, standardize_eeg=False, interval_eeg=10, verbose=False):

	n_individuals_train=getattr(data_utils, "n_individuals_train_"+dataset)
	n_individuals_test=getattr(data_utils, "n_individuals_test_"+dataset)
	recording_time=getattr(eeg_utils, "recording_time_"+dataset)
	eeg_limit=getattr(eeg_utils, "fs_"+dataset)>250

	if(verbose):
		if(file_output == None):
			print("I: Loading data")
		else:
			print("I: Loading data", file=file_output)

	X, y = data_utils.load_data_clf(dataset, n_individuals=n_individuals, 
									mutate_bands=mutate_bands, f_resample=f_resample, 
									raw_eeg=raw_eeg, raw_eeg_resample=raw_eeg_resample, 
									eeg_limit=eeg_limit, eeg_f_limit_h=eeg_f_limit_h, 
									eeg_f_limit_l=eeg_f_limit_l, recording_time=recording_time,
									standardize_eeg=standardize_eeg)

	if(verbose):
		if(file_output == None):
			print("I: Creating pairs")
		else:
			print("I: Creating pairs", file=file_output)

	X_train, y_train = data_utils.create_clf_pairs(n_individuals_train, X[:n_individuals_train*recording_time], 
											y[:n_individuals_train], 
											raw_eeg=raw_eeg,
											recording_time=recording_time, 
											interval_eeg=interval_eeg)
	X_test, y_test = data_utils.create_clf_pairs(n_individuals_test, X[:n_individuals_test*recording_time], 
												y[:n_individuals_test], 
												raw_eeg=raw_eeg,
												recording_time=recording_time,
												interval_eeg=interval_eeg)

	if(verbose):
		if(file_output == None):
			print("I: Finished loading data")
		else:
			print("I: Finished loading data", file=file_output)
	
	X_train = np.expand_dims(X_train, axis=-1)
	X_test = np.expand_dims(X_test, axis=-1)

	X_train = X_train.astype('float32')
	y_train = y_train.astype('float32')
	X_test = X_test.astype('float32')
	y_test = y_test.astype('float32')

	return (X_train, y_train), (X_test, y_test)




class Dataset_CLF_CV:
	"""
	This class is a wrapper for a dataset that returns the folds in a cross validation setting
	"""

	
	def __init__(self, dataset, mutate_bands=False, f_resample=2, raw_eeg=False, raw_eeg_resample=False, eeg_limit=False, eeg_f_limit_h=134, eeg_f_limit_l=0, standardize_eeg=False, load=True, load_path=None, seed=None, verbose=False):
		"""
		Inputs:
			* str - dataset identifier
			* bool - mutate_bands mutate frequency domain to correspond to bands (DEPRECATED)
			* bool - f_resample resample EEG signal, used for STFT window size
			* bool - raw_eeg whether to return EEG as channel time or channel frequency time
			* bool - raw_eeg_resample if (raw EEG) should be resampled to f_resample
			* bool - eeg_limit whether to give a frequency high pass filter
			* int - eeg_f_limit frequency high pass filter value
			* bool - standardize_eeg whether or not to scale the dataset to be a Normal(0,1)
			* bool - verbose whether to print state of execution
		"""
		assert dataset in ["10", "11", "15"], "Dataset not recognized, may not yet be implemented"

		self.n_individuals=getattr(data_utils, "n_individuals_"+dataset)
		self.recording_time=getattr(eeg_utils, "recording_time_"+dataset)
		self.n_classes=getattr(eeg_utils, "n_classes_"+dataset)
		
		self.eeg_limit=eeg_limit
		self.eeg_f_limit_h=eeg_f_limit_h
		self.eeg_f_limit_l=eeg_f_limit_l
		self.interval_eeg=10
		self.seed=seed

		if(load):
			X, y = data_utils.load_data_clf(dataset, n_individuals=self.n_individuals, 
											mutate_bands=mutate_bands, f_resample=f_resample, 
											raw_eeg=raw_eeg, raw_eeg_resample=raw_eeg_resample, 
											eeg_limit=eeg_limit, eeg_f_limit_h=eeg_f_limit_h, 
											eeg_f_limit_l=eeg_f_limit_l, recording_time=self.recording_time,
											standardize_eeg=standardize_eeg)

			print(X.shape, y.shape)
			
			self.X, self.y = data_utils.create_clf_pairs(self.n_individuals, 
														X, y, raw_eeg=raw_eeg,
														n_classes=self.n_classes,
														recording_time=self.recording_time, 
														interval_eeg=self.interval_eeg)
			
			self.X = np.expand_dims(self.X, axis=-1)

			print(self.X.shape, self.y.shape)
			del X,y
			gc.collect()
		else:
			self.load(load_path)

		self.folds=self.n_individuals
		self.ind_time=(self.X.shape[0])//(self.n_individuals)
	
	
	def set_folds(self, n):
		"""
		Inputs:
			* int - number of folds, default is leave one out
		"""
		
		self.folds=n
		
	
	def split(self, fold):
		"""
		Inputs:
			* int - fold number
		Outputs:
			* tuple(tuple, tuple) - train test split
		"""
		
		assert fold < self.n_individuals, "A fold is within {0, 1, ..., N-1} for a total of N folds"
		
		step=(self.n_individuals//self.folds)
		return (np.append(self.X[:fold*self.ind_time*step], self.X[fold*self.ind_time*step+self.ind_time*step:], axis=0), 
						np.append(self.y[:fold*self.ind_time*step], self.y[fold*self.ind_time*step+self.ind_time*step:], axis=0)), \
				(self.X[fold*self.ind_time*step:fold*self.ind_time*step+self.ind_time*step], 
						 self.y[fold*self.ind_time*step:fold*self.ind_time*step+self.ind_time*step])

	
	def shuffle(self):
		"""
		Shuffle dataset

		This should only be done in a CV setting with fold < individuals, i.e. please do not do this in a LOOCV setting
		"""
		data = shuffle(self.X, self.y, random_state=self.seed)
		self.X=data[0]
		self.y=data[1]


	
	def save(self, path):
		"""
		Save data to numpy array to ease loading bottleneck
		"""
		np.save(path+"X.npy", self.X, allow_pickle=True)
		np.save(path+"y.npy", self.y, allow_pickle=True)

	
	def load(self, path):
		"""
		Load data from .npy file
		"""
		self.X = np.load(path+"X.npy", allow_pickle=True)
		self.y = np.load(path+"y.npy", allow_pickle=True)


class DatasetContrastive:

	def __init__(self, X, labels, batch=4, pairs=4, repeat_pairing=False, clf=False, seed=None):
		"""

			arguemnt clf specifies if one also gives the labels of the data in the tensorflow.data.Dataset

		"""
		self.seed=seed

		self.X = X
		self.labels = labels

		#number of pairs has to be lower than the size of the dataset
		self.pairs=X.shape[0]//2

		self.data=None
		self.y=None
		self.y1=None
		self.y2=None

		self.batch=batch
		self.clf=clf
		self.repeat_pairing=repeat_pairing
		self.tf_dataset=None
	
	def shuffle(self):
		"""
		Shuffle dataset

		This should only be done in a CV setting with fold < individuals, i.e. please do not do this in a LOOCV setting
		"""
		data = shuffle(self.X, self.labels, random_state=self.seed)
		self.X=data[0]
		self.labels=data[1]

	@property
	def pairwise(self):
		del self.data, self.y, self.y1, self.y2
		gc.collect()
		
		self.data=np.empty((self.pairs*2,2,)+self.X.shape[1:], dtype=np.float32)
		
		self.y=np.empty((self.pairs*2,2), dtype=np.float32)

		if(self.clf):
			self.y1=np.empty((self.pairs*2,2), dtype=np.float32)
			self.y2=np.empty((self.pairs*2,2), dtype=np.float32)

		#2 class shape will be deprecated in the future

		instance=0

		positive=self.pairs
		negative=self.pairs
		
		while(positive>0 or negative>0):
			
			#choice is done randomly throughout the dataset
			indices = np.random.choice(np.arange(0,self.X.shape[0]), size=2, replace=False)
			i1, i2 = (indices[0], indices[1])

			if(np.all(self.labels[i1]==self.labels[i2])):
				if(positive==0):
					continue
				self.y[instance]=np.array([0.0, 1.0], dtype=np.float32)
				positive-=1
			else:
				if(negative==0):
					continue
				self.y[instance]=np.array([1.0, 0.0], dtype=np.float32)
				negative-=1

			if(self.clf):
				self.y1[instance]=self.labels[i1].astype(np.float32)
				self.y2[instance]=self.labels[i2].astype(np.float32)

			self.data[instance]=np.concatenate((self.X[i1:i1+1], self.X[i2:i2+1]), axis=0)
			instance+=1

		if(self.clf):
			return self.data, np.concatenate((np.expand_dims(self.y, axis=1), np.concatenate((np.expand_dims(self.y1,axis=1), np.expand_dims(self.y2,axis=1)), axis=1)), axis=1)

		return self.data, self.y

	def repeat(self, n, shuffle=True):
		"""
		repeat n times the dataset, this is a wrapper for the train session
		"""

		#if(not self.repeat_pairing and self.tf_dataset is not None):
		#	return self.tf_dataset

		del self.tf_dataset
		gc.collect()

		if(shuffle):
			self.tf_dataset=tf.data.Dataset.from_tensor_slices(self.pairwise).shuffle(self.X.shape[0]*self.pairs*2, seed=self.seed).batch(self.batch).repeat(n)
		else:
			self.tf_dataset=tf.data.Dataset.from_tensor_slices(self.pairwise).batch(self.batch).repeat(n)

		return self.tf_dataset


def mni_fmri(dataset, n_loaded, n_volumes):
	"""(n_loaded*n_volumes, 64, 64, 30) registered MIpost volumes (see dataset_subjects), each run z-scored with one
	scalar mean/std like the native loader."""
	assert dataset=="05", "registered volumes exist for ds002338 only"
	root=os.path.join(os.environ.get("EEG_FMRI_CACHE", os.path.expanduser("~/eeg_to_fmri/uncertainty/cache")), "xp2_mni")
	subs=sorted(d for d in os.listdir(os.path.join(os.environ["EEG_FMRI_DATASETS"], "ds002338")) if d.startswith("sub-xp2"))
	assert len(subs)==n_loaded, (len(subs), n_loaded)
	shift=getattr(fmri_utils, "bold_shift_"+dataset)
	out=np.empty((n_loaded*n_volumes, 64, 64, 30), np.float32)
	for i, sub in enumerate(subs):
		v=np.load(os.path.join(root, sub+"_MIpost_raw.npy"), mmap_mode="r")[shift:shift+n_volumes].astype(np.float32)
		assert v.shape[1:]==(60, 72, 60), v.shape
		v=np.pad(v[:, :, 4:68], ((0, 0), (2, 2), (0, 0), (0, 0)))
		v=0.5*(v[..., 0::2]+v[..., 1::2])
		out[i*n_volumes:(i+1)*n_volumes]=(v-v.mean())/(v.std()+1e-6)
	return out


XP1_EXCLUDE=("sub-xp102",)  #no motor-localizer EEG export (only its other runs)
XP1_N_SUBJECTS=9
XP1_N_VOLUMES=157  #motor localizer: 163 pulses; first 3 dropped (steady state), last windows can run past the EEG export's end


def xp1_motorloc(channels, eeg_norm, eeg_fmax, cache_dir):
	"""CHUR-XP1 (ds002336) motor localizer for the replication: (eeg (n*160, C, F), fmri (n*160, 64, 64, 30)).
	EEG: the preprocessed export (eeg_pp), magnitude STFT of each TR window [R128_k, R128_k + TR) aligned to the k-th
	scanner pulse (the export starts ~11 s before the first pulse), 1/TR = 0.5 Hz bins from 0.5 Hz up to eeg_fmax,
	normalized like dataset_subjects. fMRI: motion-corrected volumes registered to MNI (register_xp2_bold.py -dataset
	ds002336, cache/xp1_mni), cropped/averaged to 64x64x30 as mni_fmri, run z-scored."""
	import mne, re
	assert channels=="eeg" and eeg_norm=="time", "the replication uses the main pipeline (EEG channels, time normalization)"
	root=os.environ.get("EEG_FMRI_CACHE", os.path.expanduser("~/eeg_to_fmri/uncertainty/cache"))
	cache=None if cache_dir is None else os.path.join(cache_dir, "04_xp1_motorloc_n%d_v%d_f%d.npz" % (XP1_N_SUBJECTS, XP1_N_VOLUMES, int(eeg_fmax)))
	if(cache is not None and os.path.exists(cache)):
		with np.load(cache) as f:
			return f["eeg"], f["fmri"]
	data=os.path.join(os.environ["EEG_FMRI_DATASETS"], "ds002336")
	subs=sorted(d for d in os.listdir(data) if d.startswith("sub-xp1") and d not in XP1_EXCLUDE)
	assert len(subs)==XP1_N_SUBJECTS, subs
	tr, fs, skip=getattr(fmri_utils, "TR_04"), getattr(eeg_utils, "fs_04"), 3
	w=int(tr*fs); kmax=int(eeg_fmax*tr)
	E, F=[], []
	for sub in subs:
		base=os.path.join(data, "derivatives", sub, "eeg_pp", sub+"_task-motorloc_eeg_pp")
		raw=mne.io.read_raw_brainvision(base+".vhdr", preload=True, verbose=0)
		x=raw.get_data(picks=eeg_utils.pick_channels(raw, ("eeg",)))
		pulses=[int(m.group(1))-1 for m in re.finditer(r"=Response,R128,(\d+),", open(base+".vmrk").read())]
		pulses=[p for p in pulses if p+w<=x.shape[1]]  #complete TR windows only
		assert len(pulses)>=skip+XP1_N_VOLUMES, (sub, "complete TR windows", len(pulses))
		win=np.stack([x[:, p:p+w] for p in pulses[skip:skip+XP1_N_VOLUMES]], 1)  #(C, T, w)
		S=np.abs(np.fft.rfft(win-win.mean(-1, keepdims=True), axis=-1))[..., 1:kmax+1]  #(C, T, kmax), 0.5..fmax Hz
		S=np.transpose(S, (1, 0, 2))
		E.append((S-S.mean(0))/(S.std(0)+1e-6))
		v=np.load(os.path.join(root, "xp1_mni", sub+"_motorloc_raw.npy"), mmap_mode="r")[skip:skip+XP1_N_VOLUMES].astype(np.float32)
		v=np.pad(v[:, :, 4:68], ((0, 0), (2, 2), (0, 0), (0, 0))); v=0.5*(v[..., 0::2]+v[..., 1::2])
		F.append((v-v.mean())/(v.std()+1e-6))
	eeg_all, fmri_all=np.concatenate(E).astype(np.float32), np.concatenate(F).astype(np.float32)
	if(cache is not None):
		os.makedirs(cache_dir, exist_ok=True); np.savez(cache, eeg=eeg_all, fmri=fmri_all)
	return eeg_all, fmri_all

