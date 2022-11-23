import sys

sys.path.append("../../src")

import numpy as np

from layers import fft


def downsample(views, resolution, threshold=0.37, cutoff=False, cutoff_low=False,):

	dct=fft.DCT3D(*views.shape[1:-1])
	idct=fft.iDCT3D(*resolution)
	
	brain_mask=None


	if(cutoff):
		min_value=np.amin(views)
		mean_X_view=np.mean(views, axis=0)
		normalized_X_views=(mean_X_view[:,:,:,:]-np.amin(mean_X_view[:,:,:,:]))/(np.amax(mean_X_view[:,:,:,:])-np.amin(mean_X_view[:,:,:,:]))
		normalizing_indices=np.where(normalized_X_views<=threshold)
		for i in range(views.shape[0]):
			views[i][normalizing_indices]=min_value

	X_view_downsampled=np.empty((0,)+resolution+(1,), dtype="float32")

	for i in range(X_view.shape[0]):
		X_view_downsampled=np.append(X_view_downsampled,np.expand_dims(idct(dct(views[i:i+1,:,:,:,0])[:,:resolution[0],:resolution[1],:resolution[2]]).numpy(), axis=-1),axis=0)

	if(cutoff_low):	
		min_value=np.amin(X_view_downsampled)
		mean_X_view=np.mean(X_view_downsampled, axis=0)
		normalized_X_views=(mean_X_view[:,:,:,:]-np.amin(mean_X_view[:,:,:,:]))/(np.amax(mean_X_view[:,:,:,:])-np.amin(mean_X_view[:,:,:,:]))
		normalizing_indices=np.where(normalized_X_views<threshold)
		brain_mask=np.where(normalized_X_views>=threshold)
		for i in range(X_view_downsampled.shape[0]):
			X_view_downsampled[i][normalizing_indices]=min_value
			
	return X_view_downsampled, brain_mask


def build_arff(views, y_true, y_pred, resolution, cutoff_low=False, brain_mask=None):

	string_file_view_ground_truth=""
	string_file_view_pred=""

	if(cutoff_low):
		attributes=brain_mask[0].shape[0]
	else:
		attributes=resolution[0]*resolution[1]*resolution[2]

	if(format_file=="arff"):
		string_file_view_ground_truth+="@RELATION \"eeg_fmri\"\n\n"
		string_file_view_ground_truth+="@ATTRIBUTE eeg_fmri {Ignore, 0, 1}\n"

		for value in range(attributes):
			string_file_view_ground_truth+="@ATTRIBUTE "+str(value)+" NUMERIC\n"		

		string_file_view_ground_truth+="\n@DATA\n"

	if(format_file=="csv"):
		string_file_view_ground_truth+=","
		string_file_view_pred+=","
		for value in range(attributes):
			string_file_view_ground_truth+=str(value)+","
			string_file_view_pred+=str(value)+","
		string_file_view_ground_truth+="target\n"
		string_file_view_pred+="target\n"

	for individual in range(views.shape[0]):
		if(format_file=="arff"):
			string_file_view_ground_truth+=str(int(y_true[individual]))+","
			string_file_view_pred+=str(int(y_pred[individual]))+","
		elif(format_file=="csv"):
			string_file_view_ground_truth+=str(individual)+","
			string_file_view_pred+=str(individual)+","

		if(cutoff_low):
			individual_voxels=views[individual][brain_mask]
		else:
			individual_voxels=np.reshape(views[individual], (resolution[0]*resolution[1]*resolution[2],))

		for value in individual_voxels:
			string_file_view_ground_truth+="{:0.9f}".format(float(value))+","
			string_file_view_pred+="{:0.9f}".format(float(value))+","

		if(format_file=="csv"):
			string_file_view_ground_truth+=str(int(y_true[individual]))+","
			string_file_view_pred+=str(int(y_pred[individual]))+","

		string_file_view_ground_truth=string_file_view_ground_truth[:-1]+"\n"
		string_file_view_pred=string_file_view_pred[:-1]+"\n"
		
	return string_file_view_ground_truth, string_file_view_pred