import numpy as np


def make_indices2d(indices,):
	"""
	This function joins the indices to access a 2d np.ndarray
	
	Input:
		* tuple: of length 2, each entry is an np.ndarray
	Output:
		* tuple: of length 2, each entry is an np.ndarray
	"""
	x=indices[0]
	y=indices[1]
	
	x_i=np.empty((x.shape[0]*y.shape[0],), dtype=np.int32)
	y_i=np.empty((x.shape[0]*y.shape[0],), dtype=np.int32)
	
	instance=0
	for i in x:
		for j in y:
			x_i[instance]=i
			y_i[instance]=j
			instance+=1
	
	return (x_i, y_i)


def make_indices4d(indices,):
	"""
	This function joins the indices to access a 4d np.ndarray
	
	Input:
		* tuple: of length 4, each entry is an np.ndarray
	Output:
		* tuple: of length 4, each entry is an np.ndarray
	"""
	w=indices[0]
	x=indices[1][0]
	y=indices[1][1]
	z=indices[1][2]
	
	w_i=np.empty((w.shape[0]*x.shape[0],), dtype=np.int32)
	x_i=np.empty((w.shape[0]*x.shape[0],), dtype=np.int32)
	y_i=np.empty((w.shape[0]*x.shape[0],), dtype=np.int32)
	z_i=np.empty((w.shape[0]*x.shape[0],), dtype=np.int32)
	
	instance=0
	for i in w:
		for index in range(x.shape[0]):
			w_i[instance]=i
			x_i[instance]=x[index]
			y_i[instance]=y[index]
			z_i[instance]=z[index]
			instance+=1
					
	return (w_i, x_i, y_i, z_i)
	
	
def filter_voxels(mask, voxels):
	new_voxels=[np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)]
	
	for voxel in range(mask[0].shape[0]):
		if(voxel in voxels):
			new_voxels[0]=np.append(new_voxels[0],mask[0][voxel])
			new_voxels[1]=np.append(new_voxels[1],mask[1][voxel])
			new_voxels[2]=np.append(new_voxels[2],mask[2][voxel])
	
	return tuple(new_voxels)

class Pattern:
	
	"""
	This class implements what is the concept of a pattern
	
	
	"""
	
	def __init__(self, values, size, voxels, instances, pvalue, lifts):

		"""
		Intializes class attributes of a pattern
		"""
		
		self.values=values
		self.size=size
		self.voxels=voxels
		self.instances=instances
		self.pvalue=pvalue
		self.lifts=np.nan_to_num(lifts)
		
	def mask(self, X, y=None, y_ref=None, low=False, resolution=None, brain_mask=None):
		"""
		Applies this pattern by building a mask where the pattern occurs

		"""

		if(low):
			assert not resolution is None
		else:
			resolution=X.shape[1:-1]
			
		print("I: Pattern "+self.__hash__()[:10]+" with "+str(len(self.voxels))+" voxels and "+str(len(self.instances))+" instances.")
			
		if(y is not None):
			assert y_ref is not None
			instances=np.intersect1d(np.where(y_ref==y)[0],self.instances)
		else:
			instances=self.instances
		
		z = np.zeros((X.shape[0],)+resolution, dtype=np.float32)
		if(brain_mask is None):
			z = np.reshape(z, (X.shape[0], resolution[0]*resolution[1]*resolution[2]))
		else:
			voxels=filter_voxels(brain_mask, self.voxels)
			
		z[instances,:]=1.
		if(brain_mask is not None):
			z[make_indices4d((instances, voxels))]*=-1.
		else:
			z[make_indices2d((instances, self.voxels))]*=-1.
		z[np.where(z==1.)]=0.0
		z[np.where(z==-1.)]=1.0
		
		if(brain_mask is None):
			z = np.reshape(z, (X.shape[0],)+resolution)
		
		if(low):
			z_high = np.zeros(X.shape, dtype=np.float32)
			for instance in range(z.shape[0]):
				z_i=np.reshape(z[instance:instance+1], (1,)+resolution)
				z_high[instance] = np.expand_dims(fft.padded_iDCT3D(*resolution, *X.shape[1:-1])(fft.DCT3D(*resolution)(z_i)).numpy(), axis=-1)
			z=z_high
			
		return (z-np.amin(z))/(np.amax(z)-np.amin(z))
	
	def __eq__(self, other):
		"""
		To sort patterns and finds best
		"""

		return self.__hash__()==other.__hash__()
	
	def __hash__(self):
		"""
		Hash is required for __eq__
		"""
		return str(self.values)+"_"+str(self.size)+"_"+str(self.voxels)+"_"+str(self.instances)+"_"+str(self.pvalue)+"_"+str(self.lifts)		

class Bics_Patterns:
	"""
	Class that contains the parameters found


	This acts as a pattern manager

	"""
	
	def __init__(self, X, relu=None, low=True, resolution=(14,14,7)):
		"""
		Applies this pattern by building a mask where the pattern occurs

		"""

		self.patterns=[]
		self.delete_patterns=[]
		self.X=X
		self.relu=relu
		self.low=low
		self.resolution=resolution
		
	def add(self, pattern):
		"""
		Adds a pattern to the list of patterns

		This function is called upon building the class

		Inputs:
			* Pattern
		"""
		if(type(pattern) is dict):
			raise NotImplementedError  
		elif(type(pattern) is Pattern):
			self.patterns+=[pattern]
			self.delete_patterns+=[pattern]
		else:
			print("E: Pattern structure not recognized.")
		
	def get(self, index):
		"""
		Given an index returns the pattern

		Outputs:
			* Pattern
		"""
		assert index < len(self.patterns)
		
		return self.patterns[index]
	
	def get_delete(self, index):
		"""
		Given an index returns the pattern from the delete_patterns list

		Outputs:
			* Pattern
		"""

		assert index < len(self.delete_patterns)
		
		return self.delete_patterns[index]
		
	def get_best(self, delete=False):
		"""
		Returns the pattern with the best lifts, that has not been returned until now if delete is set to True

		Inputs:
			* bool: default to False, if True returns the best pattern not returned so far
		Outputs:
			* Pattern
		"""

		search_patterns=self.delete_patterns
			
		best_value=-1
		best_index=-1
		for pattern in range(len(search_patterns)):
			lifts=np.mean(self.get_delete(pattern).lifts)
			if(lifts>best_value):
				best_value=lifts
				best_index=pattern
		
		index=self.patterns.index(search_patterns[best_index])
		
		if(delete):
			del self.delete_patterns[best_index]
		
		return index
	
	def get_best_pattern(self):
		"""
		Returns the pattern with the best lifts

		Outputs:
			* Pattern
		"""

		return self.patterns[self.get_best()]
	
	def apply_mask(self, index, y=None, y_ref=None, brain_mask=None):
		"""
		Builds the pattern mask, given by the index

		Inputs:
			* int: pattern;
			* np.float32: specifies a label
			* np.ndarray: contains the labels of the dataset, this is required if y is not None
			* np.ndarray: a mask where the brain is present
		Outputs:
			* np.ndarray
		"""
		
		pattern=self.get(index)
		return pattern.mask(self.X, y=y, y_ref=y_ref, low=self.low, resolution=self.resolution, brain_mask=brain_mask)

	@classmethod
	def build(self, f, X, **kwargs):
		"""
		Builds the manager given a file path and a set of views

		Inputs:
			* str: path to file containing the output of BicPAMS;
			* np.ndarray: contains the views where BicPAMS was ran on;
		Outputs:
			* Bics_Patterns
		"""
		patterns=Bics_Patterns(X, **kwargs)
		
		with open(f, "r") as f:
			lines = f.readlines()
		for line in lines:
			if(not "I=" in line):
				continue
			if("too small" in line):
				line=line.replace(' (too small)', '')
			#	continue
			if("long" in line):
				continue
				
			line_list=line[1:-1].split(" ")
			values=np.array(line_list[0][3:-1].split(","), dtype="float32")#I
			size=(int(line_list[1][1:-1].split(",")[0]), int(line_list[1][1:-1].split(",")[1]))
			voxels=np.array(line_list[2][3:-1].split(","), dtype="int32")
			instances=np.array(line_list[3][3:-1].split(","), dtype="int32")
			pvalue=float(line_list[4].split("=")[1])
			class_lifts=np.array([float(line_list[5].split("=")[1][1:-1]),float(line_list[6][:-1])], dtype="float32")
			patterns.add(Pattern(values,size,voxels,instances,pvalue,class_lifts))
			
		return patterns