import numpy as np


class IQR:
    def __init__(self):
        
        pass
    
    def fit(self, X):
        Q1 = np.percentile(X, 25, interpolation = 'midpoint')
        Q3 = np.percentile(X, 75, interpolation = 'midpoint')
        
        IQR = Q3 - Q1  
        
        self.low = Q1 - 1.5 * IQR
        self.up = Q3 + 1.5 * IQR
    
    def transform(self, X, channels_last=True):
        _X = np.copy(X)
        
        iqr_X=np.zeros(X.shape)

        self.outlier_values=[]
        self.outlier_idx=[]

        if(channels_last):
            for x in range(_X.shape[0]):
                for y in range(_X.shape[1]):
                    for z in range(_X.shape[2]):
                        if (np.any(_X[x,y,z,:] > self.up) or np.any(_X[x,y,z,:] < self.low)):
                            self.outlier_idx.append((x,y,z))
                            self.outlier_values.append(_X[x,y,z,:])
                            _X[x,y,z,:]=np.zeros(_X[x,y,z,:].shape)
        else:
            for x in range(_X.shape[1]):
                for y in range(_X.shape[2]):
                    for z in range(_X.shape[3]):
                        if (np.any(_X[:,x,y,z] > self.up) or np.any(_X[:,x,y,z] < self.low)):
                            self.outlier_idx.append((x,y,z))
                            self.outlier_values.append(_X[:,x,y,z])
                            _X[:,x,y,z]=np.zeros(_X[:,x,y,z].shape)
        
        return _X
    
    def inverse_transform(self, X, channels_last=True):
        _X = np.copy(X)
        
        for idx in range(len(self.outlier_idx)):
            if(channels_last):
                _X[self.outlier_idx[idx][0],self.outlier_idx[idx][1],self.outlier_idx[idx][2],:] = self.outlier_values[idx]
            else:
                _X[:,self.outlier_idx[idx][0],self.outlier_idx[idx][1],self.outlier_idx[idx][2]] = self.outlier_values[idx]
        
        return _X