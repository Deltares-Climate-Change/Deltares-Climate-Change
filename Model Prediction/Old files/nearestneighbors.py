# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:43:31 2020

@author: Rens
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.common import data_to_xarray
import seaborn as sns
import xarray as xr
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


data = np.load('/Users/Rens/Desktop/Data_CC/tensor_daily_mean_5D.npy')
data_scaled = data*0
for i in range(10):
    mean = np.nanmean(data[:,:,i,:,:])
    data_scaled[:,:,i,:,:]=data[:,:,i,:,:]-mean
    st_dev = np.nanstd(data[:,:,i,:,:])
    data_scaled[:,:,i,:,:] = data_scaled[:,:,i,:,:]/st_dev
    
X = data_to_xarray(data)
Y = data_to_xarray(data_scaled) #Y is the normalized data xarray
X.head() #regular data
Y.head() #normalized data


# var1 = X.sel(var='tas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
# var2 = X.sel(var='tas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'ICHEC-EC-EARTH')
# var3 = X.sel(var='tas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'IPSL-IPSL-CM5A-MR')
# var4 = X.sel(var='tas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'MOHC-HadGEM2-ES')
# var5 = X.sel(var='tas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'MPI-M-MPI-ESM-LR')
# newvars=xr.concat([var1, var2, var3, var4], 'var').transpose()

mod1 = Y.sel(station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
mod2 = Y.sel(station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'ICHEC-EC-EARTH')

k = 5
training = 1000

nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(mod1[:training])
distances, indices = nbrs.kneighbors(mod1[training:])

pred = np.zeros([33238,7])
pred[:training] = mod2[:training]

j=training
for index in indices:
    appr = np.zeros([1,7])
    for i in range(k):
        appr = appr+np.array(mod2[index[i]])
    appr = appr/k
    pred[j] = appr
    j=j+1

# plotting
xlst = np.arange(33238)















