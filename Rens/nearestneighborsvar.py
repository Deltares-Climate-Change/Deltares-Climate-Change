# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:48:37 2020

@author: Rens
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.common import data_to_xarray
from utils.common import yearly_average
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

var1 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
var2 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'ICHEC-EC-EARTH')
var3 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'IPSL-IPSL-CM5A-MR')
var4 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'MOHC-HadGEM2-ES')
var5 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'MPI-M-MPI-ESM-LR')
newvars=xr.concat([var1, var2, var3, var4], 'var').transpose()

k = 5
training = 3000

nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(newvars[:training])
distances, indices = nbrs.kneighbors(newvars[training:])

pred=np.zeros(33238)
pred[:training]=var5[:training]

p=training
for i in range(len(indices)):
    index = indices[i]
    dist = distances[i]
    appr = 0.
    for j in range(k):
        fact = dist[j]/sum(dist)
        appr = appr+np.array(var5[index[j]])*fact
    pred[p] = appr
    p=p+1

# plotting
xlst = np.arange(6000)
plt.plot(xlst,pred[:6000],label='Prediction')
plt.plot(xlst,var5[:6000],label='Ground truth')
plt.legend()
plt.show()

dates= pd.date_range(start="2006-01-01",end="2096-12-31")
pred = xr.DataArray(pred, coords=[dates], dims=['time'])

plt.plot(yearly_average(pred),label='prediction')
plt.plot(yearly_average(var5),label='Ground truth')
plt.legend()
plt.plot()


#pd.DataFrame({'y':pred}).ewm(alpha=0.0001,min_periods=2000).mean().plot()























