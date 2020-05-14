# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:26:18 2020

@author: Lotte
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.xarray import data_to_xarray
import seaborn as sns
from scipy.stats import spearmanr
import xarray as xr
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

##importing data
data = np.load('tensor_daily_mean_5D.npy')
data_scaled = data*0
for i in range(10):
    mean = np.nanmean(data[:,:,i,:,:])
    data_scaled[:,:,i,:,:]=data[:,:,i,:,:]-mean
    st_dev = np.nanstd(data[:,:,i,:,:])
    data_scaled[:,:,i,:,:] = data_scaled[:,:,i,:,:]/st_dev
    
X = data_to_xarray(data)
Y = data_to_xarray(data_scaled) #Y is the normalized data xarray
X.head()
Y.head()

##Dates
dates= pd.date_range(start="2006-01-01",end="2096-12-31")

##Variable names
var= 1
variables = ["rsds","tas","uas","vas","clt","hurs","ps"]
VARIABLES = ['Surface Downwelling Shortwave Radiation',
             'Near-surface air temperature',
             'Eastward near-surface wind',
             'Northward near-surface wind',
             'Cloud cover',
             'Near-surface relative humidity',
             'Surface pressure']
"""
0 rsds - Surface Downwelling Shortwave Radiation
1 tas - near surface air temperature
2 uas - Eastward Near-Surface Wind
3 vas - Northward Near-Surface Wind
4 clt - Total Cloud Cover
5 hurs - Near-Surface Relative Humidity
6 ps - Surface Pressure
"""

##Station names
st = 0
STATIONS = ['Marsdiep Noord','Doove Balg West',
                'Vliestroom','Doove Balg Oost',
                'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                'Harlingen Havenmond West']
"""
0 'Marsdiep Noord'
1 'Doove Balg West',
2 'Vliestroom'
3'Doove Balg Oost'
4 'Blauwe Slenk Oost',
5 'Harlingen Voorhaven',
6 'Dantziggat',
7 'Zoutkamperlaag Zeegat'
8 'Zoutkamperlaag',
9 'Harlingen Havenmond West'
"""

##Model names
mdl = 0
MODELS = ['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH',
          'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']
"""
0 'CNRM-CERFACS-CNRM-CM5'
1 'ICHEC-EC-EARTH'
2 'IPSL-IPSL-CM5A-MR'
3 'MOHC-HadGEM2-ES'
4 'MPI-M-MPI-ESM-LR'
"""

##Experiment names
exp = 0
EXPERIMENTS = ['rcp45','rcp85']

#predict relation between temperature and the other variables

D=X.sel(var=["rsds","uas","vas","clt","hurs","ps"],station='Marsdiep Noord',model='CNRM-CERFACS-CNRM-CM5',exp='rcp45')
Q=np.array(D)
y=X.sel(var="tas",station='Marsdiep Noord',model='CNRM-CERFACS-CNRM-CM5',exp='rcp45')
D = sm.add_constant(D) #adds possible constant to the linear relation
model = sm.OLS(np.array(y), np.array(D)).fit()
beta = model.params
#predictions = model.predict(X)

print(model.summary())


def f(v,beta):
    t = beta[0]+sum(v[:]*beta[1:len(beta)])
    return t

temp_predicted = np.zeros(len(y))
for i in range(len(y)):
    temp_predicted[i] = f(Q[i,:],beta)

fig = plt.figure()    
tdata = plt.plot(y,label='T from data')
tregr = plt.plot(temp_predicted,label='Predicted T from multi linear regression')
plt.xlabel('time (days)')
plt.ylabel('temperature (Celcius)')
plt.legend((tdata,tregr), ('T from data','Predicted T from multi linear regression'))
plt.show()