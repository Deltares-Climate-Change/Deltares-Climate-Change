#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:17:13 2020

@author: ceciliacasolo
"""

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from datetime import timedelta
import matplotlib.pyplot as plt

#We should unify the location and the name of our data, that would be the easiest way
# to prevent pushing our own storage locations around. I propose one folder called Datares in the main directory
data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(data))
for i in range(len(NanINDX)):
    data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200
c=xr.DataArray(data)
darr = da.from_array(data, chunks=(33238,7,10,5,2))


dates= pd.date_range(start="2006-01-01",end="2096-12-31")
experiments=['rcp45','rcp85']
variables=["rsds","tas","uas","vas","clt","hurs","ps"]
stations=['Marsdiep Noord','Doove Balg West',
                'Vliestroom','Doove Balg Oost',
                'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                'Harlingen Havenmond West']
driving_models=['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 
          'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']
temp = xr.DataArray(data, coords=[dates,variables,stations,driving_models,experiments], dims=['time', 'var', 'station','model','exp'])
stations_numbers=[1,2,3,4,5,6,7,8,9,10]
temp_station_numbers= xr.DataArray(data, coords=[dates,variables,stations_numbers,driving_models,experiments], dims=['time', 'var', 'station','model','exp'])


#mean with respect to time in a set station,experiment,model,variable
Mean=temp.mean(dim='time')
mean_detailed=Mean.sel(var='ps',station='Marsdiep Noord',exp='rcp45',model='CNRM-CERFACS-CNRM-CM5')

#mean with respect to time in a set experiment,model,variable in the different stations
Mean=temp_station_numbers.mean(dim='time')
mean_detailed=Mean.sel(var='ps',exp='rcp45',model='CNRM-CERFACS-CNRM-CM5')
mean_detailed.plot()

#PLOTS
#time series
data_type1=temp.sel(var='ps',station='Marsdiep Noord',exp='rcp45',model='CNRM-CERFACS-CNRM-CM5')
#data_type1.plot()

#variables corresponding to the different stations given a SPECIFIC DATE
#pretty irrelevant
data_type2=temp_station_numbers.sel(var='ps',time='2017-01-01',exp='rcp45',model='CNRM-CERFACS-CNRM-CM5')
plt.figure()
data_type2.plot()
