# -*- coding: utf-8 -*-
"""
This file performs the k-nearest neighbours approach for predicting one model from another, as explained
in section 7.1 of the report. It also plots all the figures in that section.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.neighbors import NearestNeighbors

# Define a function to import the given data set
def data_to_xarray(data):
    """
    Takes the (preprocessed) data and returns a labeled xarray
    and imputes NaN values.
    """
    dates= pd.date_range(start="2006-01-01",end="2096-12-31")
    experiments=['rcp45','rcp85']
    variables=["rsds","tas","uas","vas","clt","hurs","ps"]
    stations=['Marsdiep Noord','Doove Balg West',
                    'Vliestroom','Doove Balg Oost',
                    'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                    'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                    'Harlingen Havenmond West']
    driving_models=['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']

    X = xr.DataArray(data, coords=[dates,variables,stations,driving_models,experiments], dims=['time', 'var', 'station','model','exp'])
    #Fills NaN Values appropriately
    X = X.interpolate_na(dim = 'time', method = 'nearest')

    return X

# Import the data set and create a normalised dataset, convert both to xarray
# Note that np.load should contain the device specific path to the data set
data = np.load('/Users/Rens/Desktop/Data_CC/tensor_daily_mean_5D.npy')
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

# Select correct station, experiment and model
# Station and experiment are chosen constant here
# Model ICHEC-EC-EARTH is used to predict model IPSL-IPSL-CM5A-MR
mod1N = Y.sel(station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'ICHEC-EC-EARTH')
mod2 = X.sel(station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'IPSL-IPSL-CM5A-MR')

# Define parameters, k is the number of nearest neighbours, training is the size of the training set in days
k = 10 
training = 6000 

# Perform the nearest neighbours search on the normalised data from the first model
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(mod1N[:training])
distances, indices = nbrs.kneighbors(mod1N[training:])

# Define an array for the prediction, and set the first part equal to the training set
pred = np.zeros([33238,7])
pred[:training] = mod2[:training]

# Make the prediction using the nearest neighbours algorithm
p=training
for i in range(len(indices)):
    index = indices[i]
    dist = distances[i]
    appr = np.zeros([1,7])
    for j in range(k):
        fact = dist[j]/sum(dist)
        appr = appr+np.array(mod2[index[j]])*fact
    pred[p] = appr
    p=p+1

# Convert prediction array pred to an xarray with the correct labels
dates= pd.date_range(start="2006-01-01",end="2096-12-31")
variables=["rsds","tas","uas","vas","clt","hurs","ps"]
pred = xr.DataArray(pred, coords=[dates,variables], dims=['time','var'])

# Plot the results of the prediction against the actual model data, using the exponentially weighted average with 
# alpha=0.05 and min_periods=200
# plotting determines the type of plot that is made: 0 for single plot, 1 for zoomed single plot,
# 2 for all plots together
# var determines which var is plotted in the single plot
plotting = 2 
var = 0

if plotting==0:
    ypred = pd.DataFrame({'y':pred.sel(var=variables[var])}).ewm(alpha=0.05,min_periods=200).mean()
    ytrue = pd.DataFrame({'y':mod2.sel(var=variables[var])}).ewm(alpha=0.05,min_periods=200).mean()
    
    ypred = np.array(ypred)[:,0]
    ytrue = np.array(ytrue)[:,0]
    
    plt.plot(dates,ypred, label = 'Network Predictions')
    plt.plot(dates,ytrue, label = 'Ground Truth')
    plt.legend()
    plt.show()

if plotting==1:
    ypred = pd.DataFrame({'y':pred.sel(var=variables[var])}).ewm(alpha=0.05,min_periods=200).mean()
    ytrue = pd.DataFrame({'y':mod2.sel(var=variables[var])}).ewm(alpha=0.05,min_periods=200).mean()
    
    ypred = np.array(ypred)[:,0]
    ytrue = np.array(ytrue)[:,0]
    
    plt.plot(dates[24000:25000],ypred[24000:25000], label = 'Network Predictions')
    plt.plot(dates[24000:25000],ytrue[24000:25000], label = 'Ground Truth')
    plt.legend()
    plt.show()    

if plotting==2:
    fig, axs = plt.subplots(len(variables),1, figsize = (30,50))
    for sel_var in range(len(variables)):
        ypred = pd.DataFrame({'y':pred.sel(var=variables[sel_var])}).ewm(alpha=0.05,min_periods=200).mean()
        ytrue = pd.DataFrame({'y':mod2.sel(var=variables[sel_var])}).ewm(alpha=0.05,min_periods=200).mean()
        
        ypred = np.array(ypred)[:,0]
        ytrue = np.array(ytrue)[:,0]
        
        plt.sca(axs[sel_var])
        
        plt.plot(dates,ypred, label = 'Network Predictions')
        plt.plot(dates,ytrue, label = 'Ground Truth')
        plt.title(str(variables[sel_var]))
        plt.legend()
    plt.show()

