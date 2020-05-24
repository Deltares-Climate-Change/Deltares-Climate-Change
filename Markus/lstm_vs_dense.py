import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.common import *
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl



#Importing the Data
data = np.load('Datares/tensor_daily_mean_5D.npy')
lstm_preds = np.load('Markus/Saved_Arrays/lstm_preds.npy')
dense_preds = np.load('Markus/Saved_Arrays/dense_preds.npy')
ytrue = np.load('Markus/Saved_Arrays/ytrue10k.npy')
X = data_to_xarray(data)


models = np.array(X.coords['model'])
stations = np.array(X.coords['station'])
variables = np.array(X.coords['var'])


sel_var = 1
nobs = 365
treshold = 10000
alpha = 1
minp = 20
model_input = np.array(X.sel(station = stations[0], model = models[0], exp = 'rcp85')).transpose()[:,treshold:]
xrange = treshold + np.arange(nobs)
xdates = X['time'][xrange]

fig, ax = plt.subplots(figsize = (20,10))
plt.plot(xdates,pd.DataFrame(lstm_preds[sel_var,:nobs]).ewm(alpha = alpha, min_periods = minp).mean(), label = 'LSTM Predictions for ' + models[2])
#plt.plot(xdates,pd.DataFrame(model_input[sel_var,:nobs]).ewm(alpha= alpha, min_periods = minp).mean(), label = models[1], linestyle = '--')
plt.plot(xdates,pd.DataFrame(dense_preds[sel_var,:nobs]).ewm(alpha = alpha, min_periods = minp).mean(), label = 'Dense Predictions for ' + models[2], linestyle = ':', color = 'red')
plt.plot(xdates,pd.DataFrame(ytrue[sel_var,:nobs]).ewm(alpha = alpha, min_periods = minp).mean(), label = models[2], color= 'green')
plt.legend()
plt.ylabel(variables[sel_var])
fig.autofmt_xdate()

#plt.savefig('densevslstm1_year1.png')
