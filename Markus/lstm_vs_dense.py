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
nobs = 2*365
treshold = 10000
model_input = np.array(X.sel(station = stations[0], model = models[0], exp = 'rcp85')).transpose()[:,treshold:]
xrange = treshold + np.arange(nobs)
xdates = X['time'][xrange]

fig, ax = plt.subplots(figsize = (20,10))
plt.plot(xdates,lstm_preds[sel_var,:nobs], label = 'lstm')
plt.plot(xdates,model_input[sel_var,:nobs], label = 'input', linestyle = '--')
plt.plot(xdates,ytrue[sel_var,:nobs], label = 'true')
plt.plot(xdates,dense_preds[sel_var,:nobs], label = 'dense', linestyle = ':')
plt.legend()
plt.ylabel(variables[sel_var])
fig.autofmt_xdate()
