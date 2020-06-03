#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:09:47 2020

@author: ceciliacasolo
"""

#!/usr/bin/env python
# coding: utf-8


# In[22]:


import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from datetime import timedelta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

# In[24]:

data = np.load('/Users/ceciliacasolo/Desktop/Data_CC/tensor_daily_mean_5D.npy')


# In[51]:


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

stations = ['Marsdiep Noord','Doove Balg West',
                'Vliestroom','Doove Balg Oost',
                'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                'Harlingen Havenmond West']
model_input = 'CNRM-CERFACS-CNRM-CM5'
model_output = 'IPSL-IPSL-CM5A-MR'
variables = ["rsds","tas","uas","vas","clt","hurs","ps"]
MODELS = ['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 
          'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']

# In[45]:


def normalize_data(X):
    X_prime =  X.sel(station = stations[0], model = model_input, exp = 'rcp45')
    Y_prime = X.sel(var='rsds',station = stations[0], model = model_input, exp = 'rcp45')

    xmeans = X_prime.mean(dim = 'time')
    ymeans = Y_prime.mean(dim = 'time')
    xstds = X_prime.std(dim = 'time')
    ystds = Y_prime.std(dim = 'time')
    Y_prime = (Y_prime -ymeans)/ystds

    for i in range(len(variables)):
        X_prime[dict(var = i)] = (X_prime[dict(var = i)] - xmeans[i])/xstds[i]
        
    print(X_prime.mean(dim= 'time'))
    print(Y_prime.mean(dim = 'time'))

    return X_prime, Y_prime, xstds, ystds, xmeans, ymeans


# In[46]:


X=data_to_xarray(data)


# In[87]:


class Net(nn.Module):
    def __init__(self): #scheleton architecure
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1) #3 fully connected layers
    
    def forward(self, x): #how data flows through out network
        x = F.relu(self.fc1(x)) #ReLu activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.relu(x) 
        


# In[88]:


Net()


# In[89]:


#create an instance of network
net = Net()
print(net)


# In[91]:


K=1000
learning_rate=0.01
X_prime, Y_prime, xstds, ystds, xmeans, ymeans = normalize_data(X)
train_x = torch.tensor(np.array(X_prime[:K,:]), dtype = torch.float32)
train_y = torch.tensor(np.array(Y_prime[:K]), dtype = torch.float32)
test_x = torch.tensor(np.array(X_prime[(K+1):,:]), dtype = torch.float32)
test_y = torch.tensor(np.array(Y_prime[(K+1):]), dtype = torch.float32)

# In[92]:

# create a stochastic gradient descent optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# In[96]:

# run the main training loop
for epoch in range(K):
    batch_idx = np.random.randint(low = 0,high = K, size = 50)
    optimizer.zero_grad()
    data=train_x[batch_idx,:]
    net_out = net(data)
    loss = F.mse_loss(net_out, train_y[batch_idx])
    loss.backward()
    optimizer.step()
    if (epoch % 100 == 0):
        print("Epoch: " + str(epoch))
        print("Loss: " + str(loss))


# In[107]:


net_out_train = net(train_x)
net_out_test = net(test_x)
loss_train = F.mse_loss(net_out_train, train_y)
loss_error = F.mse_loss(net_out_test, test_y)
print(loss_train)
print(loss_error)


# In[111]:


data.shape


# In[ ]:





# In[ ]:

