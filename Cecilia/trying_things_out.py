#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:51:40 2020

@author: ceciliacasolo
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

stations = ['Marsdiep Noord','Doove Balg West',
                'Vliestroom','Doove Balg Oost',
                'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                'Harlingen Havenmond West']
variables = ["rsds","tas","uas","vas","clt","hurs","ps"]
MODELS = ['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 
          'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']


data = np.load('/Users/ceciliacasolo/Desktop/Data_CC/tensor_daily_mean_5D.npy')
X=data_to_xarray(data)
X_prime, Y_prime, xstds, ystds, xmeans, ymeans = normalize_data(X)

X_train = torch.tensor(np.array(X_prime[:1000,:]), dtype = torch.float32)
Y_train = torch.tensor(np.array(X_prime[:1000,:]), dtype = torch.float32)
X_train=X_train[:,1]
Y_train=Y_train[:,1]
 
X_test = torch.tensor(np.array(X_prime[1000:,:]), dtype = torch.float32)
Y_test = torch.tensor(np.array(X_prime[1000:,:]), dtype = torch.float32)
X_test=X_test[:,1]
Y_test=Y_test[:,1]

nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(X_train, Y_train)
print(nca_pipe.score(X_test, Y_test))

from sklearn.neighbors import NearestNeighbors
import numpy as np
var1 = X.sel(station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
var2 = X.sel(station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'ICHEC-EC-EARTH')
#newvars=xr.concat([var1, var2], 'var')
#newvars=newvars.transpose()
newvars_train=newvars[:1000,]
newvars_test=newvars[1000:,]

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(var1[:1000,])
distances, indices = nbrs.kneighbors(newvars_test)


from sklearn import preprocessing
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(var1[:1000,],var2[:1000,])
X_train=var1[:1000,]
Y_train=var2[:1000,]
X_test=var1[1000:,]
Y_test=var2[1000:,]
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y_train)
clf = KNeighborsClassifier(n_neighbors=5,algorithm='ball_tree')
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
#print("Test set predictions: {}".format(predictions))
accuracy = clf.score(X_test, Y_test)
print("Test set accuracy: {:.2f}".format(accuracy))


#PART PREDICTION TO RUN
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


# plotting
xlst = np.arange(33238)

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
xlst = np.arange(6000)
plt.plot(xlst,pred[:6000],label='Prediction')
plt.plot(xlst,mod2[:6000],label='Ground truth')
plt.legend()
plt.show()

dates= pd.date_range(start="2006-01-01",end="2096-12-31")
pred2 = xr.DataArray(pred, coords=[dates,variables], dims=['time', 'var'])
var1_pred=pred2.sel(var='rsds')
var1_real=Y.sel(var='rsds',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'ICHEC-EC-EARTH')
plt.plot(xlst,var1_pred[:6000],label='Prediction')
plt.plot(xlst,var1_real[:6000],label='Ground truth')
plt.legend()
plt.show()

c=var1_pred.to_pandas()
d=var1_real.to_pandas()
c.ewm(alpha=0.0001,min_periods= 2000).mean().plot()
d.ewm(alpha=0.0001,min_periods= 2000).mean().plot()



#Variables
var1 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
var2 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'ICHEC-EC-EARTH')
var3 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'IPSL-IPSL-CM5A-MR')
var4 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'MOHC-HadGEM2-ES')
var5 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'MPI-M-MPI-ESM-LR')
newvars=xr.concat([var1, var2, var3, var4], 'var').transpose()

k = 5
training = 6000

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


#ewm
dates= pd.date_range(start="2006-01-01",end="2096-12-31")
pred = xr.DataArray(pred, coords=[dates], dims=['time'])
pred2=pred.to_pandas()
#var5=var5.to_pandas()
pred2.ewm(alpha=0.0001,min_periods= 2000).mean().plot()
var5.ewm(alpha=0.0001,min_periods= 2000).mean().plot()



#VARIABLE ONLY
variables = ["rsds","tas","uas","vas","clt","hurs","ps"]
#Variables
var1 = X.sel(var='hurs',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
var2 = X.sel(var='ps',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
var3 = X.sel(var='rsds',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
var4 = X.sel(var='uas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
var5 = X.sel(var='vas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
var6 = X.sel(var='clt',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
var7 = X.sel(var='tas',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')

newvars=xr.concat([var1, var2, var3, var4, var5, var6], 'var').transpose()

k = 10
training = 6000

nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(newvars[:training])
distances, indices = nbrs.kneighbors(newvars[training:])

pred=np.zeros(33238)
pred[:training]=var7[:training]

p=training
for i in range(len(indices)):
    index = indices[i]
    dist = distances[i]
    appr = 0.
    for j in range(k):
        fact = dist[j]/sum(dist)
        appr = appr+np.array(var7[index[j]])*fact
    pred[p] = appr
    p=p+1

# plotting
xlst = np.arange(6000)
plt.plot(xlst,pred[:6000],label='Prediction')
plt.plot(xlst,var7[:6000],label='Ground truth')
plt.legend()
plt.show()

dates= pd.date_range(start="2006-01-01",end="2096-12-31")
pred = xr.DataArray(pred, coords=[dates], dims=['time'])

plt.plot(yearly_average(pred),label='prediction')
plt.plot(yearly_average(var5),label='Ground truth')
plt.legend()
plt.plot()


#ewm
dates= pd.date_range(start="2006-01-01",end="2096-12-31")
pred = xr.DataArray(pred, coords=[dates], dims=['time'])
pred2=pred.to_pandas()
var7=var7.to_pandas()
pred2.ewm(alpha=0.05,min_periods= 2000).mean().plot(figsize=(15, 10))
var7.ewm(alpha=0.05,min_periods= 2000).mean().plot(figsize=(15, 10))









