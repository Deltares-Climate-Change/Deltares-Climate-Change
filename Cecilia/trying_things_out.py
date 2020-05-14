#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:51:40 2020

@author: ceciliacasolo
"""
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def normalize_data(X):
    X_prime =  X.sel(station = stations[0], model = model_input, exp = 'rcp45')
    Y_prime = X.sel(station = stations[0], model = model_output, exp = 'rcp45')

    xmeans = X_prime.mean(dim = 'time')
    ymeans = Y_prime.mean(dim = 'time')
    xstds = X_prime.std(dim = 'time')
    ystds = Y_prime.std(dim = 'time')

    for i in range(len(variables)):
        X_prime[dict(var = i)] = (X_prime[dict(var = i)] - xmeans[i])/xstds[i]
        Y_prime[dict(var = i)] = (Y_prime[dict(var = i)] -ymeans[i])/ystds[i]

    print(X_prime.mean(dim= 'time'))
    print(Y_prime.mean(dim = 'time'))

    return X_prime, Y_prime, xstds, ystds, xmeans, ymeans


X_prime, Y_prime, xstds, ystds, xmeans, ymeans = normalize_data(X)

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


#PART PREDECTION
k = 5
training = 1000

nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(var1[:training])
distances, indices = nbrs.kneighbors(var1[training:])

pred = np.zeros([33238,7])
pred[:training] = var2[:training]

j=training
for index in indices:
    appr = np.zeros([1,7])
    for i in range(k):
        appr = appr+np.array(var2[index[i-1]])*(distances[j-1,i-1])/(distances[j-1,0]+distances[j-1,2]+distances[j-1,3]+distances[j-1,4]+distances[j-1,1])
    appr = appr/k
    pred[j] = appr
    j=j+1

# plotting
xlst = np.arange(33238)

