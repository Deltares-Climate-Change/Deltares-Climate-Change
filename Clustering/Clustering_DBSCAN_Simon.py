# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:56:56 2020

@author: simon
"""
import datetime
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

plt.close('all')

def MonthCounter(Labels,n_clusters,StartTime):
    Dat = [[] for i in range(n_clusters)]
    Dag = StartTime
    for i in range(len(Labels)):
        if Labels[i]>=0:
            Dat[Labels[i]].append(Dag.month)
            Dag += datetime.timedelta(days = 1)
    return(Dat)
    
def YearCounter(Labels,n_clusters,StartTime):
    Dat = [[] for i in range(n_clusters)]
    Dag = StartTime
    for i in range(len(Labels)):
        if Labels[i]>=0:
            Dat[Labels[i]].append(Dag.year)
            Dag += datetime.timedelta(days = 1)
    return(Dat)

Data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200


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
exp = 0
EXPERIMENTS = ['rcp45','rcp85']

SubData = Data[:,:,:,mdl,exp]

std = 1

for i in range(SubData.shape[1]):
    SubData[:,i,:] = SubData[:,i,:]-SubData[:,i,:].mean()
    SubData[:,i,:] = SubData[:,i,:]/(SubData[:,i,:].std(ddof = std))

SubDataStation = SubData[:,:,st] #Select the station that we are going to analyse



data1 = SubDataStation
clusterer = DBSCAN(eps=3, min_samples=4).fit(data1)
cluster_labels = clusterer.labels_
#silhouette_avg = silhouette_score(data1, cluster_labels)   
#print("For n_clusters =", n_clusters, 
#      "The average silhouette_score is :", silhouette_avg)
#sample_silhouette_values = silhouette_samples(data1, cluster_labels)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)
print('The estimated number of clusters is:, ' ,str(n_clusters) )
print('The estimated number of noisy datapoints is: ',str(n_noise))

Start1 = datetime.date(2006,1,1)


Month_Counter = MonthCounter(cluster_labels,n_clusters,Start1)
Year_Counter = YearCounter(cluster_labels,n_clusters,Start1)



fig, axes = plt.subplots(nrows=1, ncols=2)
ax0, ax1 = axes.flatten()

ax0.hist(Month_Counter, 12, density=True, histtype='bar')
ax0.set_title('Divide in months')


ax1.hist(Year_Counter, 10, density=True, histtype='bar')
ax1.set_title('Divide in years')



