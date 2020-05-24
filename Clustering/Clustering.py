# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:56:56 2020
@author: simon
This is the simplest version of the clustering files. 
One can select the desired station, model, and experiment, and the number of clusters.
The script will output the average silhouette score for each number of clusters, the size
of each cluster, and show the different clusters in a histogram where one can 
see how the clusters are spread during the year as well as the spread over the years.

"""
import datetime
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def MonthCounter(Labels,n_clusters):
    "For each cluster, determine in which month the days are located"
    Dat = [[] for i in range(n_clusters)]
    Dag = datetime.date(2006,1,1)
    for i in range(len(Labels)):
        Dat[Labels[i]].append(Dag.month)
        Dag += datetime.timedelta(days = 1)
    return(Dat)
    
def YearCounter(Labels,n_clusters):
    "For each cluster, determine in which year the days are located"
    Dat = [[] for i in range(n_clusters)]
    Dag = datetime.date(2006,1,1)
    for i in range(len(Labels)):
        Dat[Labels[i]].append(Dag.year)
        Dag += datetime.timedelta(days = 1)
    return(Dat)


"Load the preprocessed data from the file placed in Datares"
Data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200
    

"Determine which station, model, and experiment will be analyzed" 
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

mdl = 4
MODELS = ['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 
          'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']
"""
0 'CNRM-CERFACS-CNRM-CM5'
1 'ICHEC-EC-EARTH'
2 'IPSL-IPSL-CM5A-MR'
3 'MOHC-HadGEM2-ES'
4 'MPI-M-MPI-ESM-LR'
"""
exp = 1
EXPERIMENTS = ['rcp45','rcp85']


"Select the appropriate data, and normalize the variables
SubData = Data[:,:,:,mdl,exp]
for i in range(SubData.shape[1]):
    SubData[:,i,:] = SubData[:,i,:]-SubData[:,i,:].mean()
    SubData[:,i,:] = SubData[:,i,:]/(SubData[:,i,:].std(ddof = 1))
SubDataStation = SubData[:,:,st] 


"Perform the clustering
range_n_clusters = [2,3]
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(SubDataStation)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(SubDataStation, cluster_labels)    
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is :", silhouette_avg)
   
   
"Analyze the clusters and plot the results 
 cluster_labels_array = np.array(cluster_labels)
 n_in_clusters = []
 n_in_clusteri = 0
 for i in range(range_n_clusters[0]):
     n_in_clusteri = len(np.where(cluster_labels_array == i)[0])
     n_in_clusters.append(n_in_clusteri)
     print('The number of datapoints in cluster '+str(i)+' is: '+str(n_in_clusteri))
    

 Month_Counter = MonthCounter(cluster_labels,n_clusters)
 Year_Counter = YearCounter(cluster_labels,n_clusters)

 fig, axes = plt.subplots(nrows=2, ncols=2)
 ax0, ax1, ax2, ax3 = axes.flatten()


 ax0.hist(Month_Counter, 12, density=True, histtype='bar')
 ax0.legend(prop={'size': 10})
 ax0.set_title('Divide in months')


 ax1.hist(Year_Counter, 10, density=True, histtype='bar')
 ax1.legend(prop={'size': 10})
 ax1.set_title('Divide in years')


 plt.show()