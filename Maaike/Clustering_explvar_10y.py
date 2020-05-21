#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:04:30 2020

@author: maaike
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def MonthCounter(Labels,n_clusters):
    Dat = [[] for i in range(n_clusters)]
    Dag = datetime.date(2006,1,1)
    for i in range(len(Labels)):
        Dat[Labels[i]].append(Dag.month)
        Dag += datetime.timedelta(days = 1)
    return(Dat)
    
def YearCounter(Labels,n_clusters):
    Dat = [[] for i in range(n_clusters)]
    Dag = datetime.date(2006,1,1)
    for i in range(len(Labels)):
        Dat[Labels[i]].append(Dag.year)
        Dag += datetime.timedelta(days = 1)
    return(Dat)
    
def SortClusters(SubData, cluster_labels_array, n_clusters):
    order_lst = []
    INDX = []
    mean_lst = []
    mean_ordered = []
    new = 0
    for cl in range(n_clusters):
        INDX = np.where(cluster_labels_array == cl)
        new = SubData[INDX,0].mean()
        mean_lst.append(new)
        mean_ordered.append(new)
    mean_ordered.sort()
    for j in range(len(mean_ordered)):
        for i in range(len(mean_lst)):
            if mean_lst[i] == mean_ordered[j]:
                order_lst.append(i)   
    return order_lst
    
#def SortClusters(Month_Counter, n_clusters):
#    order_lst = []
#    maxmonth_lst = []
#    for cl in range(n_clusters):
#        maxmonth_lst.append(max(set(Month_Counter[cl]), key = Month_Counter[cl].count))
#    maxmonth_ordered = maxmonth_lst
#    maxmonth_ordered.sort()
#    for j in range(len(maxmonth_ordered)):
#        for i in range(len(maxmonth_lst)):
#            if maxmonth_lst[i]== maxmonth_ordered[j]:
#                order_lst.append(i)   
#    return order_lst



Data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200


st = 9
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

mdl = 3
MODELS = ['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 
          'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']
"""
0 'CNRM-CERFACS-CNRM-CM5'
1 'ICHEC-EC-EARTH'
2 'IPSL-IPSL-CM5A-MR'
3 'MOHC-HadGEM2-ES'
4 'MPI-M-MPI-ESM-LR'
"""

VARIABLES = ['Surface Downwelling Shortwave Radiation',
             'Near-surface air temperature',
             'Eastward near-surface wind',
             'Northward near-surface wind',
             'Cloud cover',
             'Near-surface relative humidity',
             'Surface pressure']

exp = 0
EXPERIMENTS = ['rcp45','rcp85']

n_clusters = 2
color=['lightsalmon','red','lightsteelblue','blue','lightgreen','green','plum','darkorchid']

 
SubData = Data[:,:,st,mdl,exp]

#standardize data
std = 1
for i in range(SubData.shape[1]):
    SubData[:,i] = SubData[:,i]-SubData[:,i].mean()
    SubData[:,i] = SubData[:,i]/(SubData[:,i].std(ddof = std))


#cluster the data from 2006-2016
SubData1 = SubData[0:3650,:]
clusterer1 = KMeans(n_clusters=n_clusters, random_state=10).fit(SubData1)
cluster_labels1 = clusterer1.labels_
silhouette_avg1 = silhouette_score(SubData1, cluster_labels1)    
print("For the first 10 years and n_clusters =", n_clusters, 
      "The average silhouette_score is :", silhouette_avg1)
sample_silhouette_values1 = silhouette_samples(SubData1, cluster_labels1)
#determine size of the clusters
cluster_labels_array1 = np.array(cluster_labels1)
n_in_clusters1 = []
n_in_clusteri = 0
for i in range(n_clusters):
    n_in_clusteri = len(np.where(cluster_labels_array1 == i)[0])
    n_in_clusters1.append(n_in_clusteri)
    print('The number of datapoints in cluster '+str(i)+' is: '+str(n_in_clusteri))
Month_Counter1 = MonthCounter(cluster_labels1,n_clusters)
Year_Counter1 = YearCounter(cluster_labels1,n_clusters)
order_lst1 = SortClusters(SubData1, cluster_labels_array1, n_clusters)
#order_lst1 = SortClusters(Month_Counter1, n_clusters)


#cluster the data from 2086-2096
SubData2 = SubData[-3650:,:]
clusterer2 = KMeans(n_clusters=n_clusters, random_state=10).fit(SubData2)
cluster_labels2 = clusterer2.labels_
silhouette_avg2 = silhouette_score(SubData2, cluster_labels2)    
print("For the last 10 years and n_clusters =", n_clusters, 
      "The average silhouette_score is :", silhouette_avg2)
sample_silhouette_values2 = silhouette_samples(SubData2, cluster_labels2)
#determine size of the clusters
cluster_labels_array2 = np.array(cluster_labels2)
n_in_clusters2 = []
n_in_clusteri = 0
for i in range(n_clusters):
    n_in_clusteri = len(np.where(cluster_labels_array2 == i)[0])
    n_in_clusters2.append(n_in_clusteri)
    print('The number of datapoints in cluster '+str(i)+' is: '+str(n_in_clusteri)) 
Month_Counter2 = MonthCounter(cluster_labels2,n_clusters)
Year_Counter2 = YearCounter(cluster_labels2,n_clusters)
order_lst2 = SortClusters(SubData2, cluster_labels_array2, n_clusters)
#order_lst2 = SortClusters(Month_Counter2, n_clusters)

#create one combined Month_Counter that is ordered correctly
Month_Counter = []
Month_Counter_ordered1 = []
Month_Counter_ordered2 = []
for i in range(n_clusters):
    Month_Counter.append(Month_Counter1[order_lst1[i]])
    Month_Counter.append(Month_Counter2[order_lst2[i]])
    Month_Counter_ordered1.append(Month_Counter1[order_lst1[i]])
    Month_Counter_ordered2.append(Month_Counter2[order_lst2[i]])
        
#loading the data again
Data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200
Data = Data[:,:,st,mdl,exp]
Data1 = Data[0:3650,:]
Data2 = Data[-3650:,:]
    
#plotting the clusters of both timeslots over a year   
fig, axes = plt.subplots(nrows=1, ncols=2)
ax0, ax1 = axes.flatten()
ax0.hist(Month_Counter_ordered1, 12, density=True, histtype='bar', color=color[0:n_clusters])
ax0.legend(prop={'size': 10})
ax0.set_title('Divide in months')
ax1.hist(Month_Counter_ordered2, 12, density=True, histtype='bar', color=color[n_clusters:n_clusters*2])
ax1.legend(prop={'size': 10})
ax1.set_title('Divide in months')


#Create all the boxplots for the different variables and clusters
fig = plt.figure(2, figsize = (20,10))
#Suptitle = 'Distribution: '+ STATIONS[st]+', '+MODELS[mdl]+', '+EXPERIMENTS[exp]+', '+str(n_clusters)+' clusters'
#fig.suptitle(Suptitle,size = 'xx-large')
xlabel_long = ["'06-'16 I","'86-'96 I","'06-'16 II","'86-'96 II",
               "'06-'16 III","'86-'96 III","'06-'16 IV","'86-'96 IV",]
xlabel = xlabel_long[:n_clusters*2]
for var in range(7):
    ax = fig.add_subplot(2, 4, var+1)
    ax.set_title(VARIABLES[var])
    M = []
    INDX1 = []
    INDX2 = []
    for cl in range(n_clusters):
        cl1 = order_lst1[cl]
        cl2 = order_lst2[cl]
        INDX1 = np.where(cluster_labels_array1 == cl1)
        INDX2 = np.where(cluster_labels_array2 == cl2)
        new1 = Data1[INDX1,var]
        new2 = Data2[INDX2,var]
        M.append(new1[0])
#        xlabel.append(str(cl+1)+'a')
        M.append(new2[0])
#        xlabel.append(str(cl+1)+'b')
    ax.boxplot(M)
    plt.xticks(np.arange(1,n_clusters*2+1),xlabel,rotation=0)
    plt.grid(True)
    
#plot the clusters
ax = fig.add_subplot(2,4,8)  
#leg = ['Cluster 1a','Cluster 1b','Cluster 2a','Cluster 2b']
#       'Cluster 3a','Cluster 3b','Cluster 3a','Cluster 3b',] 
leg = xlabel
ax.hist(Month_Counter ,12, label=leg, density=True, histtype='bar', color=color[0:n_clusters*2])
ax.legend(prop={'size': 10},ncol=2)
ax.set_title('Clustering distribution over the year')
#figname = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[exp]+'_'+str(n_clusters)+'_clusters.png'
figname = 'figure_for_Simon_2_exp45'
fig.savefig(figname, bbox_inches='tight')

plt.show()
#plt.close('all')


##Export clusters to .npy file
#A = np.zeros((CLUSTERS[n_cl],2*len(VARIABLES)+1))
#for c in range(CLUSTERS[n_cl]):
#    A[c,0]= n_in_clusters[c]
#    for v in range(len(VARIABLES)):
#        INDX = np.where(cluster_labels_array == c)
#        new = Data[INDX,v,st, mdl, exp]
#        A[c,2*v+1] = np.mean(new)
#        A[c,2*v+2] = np.std(new)
#filename = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[exp]+'_'+str(range_n_clusters[0])+'_clusters.npy'
#np.save(filename,A)
                    
