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

VARIABLES = ['Surface Downwelling Shortwave Radiation',
             'Near-surface air temperature',
             'Eastward near-surface wind',
             'Northward near-surface wind',
             'Cloud cover',
             'Near-surface relative humidity',
             'Surface pressure']

exp = 0
EXPERIMENTS = ['rcp45','rcp85']

CLUSTERS = [2,4]

for st in range(9,10):
    for mdl in range(3,4):
        for exp in range(len(EXPERIMENTS)):
            for n_cl in range(len(CLUSTERS)):  
                SubData = Data[:,:,:,mdl,exp]
                
                std = 1
                for i in range(SubData.shape[1]):
                    SubData[:,i,:] = SubData[:,i,:]-SubData[:,i,:].mean()
                    SubData[:,i,:] = SubData[:,i,:]/(SubData[:,i,:].std(ddof = std))
                
                SubDataStation = SubData[:,:,st] #Select the station that we are going to analyse
                
                range_n_clusters = [CLUSTERS[n_cl]]
                
                for n_clusters in range_n_clusters:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(SubDataStation)
                    cluster_labels = clusterer.labels_
                    silhouette_avg = silhouette_score(SubDataStation, cluster_labels)
                    
                    
                    
                    print("For n_clusters =", n_clusters, 
                          "The average silhouette_score is :", silhouette_avg)
                    sample_silhouette_values = silhouette_samples(SubDataStation, cluster_labels)
                   
                    
                cluster_labels_array = np.array(cluster_labels)
                n_in_clusters = []
                n_in_clusteri = 0
                for i in range(range_n_clusters[0]):
                    n_in_clusteri = len(np.where(cluster_labels_array == i)[0])
                    n_in_clusters.append(n_in_clusteri)
                    print('The number of datapoints in cluster '+str(i)+' is: '+str(n_in_clusteri))
                    
                
                
                ## Plotting the clusters in bar plots over the months and years
                Month_Counter = MonthCounter(cluster_labels,n_clusters)
                Year_Counter = YearCounter(cluster_labels,n_clusters)
    #            
    #            fig, axes = plt.subplots(nrows=1, ncols=2)
    #            ax0, ax1 = axes.flatten()
    #            
    #            ax0.hist(Month_Counter, 12, density=True, histtype='bar')
    #            ax0.legend(prop={'size': 10})
    #            ax0.set_title('Divide in months')
    #            
    #            
    #            ax1.hist(Year_Counter, 10, density=True, histtype='bar')
    #            ax1.legend(prop={'size': 10})
    #            ax1.set_title('Divide in years')
    #            
    #            ## Plotting the variables in box plots of the different clusters standardised
    #            fig = plt.figure(2, figsize = (30,16))
    #            fig.suptitle('Paramater destribution per cluster standardised',size = 'xx-large')
    #            LAB = ['cluster 1', 'cluster 2']
    #            for var in range(7):
    #            #var = 0
    #                ax = fig.add_subplot(4, 2, var+1)
    #                ax.set_title(VARIABLES[var])
    #                M = []
    #                INDX = []
    #                for cl in range(n_clusters):
    #                    INDX = np.where(cluster_labels_array == cl)
    #                    new = SubDataStation[INDX,var]
    #                    M.append(new[0])
    #            #    M.append(Data[:,var,st, mdl, exp])
    #                ax.boxplot(M)
    #                plt.grid(True)
    #            fig.savefig('Clustering_exploring_variables_percluster_standardised.png', bbox_inches='tight')
    #            
                Data = np.load('../Datares/tensor_daily_mean_5D.npy')
                NanINDX = np.argwhere(np.isnan(Data))
                for i in range(len(NanINDX)):
                    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200
                
                ## Plotting the variables in box plots of the different clusters 
                fig = plt.figure(3, figsize = (18,10))
                Suptitle = 'Distribution: '+ STATIONS[st]+', '+MODELS[mdl]+', '+EXPERIMENTS[exp]+', '+str(range_n_clusters[0])+' clusters'
                fig.suptitle(Suptitle,size = 'xx-large')
                for var in range(7):
                #var = 0
                    ax = fig.add_subplot(2, 4, var+1)
                    ax.set_title(VARIABLES[var])
                    M = []
                    INDX = []
                    for cl in range(n_clusters):
                        INDX = np.where(cluster_labels_array == cl)
                        new = Data[INDX,var,st, mdl, exp]
                        M.append(new[0])
                #    M.append(Data[:,var,st, mdl, exp])
                    ax.boxplot(M)
                    plt.grid(True)
                
                ax = fig.add_subplot(2,4,8)  
                leg = ['Cluster '+str(i+1) for i in range(range_n_clusters[0])] 
                ax.hist(Month_Counter ,12, label = leg, density=True, histtype='bar')
                ax.legend(prop={'size': 10})
                ax.set_title('Clustering distribution over the year')
                figname = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[exp]+'_'+str(range_n_clusters[0])+'_clusters.png'
                fig.savefig(figname, bbox_inches='tight')
                
                plt.show()
                plt.close('all')
                
                
                #Export clusters to .npy file
                A = np.zeros((CLUSTERS[n_cl],2*len(VARIABLES)+1))
                for c in range(CLUSTERS[n_cl]):
                    A[c,0]= n_in_clusters[c]
                    for v in range(len(VARIABLES)):
                        INDX = np.where(cluster_labels_array == c)
                        new = Data[INDX,v,st, mdl, exp]
                        A[c,2*v+1] = np.mean(new)
                        A[c,2*v+2] = np.std(new)
                filename = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[exp]+'_'+str(range_n_clusters[0])+'_clusters.npy'
                np.save(filename,A)
                    