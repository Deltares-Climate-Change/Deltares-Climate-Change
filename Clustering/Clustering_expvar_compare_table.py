# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:35:12 2020

@author: simon
"""

from matplotlib import pyplot as plt
import numpy as np
randn = np.random.randn
from pandas import *
import seaborn as sns

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


EXPERIMENTS = ['rcp45','rcp85']

CLUSTERS = [2,3,4,8]


st = 0


for n_cl in range(len(CLUSTERS)):
    fig = plt.figure(figsize=(30,16))
    for var in range(len(VARIABLES)):
        D1 = np.zeros((len(MODELS),CLUSTERS[n_cl]))
        D2 = np.zeros((len(MODELS),CLUSTERS[n_cl]))
        for mdl in range(len(MODELS)):
            Filename = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[0]+'_'+str(CLUSTERS[n_cl])+'_clusters.npy'
            A = np.load(Filename)
            Filename = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[1]+'_'+str(CLUSTERS[n_cl])+'_clusters.npy'
            B = np.load(Filename)
            
            idxA = np.flip(np.argsort(A[:,2*var+1]))
            idxB = np.flip(np.argsort(B[:,2*var+1]))
            
#            ax0 = fig.add_subplot(4, 4, 2*var+1)
#            ax1 = fig.add_subplot(4, 4, 2*var+2)
            
            D1[mdl,:] = A[:,2*var+1][idxA]
            D2[mdl,:] = B[:,2*var+1][idxB]
            
        ax1 = fig.add_subplot(4,4,2*var+1)
        ax2 = fig.add_subplot(4,4,2*var+2)
        if var%2 ==0:
            yticks = MODELS
        else:
            yticks = False
        sns.heatmap(D1,annot=True, xticklabels = False, yticklabels = yticks, ax = ax1,linewidths=.5,cbar = False)
        sns.heatmap(D2,annot=True, xticklabels = False, yticklabels = False, ax = ax2,linewidths=.5,cbar = False)
        ax1.set_title(VARIABLES[var]+' '+EXPERIMENTS[0])
        ax2.set_title(VARIABLES[var]+' '+EXPERIMENTS[1])


    plt.show()
    figname = 'ClusExpVar_tableCompare_models_with_station_'+STATIONS[st]+'_'+str(CLUSTERS[n_cl])+'_clusters.png'
    fig.savefig(figname, bbox_inches='tight')
