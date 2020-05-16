# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:47:47 2020

@author: simon
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt 
from math import ceil
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

Data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200



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


coordinates = [[52.9836220,4.7502700],[53.0539960,5.0326430],[53.3144720,5.1603010],[53.0854860,5.2876310],[53.2256720,5.2783220],[53.1742859,5.4059014],[53.4022720,5.7274630],[53.4762460,6.0797620],[53.4305470,6.1331940],[53.1769800,5.3969300]]

C1 = [coordinates[i][0] for i in range(len(coordinates))]
C2 = [coordinates[i][1] for i in range(len(coordinates))]


MODELS = ['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 
          'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']
"""
0 'CNRM-CERFACS-CNRM-CM5'
1 'ICHEC-EC-EARTH'
2 'IPSL-IPSL-CM5A-MR'
3 'MOHC-HadGEM2-ES'
4 'MPI-M-MPI-ESM-LR'
"""

VARIABLES = ['Shortwave Radiation',
             'Temperature',
             'Eastward wind',
             'Northward wind',
             'Cloud cover',
             'Relative humidity',
             'Surface pressure']


EXPERIMENTS = ['rcp45','rcp85']


for var in range(len(VARIABLES)):
    D1 = np.zeros((len(STATIONS),len(MODELS)))
    D2 = np.zeros((len(STATIONS),len(MODELS)))
    for mdl in range(len(MODELS)):
        for st in range(len(STATIONS)):
            SubData = Data[:,var,st,mdl,:]
            D1[st,mdl] = -np.mean(SubData[:3650,0])+np.mean(SubData[3650:,0])
            D2[st,mdl] = -np.mean(SubData[:3650,1])+np.mean(SubData[3650:,1])


    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    sns.heatmap(D1[:,:],annot=True, xticklabels = MODELS, yticklabels = STATIONS, ax = ax1,linewidths=.5,cbar = False,fmt='g')
    sns.heatmap(D2[:,:],annot=True, xticklabels = MODELS, yticklabels = False, ax = ax2,linewidths=.5,cbar = False,fmt='g')
    ax1.set_title(VARIABLES[var]+' '+EXPERIMENTS[0])
    ax2.set_title(VARIABLES[var]+' '+EXPERIMENTS[1])
    
    figname = 'DifferenceInVariables_10_years_'+VARIABLES[var]+'_.png'
    fig.savefig(figname, bbox_inches='tight')

