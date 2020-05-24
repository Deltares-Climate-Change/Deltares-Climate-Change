#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:04:30 2020

@author: maaike

Create a figure that summarizes the data for all stations and variables for one
combination of model and experiment in boxplots.

Both the total data, as well as the lowest and highest 10% of the points are shown
(the most extreme conditions)
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt 
from math import ceil


#Load the data and remove the NaN entries
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


#Select the desired model and experiment
mdl = 0
exp = 0

#Make a selection of the data
SubData = Data[:,:,:,mdl,exp]

#Create the figure
fig, axes = plt.subplots(ncols = 3, nrows = len(VARIABLES), figsize = (18,30))
ax0 = axes[0,0]
ax1 = axes[0,1]
ax2 = axes[0,2]
ax0.set_title('lowest 10 percent')
ax1.set_title('All values')
ax2.set_title('highest 10 percent')

for var in range(len(VARIABLES)):
    Val_low = []
    Val = []
    Val_high = []
    for st in range(10):         
        SubDataStation = SubData[:,var,st] #Select the station that we are going to analyse
        k  = ceil(len(SubDataStation)/10)
        result = np.argpartition(SubDataStation, k)
        Val_low.append(SubDataStation[result[:k]])
        Val_high.append(SubDataStation[result[k:]])
        Val.append(SubDataStation)

        
    ax0 = axes[var,0]
    ax1 = axes[var,1]
    ax2 = axes[var,2]
    #ax0.ylabel(VARIABLES[var])
    if var == len(VARIABLES)-1:
        LAB = STATIONS
    else: 
        LAB = 10*['']
    ax0.boxplot(Val_low)
    ax0.set_ylabel(VARIABLES[var])
    ax1.boxplot(Val)
    ax2.boxplot(Val_high)
    ax0.set_xticklabels(labels = LAB,rotation=-90, fontsize=8)
    ax1.set_xticklabels(labels = LAB,rotation=-90, fontsize=8)
    ax2.set_xticklabels(labels = LAB,rotation=-90, fontsize=8)
    
figname = 'CompareStations'+MODELS[mdl]+'_'+EXPERIMENTS[exp]+'.png'
fig.savefig(figname, bbox_inches='tight')
        
plt.show()





            
