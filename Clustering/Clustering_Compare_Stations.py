#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:04:30 2020

@author: maaike
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt 
from math import ceil
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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

def plotpt(ax, extent=(4.2,6.6,52.8,53.7), **kwargs):
    ax.plot(C2,C1, 'r*', ms=20, **kwargs)
    for i in range(len(coordinates)):
        if i == 9:
            H_al = 'left'
        else:
            H_al = 'right'
        plt.text(coordinates[i][1], coordinates[i][0], STATIONS[i],
                 horizontalalignment=H_al,
                 verticalalignment = 'bottom',
                 transform=ccrs.Geodetic())
    ax.set_extent(extent)
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


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

mdl = 0
exp = 0

for mdl in range(len(MODELS)):
    for exp in range(len(EXPERIMENTS)):
        SubData = Data[:,:,:,mdl,exp]
        
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
        plt.close()

for mdl in range(4):
    fig = plt.figure(figsize=(30,16))
    D1 = np.zeros((len(STATIONS),4,len(VARIABLES)))
    D2 = np.zeros((len(STATIONS),4,len(VARIABLES)))
    for st in range(len(STATIONS)):
        Filename = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[0]+'_4_clusters.npy'
        A = np.load(Filename)
        Filename = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[1]+'_4_clusters.npy'
        B = np.load(Filename)
        for var in range(len(VARIABLES)):
            idxA = np.flip(np.argsort(A[:,2*var+1]))
            idxB = np.flip(np.argsort(B[:,2*var+1]))
        
#            ax0 = fig.add_subplot(4, 4, 2*var+1)
#            ax1 = fig.add_subplot(4, 4, 2*var+2)
        
            D1[st,:,var] = A[:,2*var+1][idxA]
            D2[st,:,var] = B[:,2*var+1][idxB]
    
    for var in range(len(VARIABLES)):    
        ax1 = fig.add_subplot(4,4,2*var+1)
        ax2 = fig.add_subplot(4,4,2*var+2)
        if var%2 ==0:
            yticks = STATIONS
        else:
            yticks = False
        sns.heatmap(D1[:,:,var],annot=True, xticklabels = False, yticklabels = yticks, ax = ax1,linewidths=.5,cbar = False,fmt='g')
        sns.heatmap(D2[:,:,var],annot=True, xticklabels = False, yticklabels = False, ax = ax2,linewidths=.5,cbar = False,fmt='g')
        ax1.set_title(VARIABLES[var]+' '+EXPERIMENTS[0])
        ax2.set_title(VARIABLES[var]+' '+EXPERIMENTS[1])
        
    ax = plt.subplot(428, projection=ccrs.Mercator())
    plotpt(ax, transform=ccrs.PlateCarree())
    plt.show()
    figname = 'ClusExpVar_tableCompare_models_with_model_'+MODELS[mdl]+'_.png'
    fig.savefig(figname, bbox_inches='tight')





            