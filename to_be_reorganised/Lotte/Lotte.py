# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:56:22 2020

@author: Lotte
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.common import *
import seaborn as sns
from scipy.stats import spearmanr
import xarray as xr

##importing data
data = np.load('Datares/tensor_daily_mean_5D.npy')
X = data_to_xarray(data)
X.head()


##making the mean 0 and the standard deviation 1
# std = 1
# for i in range(X.shape[1]):
#     X[:,i] = X[:,i] - X[:,i].mean()
#     X[:,i] = X[:,i]/(X[:,i].std(ddof = std))

##Dates
dates= pd.date_range(start="2006-01-01",end="2096-12-31")

##Variable names
var= 1
variables = ["rsds","tas","uas","vas","clt","hurs","ps"]
VARIABLES = ['Surface Downwelling Shortwave Radiation',
             'Near-surface air temperature',
             'Eastward near-surface wind',
             'Northward near-surface wind',
             'Cloud cover',
             'Near-surface relative humidity',
             'Surface pressure']
"""
0 rsds - Surface Downwelling Shortwave Radiation
1 tas - near surface air temperature
2 uas - Eastward Near-Surface Wind
3 vas - Northward Near-Surface Wind
4 clt - Total Cloud Cover
5 hurs - Near-Surface Relative Humidity
6 ps - Surface Pressure
"""

##Station names
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

##Model names
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

##Experiment names
exp = 0
EXPERIMENTS = ['rcp45','rcp85']


## MARKUS
def create_correlations(X,STATIONS,MODELS):
    corr45 = np.zeros((7,7))
    corr85 = np.zeros((7,7))
    corr45r = np.zeros((7,7))
    corr85r = np.zeros((7,7))

    for station in STATIONS:
        for model in MODELS:
            corr45 += np.corrcoef(X.sel(station= station ,exp= 'rcp45', model= model),rowvar = False)
            corr85 += np.corrcoef(X.sel(station= station ,exp= 'rcp85', model= model),rowvar = False)
            corr45r += spearmanr(X.sel(station= station ,exp= 'rcp45', model= model))[0]
            corr85r += spearmanr(X.sel(station= station ,exp= 'rcp85', model= model))[0]

    corr45 = corr45/(len(STATIONS)*len(MODELS))
    corr85 = corr85/(len(STATIONS)*len(MODELS))
    corr45r = corr45r/(len(STATIONS)*len(MODELS))
    corr85r = corr85r/(len(STATIONS)*len(MODELS))
    return [corr45,corr85,corr45r,corr85r]

def plot_correlations(corr1,corr2):
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (20,7))
    sns.heatmap(corr1,annot=True, xticklabels = variables, yticklabels = variables, ax = ax1)
    sns.heatmap(corr2,annot=True, xticklabels = variables, yticklabels = variables, ax = ax2)

    fig.suptitle('Average correlation over stations and models.')
    #ax1.set_title(str(corr1))
    #ax2.set_title(str(corr2))
    plt.show()
    pass



## LOTTE
Seasons = ['Winter','Spring','Summer','Autumn']
Winter,Spring,Summer,Autumn=divide_in_seasons(X)
seasons_correlations=np.zeros((4,4,7,7))          #season,type correlations,actual correlations
seasons_correlations[0]=create_correlations(Winter,STATIONS,MODELS)
seasons_correlations[1]=create_correlations(Spring,STATIONS,MODELS)
seasons_correlations[2]=create_correlations(Summer,STATIONS,MODELS)
seasons_correlations[3]=create_correlations(Autumn,STATIONS,MODELS)
for i in range(4):
    plot_correlations(seasons_correlations[i][0],seasons_correlations[i][1])
    plot_correlations(seasons_correlations[i][2],seasons_correlations[i][3])
