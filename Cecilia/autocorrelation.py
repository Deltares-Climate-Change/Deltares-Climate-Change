#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:03:02 2020

@author: ceciliacasolo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.common import data_to_xarray
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

data = np.load('/Users/ceciliacasolo/Desktop/Data_CC/tensor_daily_mean_5D.npy')
X = data_to_xarray(data)
X.head()

variables = ["rsds","tas","uas","vas","clt","hurs","ps"]
stations=['Marsdiep Noord','Doove Balg West',
                'Vliestroom','Doove Balg Oost',
                'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                'Harlingen Havenmond West']
driving_models=['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']

c=X.sel(var='rsds',station= 'Marsdiep Noord' ,exp= 'rcp45', model= 'CNRM-CERFACS-CNRM-CM5')
plot_acf(c)
plot_acf(c,lags=33237)
plot_acf(c,lags=3000)
plot_pacf(c,lags=33237)
plot_pacf(c,lags=3000)