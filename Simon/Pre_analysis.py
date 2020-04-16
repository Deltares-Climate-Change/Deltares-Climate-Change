# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:37:47 2020

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt 
Data = np.load('tensor_daily_mean_5D.npy')

def YearlyAverage(Data):
    st = 0
    AvDataLst = []
    for i in range(91):
        if i%4 == 2:
            SubData = Data[st:st+366,:,:,:,:]
            SubAv = np.average(SubData, 0)
            st = st+366
        else:
            SubData = Data[st:st+365,:,:,:,:]
            SubAv = np.average(SubData, 0)
            st = st+365
        AvDataLst.append(SubAv)
    AvData = np.stack(AvDataLst,axis = 0)
    print(st)
    return AvData
def DevideInSeasons(Data):
    Winter = []
    Spring = []
    Summer = []
    
var= 1
VARIABLES = ['rsds','tas','uas','vas','clt','hurs','ps']
"""
0 rsds - Surface Downwelling Shortwave Radiation
1 tas - near surface air temperature
2 uas - Eastward Near-Surface Wind
3 vas - Northward Near-Surface Wind
4 clt - Total Cloud Cover
5 hurs - Near-Surface Relative Humidity
6 ps - Surface Pressure
"""
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
MODELS = ['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']
"""
0 'CNRM-CERFACS-CNRM-CM5'
1 'ICHEC-EC-EARTH'
2 'IPSL-IPSL-CM5A-MR'
3 'MOHC-HadGEM2-ES'
4 'MPI-M-MPI-ESM-LR'
"""
exp = 0
EXPERIMENTS = ['rcp45','rcp85']

YA = YearlyAverage(Data)
fig = plt.figure(1, figsize=(12,12))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i in range(5):
    ax1.plot(list(range(2006,2097)),YA[:,var,st,i,0], 'o', label = MODELS[i])
    ax2.plot(list(range(2006,2097)),YA[:,var,st,i,1], 'o', label = MODELS[i])
ax1.legend()
ax1.set_xlabel('time [years]')
ax2.set_xlabel('time [years]')
plt.subplot(211)
plt.grid(True)
plt.title(EXPERIMENTS[0])
plt.subplot(212)
plt.grid(True)
plt.title(EXPERIMENTS[1])
fig.suptitle('Temperature evolution according to different models')

fig.savefig('DATA_EXPLORATION_temperature_evolution_models.png', bbox_inches='tight')

fig = plt.figure(2, figsize = (12,16))
fig.suptitle('Evolution of paramater destribution')


for i in range(7):
    ax = fig.add_subplot(4,2,i+1)
    plt.grid(True)
    ax.boxplot([Data[0:365,i,st,mdl,exp],Data[-365:,i,st,mdl,exp]],labels = [2006, 2096])
    plt.title(VARIABLES[i],size = 'xx-large')
fig.savefig('DATA_EXPLORATION_evolution_spread.png', bbox_inches='tight')

#for i in range(len(STATIONS)):
#    SelectedData = Data[:,var,i,mdl,exp]
#    print('Selected data')
#    print(VARIABLES[var],' ', STATIONS[st],' ', MODELS[mdl],' ',EXPERIMENTS[exp])
#
#A = np.transpose(SelectedData)
#data_to_plot = A.tolist()
#
## Create a figure instance
#fig = plt.figure(1, figsize=(9, 6))
#
## Create an axes instance
#ax = fig.add_subplot(111)
#
## Create the boxplot
#bp = ax.boxplot(data_to_plot)
#
## Save the figure
#fig.savefig('fig1.png', bbox_inches='tight')
