# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:37:47 2020

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt 

Data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200

#=================DEFINE FUNCTIONS======================================
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
    return AvData

def DevideInSeasons(Data):
    Winter = []
    Spring = []
    Summer = []
    Autumn = []
    
    st = 0
    for i in range(91):
        if i%4 == 2:
            Winter.append(Data[st:st+80,:,:,:,:])
            Spring.append(Data[st+80:st+172,:,:,:])
            Summer.append(Data[st+172:st+266,:,:,:])
            Autumn.append(Data[st+266:st+356,:,:,:])
            Winter.append(Data[st+356:st+366,:,:,:])
            st = st + 366
        else:
            Winter.append(Data[st:st+79,:,:,:,:])
            Spring.append(Data[st+79:st+171,:,:,:])
            Summer.append(Data[st+171:st+265,:,:,:])
            Autumn.append(Data[st+265:st+355,:,:,:])
            Winter.append(Data[st+355:st+365,:,:,:])
            st = st + 365
    Wi = np.concatenate(Winter,axis = 0)
    Sp = np.concatenate(Spring,axis = 0)
    Su = np.concatenate(Summer,axis = 0)
    Au = np.concatenate(Autumn,axis = 0)
    return [Wi,Sp,Su,Au]


#===============SELECT THE RELEVANT STATION, MODEL, EXPERIMENT=============

    
var= 1
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
exp = 0
EXPERIMENTS = ['rcp45','rcp85']


#==============CREATE PLOTS FOR PRE-ANALYSIS OF THE DATA======================


YA = YearlyAverage(Data)               #Array with yearly averages
[Wi,Sp,Su,Au] = DevideInSeasons(Data)  #Arrays with selected seasons


PLT = True
if PLT == True:
    
    "Plot the evolution of the temperature according to some models"
    fig = plt.figure(1, figsize = (12,8))
#    fig.suptitle('Temperature evolution according to different models')
    for i in range(5):
        plt.plot(list(range(2006,2097)),YA[:,var,st,i,0], 'o', label = MODELS[i])
    plt.legend()
    plt.xlabel('time [years]')
    plt.ylabel('temperature [degrees Celcius]')
    plt.grid(True)
    plt.title(EXPERIMENTS[0])
    fig.savefig('dataexpl_tempevol1.png', bbox_inches='tight')
    
    fig = plt.figure(2, figsize = (12,8))
    for i in range(5):
        plt.plot(list(range(2006,2097)),YA[:,var,st,i,1], 'o', label = MODELS[i])
    plt.legend()
    plt.xlabel('time [years]')
    plt.ylabel('temperature [degrees Celcius]')
    plt.grid(True)
    plt.title(EXPERIMENTS[1])
    fig.savefig('dataexpl_tempevol2.png', bbox_inches='tight')
    
    
    "Boxplot parameters in 2006 and 2096"
    fig = plt.figure(3, figsize = (12,16))
#    fig.suptitle('Evolution of paramater destribution')
    for i in range(7):
        ax = fig.add_subplot(4,2,i+1)
        plt.grid(True)
        ax.boxplot([Data[0:365,i,st,mdl,exp],Data[-365:,i,st,mdl,exp]],labels = [2006, 2096])
        plt.title(VARIABLES[i],size = 'large')
    fig.savefig('dataexpl_param0669.png', bbox_inches='tight')
    
    
    "Boxplot parameters in 2006 and 2096 per season"
    fig = plt.figure(4, figsize = (30,16))
#    fig.suptitle('Evolution of paramater destribution per season',size = 'xx-large')
    LAB = ['Winter 2006','Winter 2096',
           'Spring 2006','Spring 2096',
           'Summer 2006','Summer 2096',
           'Autumn 2006','Autumn 2096']
    for i in range(7):
        ax = fig.add_subplot(4,2,i+1)
        DAT = [Wi[:91,i,st,mdl,exp],Wi[-91:,i,st,mdl,exp],
               Sp[:91,i,st,mdl,exp],Sp[-91:,i,st,mdl,exp],
               Su[:91,i,st,mdl,exp],Su[-91:,i,st,mdl,exp],
               Au[:91,i,st,mdl,exp],Au[-91:,i,st,mdl,exp]]
        plt.title(VARIABLES[i],size = 'large')
        plt.grid(True)
        ax.boxplot(DAT,labels = LAB)
    fig.savefig('dataexpl_param0669_perseason.png', bbox_inches='tight')
    
    
    
    "Boxplot parameters per season"
    fig = plt.figure(5, figsize = (16,16))
#    fig.suptitle('Paramater destribution per season',size = 'xx-large')
    LAB = ['Winter','Spring ','Summer','Autumn']
    for i in range(7):
        ax = fig.add_subplot(4,2,i+1)
        DAT = [Wi[:,i,st,mdl,exp],Sp[:,i,st,mdl,exp],
               Su[:,i,st,mdl,exp],Au[:,i,st,mdl,exp]]
        plt.title(VARIABLES[i],size = 'large')
        plt.grid(True)
        ax.boxplot(DAT,labels = LAB)
    fig.savefig('dataexpl_param_perseason_all.png', bbox_inches='tight')

    exp = 1
    mdl = 3
    "Boxplot parameters in 2006-2016 and 2086-2096"
#    fig = plt.figure(6, figsize = (12,16))
    fig.suptitle('Evolution of paramater destribution from 2006-2016 to 2086-2096')
    for i in range(7):
        ax = fig.add_subplot(4,2,i+1)
        plt.grid(True)
        ax.boxplot([Data[0:3650,i,st,mdl,exp],Data[-3650:,i,st,mdl,exp]],labels = ['06-16', '86-96'])
        plt.title(VARIABLES[i],size = 'large')
    fig.savefig('dataexpl_param10years.png', bbox_inches='tight')
    
    
    exp = 1
    mdl = 3
    "Boxplot parameters in 2006-2016 and 2086-2096 per season"
    fig = plt.figure(7, figsize = (26,26))
#    fig.suptitle('Evolution of paramater destribution per season 2006-2016 to 2086-2096',size = 'xx-large')
    LAB = ['Winter 06-16','Winter 86-96',
           'Spring 06-16','Sprint 86-96',
           'Summer 06-16','Summer 86-96',
           'Autumn 06-16','Autumn 86-96']
    for i in range(7):
        ax = fig.add_subplot(4,2,i+1)
        DAT = [Wi[:910,i,st,mdl,exp],Wi[-910:,i,st,mdl,exp],
               Sp[:910,i,st,mdl,exp],Sp[-910:,i,st,mdl,exp],
               Su[:910,i,st,mdl,exp],Su[-910:,i,st,mdl,exp],
               Au[:910,i,st,mdl,exp],Au[-910:,i,st,mdl,exp]]
        plt.title(VARIABLES[i],size = 'x-large')
        plt.grid(True)
        ax.boxplot(DAT,labels = LAB)
    fig.savefig('dataexpl_param10years_perseason.png', bbox_inches='tight')
    
    

