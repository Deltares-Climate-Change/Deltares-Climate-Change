# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:55:33 2020

@author: simon
"""
import numpy as np
import matplotlib.pyplot as plt

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

CLUSTERS = [4]


st = 0

for st in range(len(STATIONS)):
    for n_cl in range(len(CLUSTERS)):
    
        ClusRng = list(range(CLUSTERS[n_cl]))
        fig = plt.figure(figsize = (20,15))
        for mdl in range(len(MODELS)):
            Filename = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[0]+'_'+str(CLUSTERS[n_cl])+'_clusters.npy'
            A = np.load(Filename)
            Filename = 'ClusExpVar_'+STATIONS[st]+'_'+MODELS[mdl]+'_'+EXPERIMENTS[1]+'_'+str(CLUSTERS[n_cl])+'_clusters.npy'
            B = np.load(Filename)
            
            for var in range(len(VARIABLES)):
                idxA = np.flip(np.argsort(A[:,2*var]))
                idxB = np.flip(np.argsort(B[:,2*var]))
                
                ax0 = fig.add_subplot(4, 4, 2*var+1)
                ax1 = fig.add_subplot(4, 4, 2*var+2)
                
                #ax0.errorbar(ClusRng,A[:,2*var+1][idxA],yerr= A[:,2*var+2][idxA], fmt='o',label = MODELS[mdl])
                #ax1.errorbar(ClusRng,B[:,2*var+][idxB],yerr= B[:,2*var+2][idxB], fmt='o',label = MODELS[mdl])
                
                ax0.errorbar(ClusRng,A[:,2*var][idxA], fmt='-o',label = MODELS[mdl])
                ax1.errorbar(ClusRng,B[:,2*var][idxB], fmt='-o',label = MODELS[mdl])
    
                ax0.set_title(VARIABLES[var]+' '+EXPERIMENTS[0])
                ax1.set_title(VARIABLES[var]+' '+EXPERIMENTS[1])
        
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels)
                  
        fig.show()
        figname = 'ClusExpVar_Compare_models_with_station_'+STATIONS[st]+'_'+str(CLUSTERS[n_cl])+'_clusters.png'
        fig.savefig(figname, bbox_inches='tight')


