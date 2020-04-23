import numpy as np
from utils.xarray import data_to_xarray
import pandas as pd

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