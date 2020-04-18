import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.xarray import data_to_xarray
import seaborn as sns
from scipy.stats import spearmanr

data = np.load('Datares/tensor_daily_mean_5D.npy')
X = data_to_xarray(data)
X.head()

variables = ["rsds","tas","uas","vas","clt","hurs","ps"]
stations=['Marsdiep Noord','Doove Balg West',
                'Vliestroom','Doove Balg Oost',
                'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                'Harlingen Havenmond West']
driving_models=['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']


corr45 = np.zeros((7,7))
corr85 = np.zeros((7,7))
corr45r = np.zeros((7,7))
corr85r = np.zeros((7,7))

for station in stations:
    for model in driving_models:
        corr45 += np.corrcoef(X.sel(station= station ,exp= 'rcp45', model= model),rowvar = False)
        corr85 += np.corrcoef(X.sel(station= station ,exp= 'rcp85', model= model),rowvar = False)
        corr45r += spearmanr(X.sel(station= station ,exp= 'rcp45', model= model))[0]
        corr85r += spearmanr(X.sel(station= station ,exp= 'rcp85', model= model))[0]

corr45 = corr45/(len(stations)*len(driving_models))
corr85 = corr85/(len(stations)*len(driving_models))
corr45r = corr45r/(len(stations)*len(driving_models))
corr85r = corr85r/(len(stations)*len(driving_models))



fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (20,7))
sns.heatmap(corr45,annot=True, xticklabels = variables, yticklabels = variables, ax = ax1)
sns.heatmap(corr85,annot=True, xticklabels = variables, yticklabels = variables, ax = ax2)


fig.suptitle('Average product moment correlation over stations and models.')
ax1.set_title('rcp45')
ax2.set_title('rcp85')
plt.show()


fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (20,7))
sns.heatmap(corr45r,annot=True, xticklabels = variables, yticklabels = variables, ax = ax1)
sns.heatmap(corr85r,annot=True, xticklabels = variables, yticklabels = variables, ax = ax2)

fig.suptitle('Average rank correlation over stations and models.')
ax1.set_title('rcp45')
ax2.set_title('rcp85')
plt.show()
