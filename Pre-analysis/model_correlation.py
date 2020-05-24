import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.common import data_to_xarray
import seaborn as sns
from scipy.stats import spearmanr

data = np.load('Datares/tensor_daily_mean_5D.npy')
X = data_to_xarray(data)

models = np.array(X.coords['model'])
stations = np.array(X.coords['station'])
variables = np.array(X.coords['var'])


fig,ax = plt.subplots(len(variables),1, figsize=(15,len(variables)*2), sharex = True)

for (i,var) in enumerate(variables):
    cmatrix = np.zeros((len(models),len(models)))
    for station in stations:
        cmatrix += spearmanr(X.sel(var = var, exp = 'rcp45', station = station))[0]
    cmatrix = cmatrix/len(stations)
    plt.sca(ax[i])
    #print(var)
    ax[i].set_title(var)
    sns.heatmap(cmatrix, annot=True, xticklabels = models, yticklabels = models)

plt.savefig('modelcorrelation.pdf')

#x1 = X.sel(var = 'ps', exp = 'rcp45', station = stations[0], model = models[0])
#x2 = X.sel(var = 'ps', exp = 'rcp45', station = stations[0], model = models[1])

#pd.DataFrame.ewm(x1.to_pandas(), alpha = 0.005).mean().plot()
#pd.DataFrame.ewm(x2.to_pandas(), alpha = 0.005).mean().plot()


sel = 2


fig,ax = plt.subplots(1, figsize = (10,10))

cmatrix = np.zeros((len(models),len(models)))
for station in stations:
    cmatrix += spearmanr(X.sel(var = variables[sel], exp = 'rcp45', station = station))[0]
cmatrix = cmatrix/len(stations)
#print(var)
ax.set_title(variables[sel])
sns.heatmap(cmatrix, annot=True, xticklabels = models, yticklabels = models)
plt.tight_layout()


plt.savefig('modelcorrelation2')

len(variables)
