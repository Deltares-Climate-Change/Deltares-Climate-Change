import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.xarray import data_to_xarray
import seaborn as sns

data = np.load('Datares/tensor_daily_mean_5D.npy')
X = data_to_xarray(data)
X.head()

#Managed to do it in one line
ax = sns.heatmap(np.corrcoef(X.sel(station='Harlingen Havenmond West',exp='rcp45',model='CNRM-CERFACS-CNRM-CM5'), rowvar = False),annot=True)
