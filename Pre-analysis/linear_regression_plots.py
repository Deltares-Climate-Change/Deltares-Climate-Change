import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.common import *
import seaborn as sns
from sklearn.linear_model import LinearRegression
import xarray as xr

#Get Data
data = np.load('Datares/tensor_daily_mean_5D.npy')
X = data_to_xarray(data)
X.head()


#Get Yearly Averages
X_yearly = yearly_average(X)

"""#Linear Regression for a selected setting
x = np.array(X_yearly['year']).reshape((-1,1))
y = X_yearly.sel(station = 'Marsdiep Noord',model = 'CNRM-CERFACS-CNRM-CM5', exp = 'rcp85', var = 'tas' )
model = LinearRegression().fit(x,y)
plt.plot(x,model.predict(x))
plt.scatter(x,y)"""


#Took a while but this is smooth
lmdomain45 = X_yearly.sel(var = 'tas', exp = 'rcp45').mean(dim = 'station')
long_format_data45 = lmdomain45.to_pandas().reset_index().melt(id_vars = 'year')

sns.lmplot(x = 'year', y = 'value', col = 'model', hue = 'model', data = long_format_data45,
                col_wrap = 3, height = 5, sharex = False, sharey = False, ci = 99)

lmdomain85 = X_yearly.sel(var = 'tas', exp = 'rcp85').mean(dim = 'station')
long_format_data85 = lmdomain85.to_pandas().reset_index().melt(id_vars = 'year')

sns.lmplot(x = 'year', y = 'value', col = 'model', hue = 'model', data = long_format_data85,
                col_wrap = 3, height = 5, sharex = False, sharey = False, ci = 99)
