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
X_yearly = YearlyAverage(X)

#Linaer Regression for a selected setting
x = np.array(X_yearly['year']).reshape((-1,1))
y = X_yearly.sel(station = 'Marsdiep Noord',model = 'CNRM-CERFACS-CNRM-CM5', exp = 'rcp85', var = 'tas' )

model = LinearRegression().fit(x,y)

plt.plot(x,model.predict(x))
plt.scatter(x,y)
