# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
data = np.load('/Users/ceciliacasolo/Desktop/Data_CC/tensor_daily_mean_5D.npy')

#TIME SERIES PLOT OF VARIABLE COMPARED AMONG THE DIFFERENT STATIONS
fig, ax = plt.subplots(10, sharey=True)
fig.suptitle('Near surface Air temperature per station')
ax[0].plot(data[:,1,0,1,1])
ax[1].plot(data[:,1,1,1,1])
ax[2].plot(data[:,1,2,1,1])
ax[3].plot(data[:,1,3,1,1])
ax[4].plot(data[:,1,4,1,1])
ax[5].plot(data[:,1,5,1,1])
ax[6].plot(data[:,1,6,1,1])
ax[7].plot(data[:,1,7,1,1])
ax[8].plot(data[:,1,8,1,1])
ax[9].plot(data[:,1,9,1,1])

fig, ax = plt.subplots(10, sharey=True)
fig.suptitle('Surface Downwelling Shortwave Radiation per station')
ax[0].plot(data[:,0,0,1,1])
ax[1].plot(data[:,0,1,1,1])
ax[2].plot(data[:,0,2,1,1])
ax[3].plot(data[:,0,3,1,1])
ax[4].plot(data[:,0,4,1,1])
ax[5].plot(data[:,0,5,1,1])
ax[6].plot(data[:,0,6,1,1])
ax[7].plot(data[:,0,7,1,1])
ax[8].plot(data[:,0,8,1,1])
ax[9].plot(data[:,0,9,1,1])

fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Eastward Near-Surface Wind in Stations 1-5')
ax[0].plot(data[:,2,0,1,1])
ax[1].plot(data[:,2,1,1,1])
ax[2].plot(data[:,2,2,1,1])
ax[3].plot(data[:,2,3,1,1])
ax[4].plot(data[:,2,4,1,1])

fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Eastward Near-Surface Wind in Stations 6-10')
ax[0].plot(data[:,2,5,1,1])
ax[1].plot(data[:,2,6,1,1])
ax[2].plot(data[:,2,7,1,1])
ax[3].plot(data[:,2,8,1,1])
ax[4].plot(data[:,2,9,1,1])

fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Northward Near-Surface Wind in Stations 1-5')
ax[0].plot(data[:,3,0,1,1])
ax[1].plot(data[:,3,1,1,1])
ax[2].plot(data[:,3,2,1,1])
ax[3].plot(data[:,3,3,1,1])
ax[4].plot(data[:,3,4,1,1])

fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Northward Near-Surface Wind in Stations 6-10')
ax[0].plot(data[:,3,5,1,1])
ax[1].plot(data[:,3,6,1,1])
ax[2].plot(data[:,3,7,1,1])
ax[3].plot(data[:,3,8,1,1])
ax[4].plot(data[:,3,9,1,1])


fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Total Cloud Cover in Stations 1-5')
ax[0].plot(data[:,4,0,1,1])
ax[1].plot(data[:,4,1,1,1])
ax[2].plot(data[:,4,2,1,1])
ax[3].plot(data[:,4,3,1,1])
ax[4].plot(data[:,4,4,1,1])

fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Total Cloud Cover in Stations 6-10')
ax[0].plot(data[:,4,5,1,1])
ax[1].plot(data[:,4,6,1,1])
ax[2].plot(data[:,4,7,1,1])
ax[3].plot(data[:,4,8,1,1])
ax[4].plot(data[:,4,9,1,1])

fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Near-Surface Relative Humidity in Stations 1-5')
ax[0].plot(data[:,5,0,1,1])
ax[1].plot(data[:,5,1,1,1])
ax[2].plot(data[:,5,2,1,1])
ax[3].plot(data[:,5,3,1,1])
ax[4].plot(data[:,5,4,1,1])

fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Near-Surface Relative Humidity in Stations 6-10')
ax[0].plot(data[:,5,5,1,1])
ax[1].plot(data[:,5,6,1,1])
ax[2].plot(data[:,5,7,1,1])
ax[3].plot(data[:,5,8,1,1])
ax[4].plot(data[:,5,9,1,1])

fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Surface Pressure in Stations 1-5')
ax[0].plot(data[:,6,0,1,1])
ax[1].plot(data[:,6,1,1,1])
ax[2].plot(data[:,6,2,1,1])
ax[3].plot(data[:,6,3,1,1])
ax[4].plot(data[:,6,4,1,1])

fig, ax = plt.subplots(5, sharey=True)
fig.suptitle('Surface Pressure in Stations 6-10')
ax[0].plot(data[:,6,5,1,1])
ax[1].plot(data[:,6,6,1,1])
ax[2].plot(data[:,6,7,1,1])
ax[3].plot(data[:,6,8,1,1])
ax[4].plot(data[:,6,9,1,1])

#TIME SERIES PLOT OF Eastward Near-Surface Wind AND Northward Near-Surface Wind
#IN THE SAME STATION
fig, ax = plt.subplots(2, sharey=True)
fig.suptitle('Eastward VS Northward Near-Surface Wind in Marsdiep Noord')
ax[0].plot(data[:,2,0,1,1])
ax[1].plot(data[:,3,0,1,1])

fig, ax = plt.subplots(2, sharey=True)
fig.suptitle('Eastward VS Northward Near-Surface Wind in Doove Balg West')
ax[0].plot(data[:,2,1,1,1])
ax[1].plot(data[:,3,1,1,1])

fig, ax = plt.subplots(2, sharey=True)
fig.suptitle('Eastward VS Northward Near-Surface Wind in Zoutkamperlaag')
ax[0].plot(data[:,2,9,1,1])
ax[1].plot(data[:,3,9,1,1])

#TIME SERIES PLOT OF THE DIFFEREN VARIABLES IN THE SAME STATION
fig, ax = plt.subplots(7)
fig.suptitle('Different variables in Marsdiep Noord')
ax[0].plot(data[:,0,0,1,1])
ax[1].plot(data[:,1,0,1,1])
ax[2].plot(data[:,2,0,1,1])
ax[3].plot(data[:,3,0,1,1])
ax[4].plot(data[:,4,0,1,1])
ax[5].plot(data[:,5,0,1,1])
ax[6].plot(data[:,6,0,1,1])






