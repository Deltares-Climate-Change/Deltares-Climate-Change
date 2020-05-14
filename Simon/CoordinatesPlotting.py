# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:47:32 2020

@author: simon
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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


coordinates = [[52.9836220,4.7502700],[53.0539960,5.0326430],[53.3144720,5.1603010],[53.0854860,5.2876310],[53.2256720,5.2783220],[53.1742859,5.4059014],[53.4022720,5.7274630],[53.4762460,6.0797620],[53.4305470,6.1331940],[53.1769800,5.3969300]]

C1 = [coordinates[i][0] for i in range(len(coordinates))]
C2 = [coordinates[i][1] for i in range(len(coordinates))]



ax1bis = plt.subplot(111, projection=ccrs.Mercator())


def plotpt(ax, extent=(4.2,6.6,52.8,53.7), **kwargs):
    ax.plot(C2,C1, 'r*', ms=20, **kwargs)
    for i in range(len(coordinates)):
        if i == 9:
            H_al = 'left'
        else:
            H_al = 'right'
        plt.text(coordinates[i][1], coordinates[i][0], STATIONS[i],
                 horizontalalignment=H_al,
                 verticalalignment = 'bottom',
                 transform=ccrs.Geodetic())
    ax.set_extent(extent)
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


plotpt(ax1bis, transform=ccrs.PlateCarree()) # Correct, projection and transform are different!


plt.show()