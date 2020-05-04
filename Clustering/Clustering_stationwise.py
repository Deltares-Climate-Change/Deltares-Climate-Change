import datetime
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

Data = np.load('../Datares/tensor_daily_mean_5D.npy')
NanINDX = np.argwhere(np.isnan(Data))
for i in range(len(NanINDX)):
    Data[NanINDX[i][0],NanINDX[i][1],NanINDX[i][2],NanINDX[i][3],NanINDX[i][4]] = 200

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

SubData = Data[:,:,:,mdl,exp]
AllData = SubData[:,:,0]
for i in range(1,10):
    AllData = np.vstack((AllData,SubData[:,:,i]))
    
n_clusters = 7
clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(AllData)
cluster_labels = clusterer.labels_

n = len(SubData)

def StationCounter(Labels,n_clusters,AllData):
    Dat = [[] for i in range(n_clusters)]
    for station in range(10):
        for i in range(len(Data)):
            indx = n*station + i
            lab = Labels[indx]
            Dat[lab].append(station)       
    return(Dat)

test = StationCounter(cluster_labels,n_clusters,AllData)
fig1, axes = plt.subplots(nrows=2, ncols=1)
ax0, ax1 = axes.flatten()
ax0.hist(test, 10, density=True, histtype='bar')
ax0.set_title('Divide in months')

