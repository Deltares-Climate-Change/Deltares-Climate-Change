import numpy as np
import pandas as pd
import xarray as xr

def data_to_xarray(data):
    """
    Takes the (preprocessed) data and returns a labeled xarray
    and imputes NaN values.
    """
    dates= pd.date_range(start="2006-01-01",end="2096-12-31")
    experiments=['rcp45','rcp85']
    variables=["rsds","tas","uas","vas","clt","hurs","ps"]
    stations=['Marsdiep Noord','Doove Balg West',
                    'Vliestroom','Doove Balg Oost',
                    'Blauwe Slenk Oost','Harlingen Voorhaven','Dantziggat',
                    'Zoutkamperlaag Zeegat','Zoutkamperlaag',
                    'Harlingen Havenmond West']
    driving_models=['CNRM-CERFACS-CNRM-CM5','ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-MR','MOHC-HadGEM2-ES','MPI-M-MPI-ESM-LR']

    X = xr.DataArray(data, coords=[dates,variables,stations,driving_models,experiments], dims=['time', 'var', 'station','model','exp'])
    #Fills NaN Values appropriately
    X = X.interpolate_na(dim = 'time', method = 'nearest')

    return X
