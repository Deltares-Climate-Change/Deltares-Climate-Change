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


def DivideInSeasons(X):
    """
    Takes the whole dataset and return one xarrays for each season.
    """
    Winter = X.sel(time=pd.date_range(start="2006-01-01",end="2006-03-20"))
    Spring = X.sel(time=pd.date_range(start="2006-03-21",end="2006-06-20"))
    Summer = X.sel(time=pd.date_range(start="2006-06-21",end="2006-09-20"))
    Autumn = X.sel(time=pd.date_range(start="2006-09-21",end="2006-12-20"))
    for year in range(2007,2097):
        Winter = xr.concat([Winter,X.sel(time=pd.date_range(start=(str(year-1)+"-12-21"),end=(str(year)+"-03-20")))], dim='time')
        Spring = xr.concat([Spring,X.sel(time=pd.date_range(start=(str(year)+"-03-21"),end=(str(year)+"-06-20")))], dim='time')
        Summer = xr.concat([Summer,X.sel(time=pd.date_range(start=(str(year)+"-06-21"),end=(str(year)+"-09-20")))], dim='time')
        Autumn = xr.concat([Autumn,X.sel(time=pd.date_range(start=(str(year)+"-09-21"),end=(str(year)+"-12-20")))], dim='time')
    return Winter,Spring,Summer,Autumn


def YearlyAverage(Data):
    """
    Takes the whole dataset as xarray and returns yearly averages for all variables.
    """
    yearbeginnings = pd.date_range(start = "2006-01-01", periods = 91, freq = pd.offsets.YearBegin(1))
    yearends =  pd.date_range(start = "2006-01-01", periods = 91, freq = 'Y')
    average_xr = Data.sel(time = pd.date_range(start = yearbeginnings[0], end = yearends[0])).mean(dim = 'time')
    for i in range(1,len(yearbeginnings)):
        average_xr = xr.concat([average_xr,Data.sel(time = pd.date_range(start = yearbeginnings[i], end = yearends[i])).mean(dim = 'time')], dim = 'year')

    average_xr['year'] = np.arange(2006,2097)
    return average_xr
