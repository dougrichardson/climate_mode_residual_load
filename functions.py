import xarray as xr
import numpy as np
from scipy.stats import spearmanr

def load_monthly(name, convert="TWh", time_slice=slice("1941", None)):
    """
    Load monthly data.
    
    name: str, filename
    convert: str, 'GWh' or 'TWh' divides by 1e3 or 1e6, respectively
    time_slice: slice, times to select.
    """
    if convert == "GWh":
        divisor = 1000
    elif convert == "TWh":
        divisor = 1e6
    else:
        divisor = 1
        
    ds = xr.open_dataset("/g/data/w42/dr6273/work/projects/Aus_energy/monthly_data/" + name + ".nc")
    return ds.sel(time=time_slice) / divisor

# def load_add_mean(file, var, convert="TWh"):
#     """
#     Load monthly detrended data, then add mean of non-detrended and divide by 1000 (MWh to GWh).
    
#     file: str, filepath
#     var: str, name of variable
#     convert: str, 'GWh' or 'TWh' divides by 1e3 or 1e6, respectively
#     """
#     if convert == "GWh":
#         divisor = 1000
#     elif convert == "TWh":
#         divisor = 1e6
#     else:
#         divisor = 1
        
#     ds = load_monthly(file)
#     mean = ds[var].mean("time")
#     return (ds[var + "_detrended"] + mean) / divisor

def sel_month(ds, month):
    """
    Return array for specified month
    
    ds: dataset to select from
    month: int or list of int between 1 and 12, default is None
    """
    if month is None:
        return ds
    elif isinstance(month, int):
        if 1 <= month <= 12:
            return ds.isel(time=ds.time.dt.month == month)
        else:
            raise ValueError("Incorrect month specified.")
    elif isinstance(month, list):
        if all((isinstance(i, int)) & (1 <= i <= 12) for i in month):
            return ds.isel(time=ds.time.dt.month.isin(month))
        else:
            raise ValueError("Incorrect month specified.")

def detrend_dim(da, dim, deg=1):
    """
    Detrend along a single dimension.
    
    da: array to detrend
    dim: dimension along which to detrend
    deg: degree of polynomial to fit (1 for linear fit)
    
    Adapted from the original code here:
    Author: Ryan Abernathy
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def normalise(ds, groupby=None):
    """
    Return values with mean subtracted and divided by standard deviation.
    
    ds: dataset with 'time' dimension
    groupby: None, or str in form e.g. 'time.month'
    """
    if groupby is not None:
        return ds.groupby(groupby).apply(lambda x: (x - x.mean("time")) / x.std("time"))
    else:
        return (ds - ds.mean("time")) / ds.std("time")

def calc_contribution(ds, regions):
    """
    Percentage contribution of each region to the sum of all regions
    
    ds: dataset
    regions: list, regions to select from ds
    """
    cont = ds.sel(region=regions) / ds.sel(region=regions).sum("region") * 100
    return cont.rename("shortfall_contribution")

def xr_spearmanr(ds1, ds2):
    """
    xarray wrapper for scipt.stats.spearmanr
    """
    def _spearman(x, y):
        res = spearmanr(x, y, nan_policy='omit')
        return res.correlation, res.pvalue

    rho, pval = xr.apply_ufunc(
        _spearman,
        ds1,
        ds2,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],
        output_dtypes=[float, float],
        vectorize=True,
    )
    
    return rho, pval

def fdr(p_values_da, alpha=0.1):
    """
    Calculates significance on a DataArray of gridded p-values (p_values_da)
    by controlling the false discovery rate.
    Returns a DataArray of ones (significant) and zeros (not significant).
    
    p_values_da: array of p-values
    alpha: significance level (often double the alpha in standard hypothesis test)
    """
    p_1d = p_values_da.values.reshape(-1) # 1-D array of p-values
    p_1d = p_1d[~np.isnan(p_1d)] # Remove NaNs
    
    sorted_pvals = np.sort(p_1d) # sort p-values
    N = len(sorted_pvals) # sample size
    
    fdr_criteria = alpha * (np.arange(1, N+1) / N) # the diagonal line of criteria
    pvals_less_than_fdr_criteria = np.where(sorted_pvals < fdr_criteria)[0]
    
    if len(pvals_less_than_fdr_criteria) > 0: #if any p-values satisfy the FDR criteria
        # index of the largest p-value still under the fdr_criteria line.
        largest_p_less_than_criteria = pvals_less_than_fdr_criteria[-1]
        # the p-value for controlling the FDR
        p_fdr = sorted_pvals[largest_p_less_than_criteria] 
    else:
        p_fdr = -1 # abritrary number < 0. Ensures no significant results.
    
    # massage data into binary indicators of FDR significance
    keep_signif = p_values_da.where(p_values_da <= p_fdr, -999)
    signif_da = keep_signif.where(keep_signif == -999, 1)
    signif_da = signif_da.where(signif_da == 1, 0)
    
    return signif_da.where(p_values_da.notnull(), np.nan) #, sorted_pvals, fdr_criteria