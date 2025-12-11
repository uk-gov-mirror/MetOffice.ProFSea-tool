import dask.array as da
import numpy as np
from scipy.spatial.distance import cdist
import xarray as xr

def sample_members_2D(array: np.ndarray, percentiles: list|np.ndarray) -> np.ndarray:
    """Sample real ensemble members from a 2D numpy array."""
    # Caculate statistical timeseries, then match with closest real timeseries 
    array_percentiles = np.percentile(array, percentiles, axis=0)
    distances = cdist(array_percentiles, array)
    mem_indices = np.argmin(distances, axis=1)
    return array[mem_indices]


def interpolate(data: da.array, lats: int, lons: int) -> np.ndarray:
    """
    """
    original_da = xr.DataArray(
        data.data,
        coords=[
            ("lat", data[data.dims[0]].values), 
            ("lon", data[data.dims[1]].values)
        ],
        name="v")

    target_lat = np.linspace(90, -90, lats) + 0.5
    target_lon = np.linspace(-180, 180, lons, endpoint=False) + 0.5
    data_interp = original_da.interp(
        lat=target_lat, lon=target_lon, method="linear").data

    data_interp = da.roll(data_interp, 180, axis=1)
    return data_interp
