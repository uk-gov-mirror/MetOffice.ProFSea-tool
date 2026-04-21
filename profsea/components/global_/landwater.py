import functools

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import truncnorm

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState

@functools.lru_cache(maxsize=1)
def load_landwater_projection():
    """Loads the NetCDF once and keeps the VALUES in memory."""
    path = Path(__file__).parent.parent/ "aux_data" / "ssp_global_landwater_projections.nc"
    with xr.open_dataset(path) as ds:
        ds.load()
    return ds

class LandwaterAR6(Component):
    def __init__(self):
        self.lw_ds = load_landwater_projection()

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Project land water storage contribution to GMSLR.
        This follows the IPCC AR6 methodology as closely as possible.
        Projections are relative to 1996-2014 baseline.

        Returns
        -------
        np.ndarray
            Land water storage contribution to GMSLR.
        """

        lw_ds = self.lw_ds

        # Interpolate to annual projections
        interp_ds = lw_ds.interp(
            years=np.arange(2005, state.end_yr+1, 1), method="linear"
        ).squeeze()
        lw = interp_ds["sea_level_change"].values * 1e-3  # mm to m

        del interp_ds

        # Make a Monte Carlo ensemble of projections
        full_repeats = (state.nt * state.nm) // lw.shape[0]
        remainder = (state.nt * state.nm) % lw.shape[0]
        lw = np.vstack([np.tile(lw, (full_repeats, 1)), lw[:remainder]])
        lw = lw.reshape(state.nt * state.nm, lw.shape[1])
        lw = lw[:, 1:state.nyr+1]  # Start at 2006, end at end_yr
        
        return lw

    
