import functools

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState

@functools.lru_cache(maxsize=1)
def load_greenland_calibration():
    """Loads the CSV once and keeps it in memory."""
    # path = Path(__file__).parent / "aux_data" / "ISMIP_GIS_calibration.csv"
    # Path is actually relative to the project root, not the component file
    path = Path(__file__).parent.parent / "aux_data" / "ISMIP_GIS_calibration.csv"
    return pd.read_csv(path)


class GreenlandAR6(Component):
    def __init__(self):
        self.df = load_greenland_calibration()

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Project Greenland ice-sheet contribution to GMSLR.
        This follows the IPCC AR6 methodology as closely as possible.
        Projections are relative to 1996-2014 baseline.

        Returns
        -------
        np.ndarray
            Total GIS contribution to GMSLR.
        """
        df = self.df
        b0 = df["b0"].values[None, :, None]
        b1 = df["b1"].values[None, :, None]
        b2 = df["b2"].values[None, :, None]
        b3 = df["b3"].values[None, :, None]
        b4 = df["b4"].values[None, :, None]
        b5 = df["b5"].values[None, :, None]
        sigma = df["sigma"].values
        time_delta = np.arange(state.nyr)

        # GIS trend values taken from FACTS GitHub repo
        trend_mean = 0.19
        trend_std = 0.1

        # Calculate trend contribution distribution
        trend = truncnorm.ppf(
            rng.random(state.nm), a=0.0, b=99999.9, loc=trend_mean, scale=trend_std
        )
        trend = trend[:, None] * time_delta[None, :]
        trend = trend[:, None, :]
        trend /= 1e3  # convert mm to m SLE

        # Calculate GIS contribution rate
        dsle = (
            b0
            + (b1 * state.T_ens[:, None, :])
            + (b2 * state.T_ens[:, None, :] ** 2)
            + (b3 * state.T_ens[:, None, :] ** 3)
            + (b4 * time_delta[None, None, :])
            + (b5 * time_delta[None, None, :] ** 2)
        )

        # Now integrate
        sle = np.cumsum(dsle, axis=2)  # mm SLE per K of global warming
        sle = sle * 1e-3  # convert mm to m SLE

        # Make a Monte Carlo ensemble of projections for each model in the calibration
        sle_ens = np.zeros((state.nm, state.nt, state.nyr))
        r_per_model = state.nm // sle.shape[1]
        r_remainder = state.nm % sle.shape[1]

        # We want to distribute the remainder evenly across the models
        unc = rng.normal(scale=sigma)
        current_ensemble_idx = 0
        for i in range(sle.shape[1]):
            num_reals_for_model_i = r_per_model + 1 if i < r_remainder else r_per_model
            ifirst = current_ensemble_idx
            ilast = current_ensemble_idx + num_reals_for_model_i
            model_term = sle[:, i, :]  # Shape (nt, nyr)
            uncertainty_term = (
                model_term * unc[None, i, None]
            )  # Shape (num_reals_for_model_i, nt, nyr_param)

            sle_ens[ifirst:ilast, :, :] = model_term[None, :, :] + uncertainty_term
            current_ensemble_idx = ilast

        sle_ens += trend

        # Persist 2100 rate of change
        if(state.end_yr >= 2100):
            rate = np.diff(sle_ens, axis=2)[:, :, 94]
            sle_ens[:, :, 95:] = sle_ens[:, :, 94:95] + (
                rate[:, :, None] * time_delta[None, None, 1 : state.nyr - 94]
            )
        
        sle_ens = sle_ens.reshape((state.nm * state.nt, state.nyr))
        return sle_ens
