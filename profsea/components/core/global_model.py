import numpy as np
from typing import Dict

from .state import ClimateState
from .base import SLRComponent

class Global:
    def __init__(
        self,
        components: Dict[str, SLRComponent],
        end_yr: int,
        seed: int = 42,
        nt: int = 100,
        nm: int = 1000,
    ):
        self.components = components
        self.end_yr = end_yr
        self.nt = nt
        self.nm = nm
        
        self.seed_seq = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(self.seed_seq)
        
        # Independent RNGs for each provided component
        child_seeds = self.seed_seq.spawn(len(self.components))
        self.component_rngs = {
            name: np.random.default_rng(s) 
            for name, s in zip(self.components.keys(), child_seeds)
        }

    def run(self, scenario: str, T_change: np.ndarray, OHC_change: np.ndarray):
        T_ens, therm_ens, T_int_ens, T_int_med = self._calculate_drivers(T_change, OHC_change)
        
        # Shared correlation arrays
        fraction = self.rng.random(self.nm * self.nt)
        
        # Add context to shared state
        state = ClimateState(
            scenario=scenario, T_ens=T_ens, T_int_ens=T_int_ens, 
            T_int_med=T_int_med, therm_ens=therm_ens, fraction=fraction,
            nyr=self.end_yr - 2006, nt=self.nt, nm=self.nm
        )

        # Project components!
        results = {}
        for name, comp in self.components.items():
            comp_rng = self.component_rngs[name]
            results[name] = comp.project(state, comp_rng)

        # Calculate total GMSLR
        results["gmslr"] = np.sum(list(results.values()), axis=0)
        return results

    def _calculate_drivers(self, T_change: np.ndarray, OHC_change: np.ndarray) -> tuple:
        """Calculate the drivers of GMSLR: temperature change and
        thermosteric sea level rise.

        Returns
        -------
        T_ens: np.ndarray
            Ensemble of temperature changes.
        therm_ens: np.ndarray
            Ensemble of thermosteric sea level rise.
        T_int_ens: np.ndarray
            Ensemble of time-integral temperature anomalies.
        T_int_med: np.ndarray
            Median of time-integral temperature anomalies.
        """
        # Sensitivity of thermosteric SLR to ocean heat content change
        # From Turner et al. (2023)
        exp_efficiency = (
            self.rng.normal(loc=0.113, scale=0.013, size=self.nt)[:, None] * 1e-24
        )  # m/YJ

        if self.input_ensemble:
            # Check if dimensions are the right way around
            if T_change.shape[1] != self.nyr:
                T_change = T_change.T
            if OHC_change.shape[1] != self.nyr:
                OHC_change = OHC_change.T

            T_med = np.percentile(T_change, 50, axis=0)
            T_std = np.std(T_change, axis=0)

            therm_med = np.percentile(OHC_change, 50, axis=0) * exp_efficiency
            therm_std = np.std(OHC_change * exp_efficiency, axis=0)

        else:
            if self.T_percentile_95 is not None:
                T_med = T_change
                therm_med = OHC_change * exp_efficiency

                T_std = (self.T_percentile_95 - T_change) / 1.645
                therm_std = (
                    (self.OHC_percentile_95 - OHC_change) * exp_efficiency / 1.645
                )

            else:
                raise ValueError(
                    "If input_ensemble is False, and T_change and OHC_change "
                    "are not 2D arrays, you must provide a 95th percentile "
                    "timeseries for T_change and OHC_change. Add this using "
                    "T_percentile_95 and OHC_percentile_95 keyword arguments."
                )

        # Time-integral of temperature anomaly
        T_int_med = np.cumsum(T_med)
        T_int_std = np.cumsum(T_std)

        # Generate a sample of perfectly correlated timeseries fields of temperature,
        # time-integral temperature and expansion, each of them [realisation,time]
        z = self.rng.standard_normal(self.nt) * self.tcv

        # For each quantity, mean + standard deviation * normal random number
        # reshape to [realisation,time]
        T_ens = z[:, np.newaxis] * T_std + T_med
        therm_ens = z[:, np.newaxis] * therm_std + therm_med
        T_int_ens = z[:, np.newaxis] * T_int_std + T_int_med
        return T_ens, therm_ens, T_int_ens, T_int_med