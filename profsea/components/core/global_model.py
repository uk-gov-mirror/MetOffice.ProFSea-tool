import concurrent.futures
from pathlib import Path
import os
from typing import Dict

import numpy as np
from rich.console import Console
import xarray as xr

from .state import ClimateState
from .base import Component
from profsea.utils import sample_members_2D

console = Console()


class Global:
    def __init__(
        self,
        components: Dict[str, Component],
        end_yr: int,
        nt: int = 100,
        nm: int = 1000,
        tcv: float = 1.0,
        parallel: bool = True,
        input_ensemble: bool = True,
        output_percentiles: list | np.ndarray = None,
        random_sample: bool = False,
    ):
        self.components = components
        self.end_yr = end_yr
        self.nt = nt
        self.nm = nm
        self.tcv = tcv
        self.parallel = parallel
        self.input_ensemble = input_ensemble
        self.output_percentiles = output_percentiles
        self.random_sample = random_sample

        self.endofhistory = 2006
        self.endofAR5 = 2100
        self.nyr = self.end_yr - self.endofhistory

    def _check_shapes(
        self, T_change: np.ndarray, OHC_change: np.ndarray, n_time: int
    ) -> None:
        """Check that the input arrays have the correct shape.

        Parameters
        ----------
        T_change: np.ndarray
            Array of surface temperature changes.
        OHC_change: np.ndarray
            Array of ocean heat content changes.
        n_time: int
            Expected number of time steps.

        Returns
        -------
        None
        """
        if T_change.ndim == 1:
            T_change = T_change[np.newaxis, :]
        if OHC_change.ndim == 1:
            OHC_change = OHC_change[np.newaxis, :]

        if T_change.shape[1] != n_time:
            # Split over lines for readability
            raise ValueError(
                f"T_change should have shape (realisation, time) with time \
                dimension of length {n_time}. Got {T_change.shape}."
            )
        if OHC_change.shape[1] != n_time:
            raise ValueError(
                f"OHC_change should have shape (realisation, time) with time \
                dimension of length {n_time}. Got {OHC_change.shape}."
            )

    def run(
        self,
        scenario: str,
        T_change: np.ndarray,
        OHC_change: np.ndarray,
        member_seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """Run the emulator to project GMSLR components for a specific state."""
        seed_seq = np.random.SeedSequence(member_seed)
        run_rng = np.random.default_rng(seed_seq)

        self._check_shapes(T_change, OHC_change, self.nyr)

        if self.input_ensemble:
            self.nt = T_change.shape[0]

        T_ens, therm_ens, T_int_ens, T_int_med = self._calculate_drivers(
            T_change, OHC_change, run_rng
        )

        # Shared physical correlation state
        fraction = run_rng.random(self.nm * self.nt)

        state = ClimateState(
            scenario=scenario,
            T_ens=T_ens,
            T_int_ens=T_int_ens,
            T_int_med=T_int_med,
            therm_ens=therm_ens,
            fraction=fraction,
            endofAR5=self.endofAR5,
            endofhistory=self.endofhistory,
            end_yr=self.end_yr,
            nyr=self.nyr,
            nt=self.nt,
            nm=self.nm,
        )

        # Child RNGs for each component
        child_seeds = seed_seq.spawn(len(self.components))
        comp_rngs = {
            name: np.random.default_rng(s)
            for name, s in zip(self.components.keys(), child_seeds)
        }

        results = {}
        if self.parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(comp.project, state, comp_rngs[name]): name
                    for name, comp in self.components.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    comp_name = futures[future]
                    try:
                        results[comp_name] = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Component '{comp_name}' failed.") from e
        else:
            for name, comp in self.components.items():
                results[name] = comp.project(state, comp_rngs[name])

        # Random Sampling
        if self.random_sample:
            random_idx = run_rng.integers(low=0, high=self.nm)
            for comp_name, data in results.items():
                if data.ndim > 1:
                    results[comp_name] = data[random_idx][None, :]

        # Output percentiles
        if self.output_percentiles is not None:
            console.log(
                f"Sampling {len(self.output_percentiles)} members per component..."
            )
            for comp_name, data in results.items():
                results[comp_name] = sample_members_2D(data, self.output_percentiles)

        self.results = results

        return results

    def sum_components(self, components: Dict[str, np.ndarray]) -> np.ndarray:
        """Sum the components to get total GMSLR."""
        components["gmslr"] = np.sum(list(components.values()), axis=0)
        return components["gmslr"]

    def save_components(
        self, components: Dict[str, np.ndarray], output_dir: str, scenario_name: str
    ) -> None:
        """Save all SLR components as .npy files to a directory.

        Parameters
        ----------
        output_directory: str
            Directory to save components to.
        scenario_name: str
            Name of the scenario you've run the emulator for.

        Returns
        -------
        None
        """
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save data in netcdf format
        ds = xr.Dataset()
        member_dim = "percentile" if self.output_percentiles is not None else "member"
        for name, component in components.items():
            xr_dataArray = xr.DataArray(
                component,
                dims=[member_dim, "time"],
                coords={
                    member_dim: self.output_percentiles
                    if self.output_percentiles is not None
                    else np.arange(
                        component.shape[0]
                    ),  # handle if no output percentiles
                    "time": np.arange(2006, component.shape[1] + 2006),
                },
            )
            xr_dataArray.attrs["units"] = "m"
            ds[name] = xr_dataArray
        ds.to_netcdf(os.path.join(output_dir, f"{scenario_name}_global.nc"))

    def _calculate_drivers(
        self, T_change: np.ndarray, OHC_change: np.ndarray, rng: np.random.Generator
    ) -> tuple:
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
            rng.normal(loc=0.113, scale=0.013, size=self.nt)[:, None] * 1e-24
        )  # m/YJ

        if self.input_ensemble:
            T_med = sample_members_2D(T_change, [50])
            T_std = np.std(T_change, axis=0)

            therm_med = sample_members_2D(OHC_change, [50]) * exp_efficiency
            therm_std = np.std(OHC_change, axis=0) * exp_efficiency

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
        z = rng.standard_normal(self.nt) * self.tcv

        # For each quantity, mean + standard deviation * normal random number
        # reshape to [realisation,time]
        T_ens = z[:, np.newaxis] * T_std + T_med
        therm_ens = z[:, np.newaxis] * therm_std + therm_med
        T_int_ens = z[:, np.newaxis] * T_int_std + T_int_med
        return T_ens, therm_ens, T_int_ens, T_int_med
