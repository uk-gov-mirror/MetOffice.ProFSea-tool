"""
Copyright (c) 2023, Met Office
All rights reserved.
"""
import pickle
import os
from pathlib import Path
import warnings

import dask.array as da
import numpy as np
from rich.console import Console
from rich.progress import track
import xarray as xr

from profsea.utils import interpolate

console = Console()
warnings.filterwarnings("ignore")

class Spatial:
    """ Spatial sea level rise component emulator.

    Parameters
    ----------
    scenario: str
        Name of the scenario.
    expansion_patterns_dir: str
        Direcotry path of regression patterns of thermal expansion component from cmip models.
    fingerprint_dir: str
        Direcotry path of GRD (Gravitational, Rotational, Deformational) fingerprint data
    gia_dir: str
        Direcotry path of GIA (Glacial Isostatic Adjustment) data
    components_dir: str
        Path to global sea-level rise components (output of ProFSea's gmslr module)
    component_list: list
        Namelist of components for spatial projections
    end_year: int
        End year of the projections.
    output_percentiles: int
        List of percentiles for output
    output_dir: str
        Path to output directory for saving spatial projections.
    random_seed: bool
        Seed for numpy.random.
    """

    def __init__(
            self, 
            scenario: str,
            expansion_patterns_dir: str|Path,
            fingerprint_dir: str|Path,
            gia_dir: str|Path,
            components_dir: str|Path=None, 
            component_list: list=None, 
            end_year: int=2301, 
            baseline_yrs: tuple=(1986, 2005),
            output_percentiles: list|np.ndarray=[5, 17, 50, 83, 95],
            output_dir: str|Path=None,
            cmip5_patterns: bool=False,
            random_seed: int=None
        ):

        # Start off with some error handling
        if not components_dir:
            raise ValueError(
                "Provide an input directory")

        if not component_list:
            raise ValueError(
                "Provide namelist of global projection components. "
                "Currently allowed components are expansion, antdyn, "
                "antsmb, wais, eais, glacier, greenland and landwater")

        self.components = {}
        component_path = Path(components_dir)
        ds_component = xr.load_dataset(component_path).sel(scenario=scenario)
        for comp in component_list:
            self.components[comp] = ds_component[comp].data
  
        if not output_dir:
            output_dir = Path.cwd()

        # Assign attributes
        self.scenario = scenario
        self.expansion_patterns_dir = expansion_patterns_dir
        self.fingerprint_dir = fingerprint_dir
        self.gia_dir = gia_dir
        self.components_dir = components_dir
        self.end_year = end_year
        self.baseline_yrs = baseline_yrs
        self.output_percentiles = output_percentiles
        self.output_dir = output_dir
        self.start_year = 2006
        self.n_years = self.end_year - self.start_year
        self.n_samples = self.components["expansion"].shape[0]
        self.rng = np.random.default_rng(seed=random_seed)

        # Get lat/lon sizes
        sample_pattern_filepath = Path(expansion_patterns_dir) \
            / "ACCESS-CM2/zos_regression_ssp245_ACCESS-CM2.npy"
        try:
            sample_pattern = np.load(sample_pattern_filepath, mmap_mode="r")
        except FileNotFoundError:
            raise FileNotFoundError(
                "expansion_patterns_dir expects the " 
                "patterns' parent directory")
        self.nlat, self.nlon = sample_pattern.shape[0], sample_pattern.shape[1]
        console.log("INFO: Spatial() expects sterodynamic patterns on half-integer grids:\n"
                    "\tlat: (-89.5, ..., 89.5)\n"
                    "\tlon: (-179.5, ..., 179.5)")
        console.log(f"Baseline period = {self.baseline_yrs[0]} to {self.baseline_yrs[1]}")

    def _calc_baseline_period(self) -> float:
        """
        Baseline years used for IPCC AR5 and Palmer et al 2020 -- 1986-2005
        :param yrs: years of the projections
        :return: baseline years
        """
        midyr = (self.baseline_yrs[1] - self.baseline_yrs[0] + 1) * 0.5 + self.baseline_yrs[0]
        return self.start_year - midyr

    def _calc_gia_contribution(self) -> None:
        """
        Calculate the glacial isostatic adjustment (GIA) contribution to the
        regional component of sea level rise.
        Option to use the generic global GIA estimates or use the GIA estimates
        developed for UK as part of UKCP18.
        :return: GIA estimates converted to mm/yr
        """
        console.log('Calculating GIA contribution...')
        nGIA, GIA_vals = self._read_gia_estimates()
        Tdelta = self._calc_baseline_period()
        num_percs = len(self.output_percentiles)

        # Unit series of mm/yr expressed as m/yr
        unit_series = (np.arange(self.n_years) + Tdelta) * 0.001
        GIA_unit_series = np.ones([num_percs, self.n_years]) * unit_series

        # rgiai is an array of random GIA indices the size of the sample years
        rgiai = self.rng.integers(nGIA, size=num_percs)

        GIA_T = da.from_array(GIA_unit_series)
        GIA_vals = da.from_array(GIA_vals)
        GIA_series = GIA_T[:, :, None, None] * GIA_vals[rgiai, None, :, :]

        # Save data in netcdf format (Assuming first dimension is percentile, but can be more general percentile/ensemble)
        xr_dataArray = xr.DataArray(
            GIA_series, 
            dims=["samples", "time", "lat", "lon"], 
            coords={
                "samples": np.arange(num_percs),
                "time": np.arange(self.start_year, self.end_year),
                "lat": np.arange(-90, 90) + 0.5, 
                "lon": np.arange(0, 360) + 0.5})
        xr_dataArray.attrs["units"] = "m"
        xr_dataArray.attrs["long_name"] = "Regional GIA sea-level projections"
        ds = xr_dataArray.to_dataset(name='gia')

        ds.attrs["source"] = "ProFSea-Climate v0.1"

        file_header = f"gia_{self.scenario}_projection_{self.end_year}"
        R_file = '_'.join([file_header, 'regional']) + '.nc'
        encoding = {'gia': {"zlib": True, "complevel": 5, "dtype": "float32"}}
        ds.to_netcdf(os.path.join(self.output_dir, R_file), encoding=encoding, compute=True)
        del GIA_series

    def project(self) -> None:
        """
        Calculates global and regional component part contributions to sea level
        change.
        :param mcdir: location of Monte Carlo time series for new projections
        :param components: sea level components
        :param scenario: emission scenario
        :param yrs: years of the projections
        :param array_dims: Array of nesm, nsmps and nyrs
            nesm --> Number of ensemble members in time series
            nsmps --> Determine the number of samples you wish to make
            nyrs --> Number of years in each projection time series
        :return: montecarlo_G (global contribution to sea level rise) and
            montecarlo_R (regional contribution to sea level change)
        """  
        console.log(f"Running with {self.n_samples} ensemble members")

        resamples = self.rng.choice(self.n_samples, size=self.n_samples)  # Preserve correlations across comps

        # Calculate GIA contribution and save it out
        self._calc_gia_contribution()
        nFPs, FPlist = self._load_fingerprints()
        rfpi = self.rng.integers(nFPs, size=self.n_samples)
        for comp in track(list(self.components.keys()), description="Calculating components..."):
            montecarlo_R = da.zeros(
                (self.n_samples, self.n_years, self.nlat, self.nlon),
                dtype=np.float32)  # (FPs applied) + GIA
            montecarlo_G = da.zeros(
                (self.n_samples, self.n_years, self.nlat, self.nlon),
                dtype=np.float32)  # (no FPs applied)

            # Load global projections in for the component
            mc_timeseries = self.components[comp] # shape (mem, time)
            sampled_mc = mc_timeseries[resamples, :self.n_years]
            montecarlo_G[:, :] = da.from_array(sampled_mc[:, :, None, None], chunks="auto")

            if comp == "expansion":
                sampled_coeffs = self._calc_expansion_contribution()
                montecarlo_R = montecarlo_G * sampled_coeffs[:, None, :, :]
                del sampled_coeffs

            elif comp == "landwater":
                landwater_vals = self._calc_landwater_contribution(FPlist[0]["landwater"])
                montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * landwater_vals[None, None, :, :]
                del landwater_vals

            elif comp == "greenland":
                greenland_fp = self._calc_greenland_fingerprint_ar6()
                montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * greenland_fp[None, None, :, :]
            
            elif comp == "wais":
                wais_fp = self._calc_antarctic_fingerprint(comp)
                montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * wais_fp[None, None, :, :]
                del wais_fp

            elif comp == "eais":
                eais_fp = self._calc_antarctic_fingerprint(comp)
                montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * eais_fp[None, None, :, :]
                del eais_fp

            else:
                fp_vals = self._calc_fingerprint_contributions(FPlist, comp)
                montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * fp_vals[rfpi][:, None, :, :]
                del fp_vals

            montecarlo_R = da.percentile(montecarlo_R, self.output_percentiles, axis=0)
            montecarlo_R = montecarlo_R.astype(np.float32)

            # Create the output sea level projections file directory and filename
            self._save_projections(montecarlo_R, comp)

    def _save_projections(self, montecarlo_R: da.array, component: str) -> None:
        """
        Save the regional sea level projections to a file.
        :param montecarlo_R: regional sea level projections
        :param component: sea level component
        :param scenario: emission scenario
        :param percentile: percentiles used for spatial projections
        """
        # Save data in netcdf format (Assuming first dimension is percentile, but can be more general percentile/ensemble)
        xr_dataArray = xr.DataArray(montecarlo_R, dims=["percentile", "time", "lat", "lon"], 
                                    coords={"percentile": self.output_percentiles, 
                                            "time": np.arange(2006, montecarlo_R.shape[1] + 2006),
                                            "lat": np.arange(-90, 90) + 0.5, "lon": np.arange(0, 360) + 0.5})
        xr_dataArray.attrs["units"] = "m"
        xr_dataArray.attrs["long_name"] = f"Regional {component} sea-level projections"
        ds = xr_dataArray.to_dataset(name=component)

        ds.attrs["source"] = "ProFSea-Climate v0.1"

        file_header = f"{component}_{self.scenario}_projection_{self.end_year}"
        R_file = '_'.join([file_header, 'regional']) + '.nc'
        encoding = {component: {"zlib": True, "complevel": 5, "dtype": "float32"}}
        ds.to_netcdf(os.path.join(self.output_dir, R_file), encoding=encoding, compute=True)

    def _read_gia_estimates(self) -> tuple:
        """
        Read in pre-processed interpolator objects of GIA estimates (Lambeck,
        ICE5G)
        :param: none
        :return: length of GIA_vals and numpy array of pre-processed interpolator
        objects of GIA estimates
        """
        gia_path = next(Path(self.gia_dir).glob("global*"))
        with open(gia_path, "rb") as ifp:
            GIA_dict = pickle.load(ifp, encoding='latin1') # Interp objects

        GIA_vals = []
        for key in list(GIA_dict.keys()):
            val = GIA_dict[key].values
            GIA_vals.append(val)

        nGIA = len(GIA_vals)
        GIA_vals = np.array(GIA_vals)

        # Sort out the crazy values in the 0th GIA array
        GIA_vals[0][GIA_vals[0] < -99999] = 0

        # AND shift them from -180, 180 to 0, 360
        GIA_vals = np.roll(GIA_vals, 180, axis=2)
        return nGIA, GIA_vals

    def _calc_expansion_contribution(self) -> da.array:
        """
        Calculate the thermal expansion contribution to the regional component of
        sea level rise.
        :param scenario: emission scenario
        :param nsmps: determine the number of samples
        :return: expansion estimates converted to mm/yr
        """
        # Select slope coefficients based on the MIP
        coeffs = self._load_CMIP6_slopes()
        coeffs = da.roll(coeffs, 180, axis=2)

        rand_samples = self.rng.choice(
            coeffs.shape[0], size=self.n_samples, replace=True)               
        rand_coeffs = coeffs[rand_samples, :, :]
        return rand_coeffs

    def _calc_landwater_contribution(self, data: xr.DataArray) -> da.array:
        """
        Calculate the regional landwater contribution to sea level rise.
        :param interpolator: dictionary of interpolator objects
        :return: numpy array of landwater values
        """
        landwater_vals = interpolate(data, self.nlat, self.nlon)
        landwater_vals = da.roll(landwater_vals, 180, axis=1) # change lons from (-180,180) to (0,360)
        return landwater_vals

    def _calc_fingerprint_contributions(self, FPlist: list, comp: str) -> da.array:
        # Initiate an empty list for fingerprint values
        fp_vals = []
        for FP_dict in FPlist:
            # Interpolate values to target lat/lon
            val = FP_dict[comp]
            val = interpolate(val, self.nlat, self.nlon)
            val = da.roll(val, 180, axis=1) # change lons from (-180,180) to (0,360)
            fp_vals.append(val)

        fp_vals = da.stack(fp_vals, axis=0)
        return fp_vals

    def _calc_greenland_fingerprint_ar6(self) -> da.array:
        """Load and prepare the GIS fingerprint.

        This fingerprint was/is used by FACTS for AR6 projections, and was 
        originally calculated by Mitrovica et al., (2011).
        
        :return dask array containing GIS fingerprint
        """
        # Load in the fingerprint
        fp_path = Path(self.fingerprint_dir) / "greenland_ar6.nc"
        fp_ds = xr.open_dataset(fp_path, chunks={})

        # Interpolate to (180, 360) grid
        fp_vals = fp_ds.fp.interp(
            lat=np.linspace(-90, 90, self.nlat, endpoint=False) + 0.5, 
            lon=np.linspace(0, 360, self.nlon, endpoint=False) + 0.5, 
            method="linear").data * 1000  # convert mm to m SLE per m GMSLR
        return fp_vals

    def _calc_antarctic_fingerprint(self, fname: str) -> da.array:
        """Load and prepare the WAIS fingerprint.

        Fingerprints from Kopp, R. E. (2022). Framework for Assessing Changes 
        To Sea-level (FACTS) Module Data (1.0) [Data set]. Zenodo.
        
        :return dask array containing Antarctic fingerprint
        """
        # Load in the fingerprint
        fp_path = Path(self.fingerprint_dir) / os.path.join(fname, ".nc")
        print(fp_path)
        fp_ds = xr.open_dataset(fp_path, chunks={})

        # Interpolate to (180, 360) grid
        fp_vals = fp_ds.fp.interp(
            lat=np.linspace(-90, 90, self.nlat, endpoint=False) + 0.5, 
            lon=np.linspace(0, 360, self.nlon, endpoint=False) + 0.5, 
            method="linear").data * 1000  # convert mm to m SLE per m GMSLR
        return fp_vals

    def _load_CMIP6_slopes(self) -> np.ndarray:
        """
        Load in the CMIP6 slope coefficients.
        :param site_loc: name of the site location
        :param scenario: emissions scenario
        :return: 1D array of regression coefficients
        """
        # Read in the sea level regressions
        slope_files = list(Path(
            self.expansion_patterns_dir
        ).glob("*/zos_regression_ssp585_*.npy"))

        def _load_one_slope(f):
            return np.load(f, mmap_mode='r')

        # Create a list of lazy dask arrays
        lazy_slopes = [
            da.from_array(_load_one_slope(f), chunks=(180, 360)) 
            for f in slope_files]
        slopes_stack = da.stack(lazy_slopes, axis=0)

        return slopes_stack

    def _load_fingerprints(self) -> tuple:
        """
        Create 2D Interpolator objects for the Slangen, Spada and Klemann
        fingerprints
        :param components: list of sea level components
        :return nFPs: length of FPlist and interpolator objects of all sea level
        components
        """
        # Create empty dictionaries for the Slangen, Spada and Klemann fingerprints
        # interpolator objects.
        slangen_FPs = {}
        spada_FPs = {}
        klemann_FPs = {}

        # Only 1 fingerprint for Landwater
        comp = "landwater"
        landwater_path = Path(self.fingerprint_dir) / f"{comp}_slangen_nomask.nc"
        slangen_FPs[comp] = xr.open_dataarray(landwater_path, chunks={})

        # Other FPs have multiple components
        components_todo = [
            c for c in list(self.components.keys()) 
            if c not in ["expansion", "landwater", "greenland", "wais", "eais"]]
        for comp in components_todo:
            slangen_path = Path(self.fingerprint_dir) / f"{comp}_slangen_nomask.nc"
            spada_path = Path(self.fingerprint_dir) / f"{comp}_spada_nomask.nc"
            klemann_path = Path(self.fingerprint_dir) / f"{comp}_klemann_nomask.nc"
            slangen_FPs[comp] = xr.open_dataarray(slangen_path, chunks={})
            spada_FPs[comp] = xr.open_dataarray(spada_path, chunks={})
            klemann_FPs[comp] = xr.open_dataarray(klemann_path, chunks={})

        FPlist = [slangen_FPs, spada_FPs, klemann_FPs]
        nFPs = len(FPlist)
        return nFPs, FPlist
