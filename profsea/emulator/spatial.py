"""
Copyright (c) 2023, Met Office
All rights reserved.
"""
import glob
import pickle
import os
from pathlib import Path
import warnings

import dask.array as da
from netCDF4 import Dataset
import numpy as np
from rich.console import Console
from rich.progress import track
import xarray as xr

from profsea.config import settings
from profsea.directories import read_dir

console = Console()
warnings.filterwarnings("ignore")

class Spatial:
    """
    """
    def __init__(
            self, 
            components: dict, 
            end_year=2301, 
            output_ensemble: list|np.ndarray=[],
            output_dir: str|Path=None
        ):
        self.n_samples = components["gmslr"].shape[0]
        self.end_year = end_year
        self.start_year = 2006
        self.n_years = self.end_year - self.start_year

    def _calc_baseline_period(self) -> float:
        """
        Baseline years used for IPCC AR5 and Palmer et al 2020 -- 1986-2005
        :param yrs: years of the projections
        :return: baseline years
        """
        byr1 = 1986.
        byr2 = 2005.

        console.log("Baseline period = ", byr1, "to", byr2)
        midyr = (byr2 - byr1 + 1) * 0.5 + byr1
        return self.start_year - midyr

    def _calc_gia_contribution(self, scenario: str) -> None:
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
        rgiai = np.random.randint(nGIA, size=num_percs)

        GIA_T = da.from_array(GIA_unit_series)
        GIA_vals = da.from_array(GIA_vals)
        GIA_series = GIA_T[:, :, None, None] * GIA_vals[rgiai, None, :, :]

        file_header = '_'.join(
            ['gia', scenario, "projection", self.end_year])

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
        # Numbers of ensemble members, samples, years
        # Select dimensions from sample file, [time, realisation]
        console.log(f"Running with {self.n_samples} ensemble members")

        grid_path = os.path.join(
            settings["cmipinfo"]["sealevelbasedir"], 
            "ACCESS-CM2/zos_regression_ssp245_ACCESS-CM2.npy")
        nFPs, FPlist = self._load_fingerprints(components)
        resamples = np.random.choice(self.n_samples, self.n_samples) # Preserve correlations across comps
        rfpi = np.random.randint(nFPs, size=self.n_samples)

        # Calculate GIA contribution and save it out
        self._calc_gia_contribution(scenario)
        components = [
            'expansion', 'antdyn', 'antsmb', 
            'greenland', 'glacier', 'landwater']
        for comp in track(components, description="Calculating components..."):
            montecarlo_R = da.zeros((nsmps, nyrs, lats, lons), dtype=np.float32) # (FPs applied) + GIA
            montecarlo_G = da.zeros((nsmps, nyrs, lats, lons), dtype=np.float32) # (no FPs applied)

            # Load global projections in for the component
            #mc_timeseries = np.load(os.path.join(mcdir, f'{scenario}_{comp}.npy'))
            mc_timeseries = np.load(os.path.join(settings["baseoutdir"],settings["experiment_name"],
                                                'data','gmslr',f'{scenario}_{comp}.npy'), mmap_mode='r')
            sampled_mc = mc_timeseries[resamples, :nyrs]
            montecarlo_G[:, :] = da.from_array(sampled_mc[:, :, None, None], chunks="auto")

            if comp == "expansion":
                sampled_coeffs = self._calc_expansion_contribution(scenario, nsmps)
                montecarlo_R = montecarlo_G * sampled_coeffs[:, None, :, :]
                del sampled_coeffs

            elif comp == "landwater":
                landwater_vals = self._calc_landwater_contribution(FPlist[0]["landwater"], lats, lons)
                montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * landwater_vals[None, None, :, :]
                del landwater_vals

            elif comp == "greenland":
                greenland_fp = self._calc_greenland_fingerprint_ar6(lats, lons)
                montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * greenland_fp[None, None, :, :]

            else:
                fp_vals = self._calc_fingerprint_contributions(FPlist, comp, lats, lons)
                montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * fp_vals[rfpi][:, None, :, :]
                del fp_vals

            montecarlo_R = da.percentile(montecarlo_R, self.output_percentiles, axis=0)
            montecarlo_R = montecarlo_R.astype(np.float32)

            # Create the output sea level projections file directory and filename
            self._save_projections(montecarlo_R, comp, scenario)

    def _save_projections(self, montecarlo_R: da.array, component: str, scenario: str) -> None:
        """
        Save the regional sea level projections to a file.
        :param montecarlo_R: regional sea level projections
        :param component: sea level component
        :param scenario: emission scenario
        :param percentile: percentiles used for spatial projections
        """
        file_header = '_'.join([component, scenario, "projection", 
                                f"{settings['projection_end_year']}"])

        # Save data in netcdf format (Assuming first dimension is percentile, but can be more general percentile/ensemble)
        xr_dataArray = xr.DataArray(montecarlo_R, dims=["percentile", "time", "lat", "lon"], 
                                    coords={"percentile": self.output_percentiles, 
                                            "time": np.arange(2006, montecarlo_R.shape[1] + 2006),
                                            "lat": np.arange(-90, 90) + 0.5, "lon": np.arange(0, 360) + 0.5})
        xr_dataArray.attrs["units"] = "m"
        xr_dataArray.attrs["long_name"] = f"Regional {component} sea-level projections"
        ds = xr_dataArray.to_dataset(name=component)

        ds.attrs["source"] = "ProFSea-Climate v0.1"

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
        gia_file = settings["giaestimates"]["global"]
        with open(gia_file, "rb") as ifp:
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

    def _calc_expansion_contribution(self, scenario: str, nsmps: int) -> da.array:
        """
        Calculate the thermal expansion contribution to the regional component of
        sea level rise.
        :param scenario: emission scenario
        :param nsmps: determine the number of samples
        :return: expansion estimates converted to mm/yr
        """
        # Select slope coefficients based on the MIP
        if settings["emulator_settings"]["emulator_mode"]:
            if settings["cmipinfo"]["mip"].lower() == "cmip6":
                coeffs = self._load_CMIP6_slopes('ssp585')
                coeffs = da.roll(coeffs, 180, axis=2)
            else:
                coeffs = self._load_CMIP5_slope_coeffs('rcp85')
        else:
            coeffs = self._load_CMIP5_slope_coeffs(scenario)

        rand_samples = np.random.choice(
            coeffs.shape[0], size=nsmps, replace=True)               
        rand_coeffs = coeffs[rand_samples, :, :]
        return rand_coeffs


    def _calc_landwater_contribution(self, data: dict, lats: int, lons: int) -> da.array:
        """
        Calculate the regional landwater contribution to sea level rise.
        :param interpolator: dictionary of interpolator objects
        :return: numpy array of landwater values
        """
        landwater_vals = self._interpolate(data, lats, lons)
        landwater_vals = da.roll(landwater_vals, 180, axis=1)
        return landwater_vals


    def _interpolate(self, data: da.array, lats: int, lons: int) -> np.ndarray:
        """
        """
        original_da = xr.DataArray(
            data.data,
            coords=[
                ("lat", np.linspace(90, -90, data.shape[0])), 
                ("lon", np.linspace(-180, 180, data.shape[1], endpoint=False))
            ],
            name="v")

        target_lat = np.linspace(90, -90, lats) + 0.5
        target_lon = np.linspace(-180, 180, lons, endpoint=False) + 0.5
        data_interp = original_da.interp(
            lat=target_lat, lon=target_lon, method="linear").data

        data_interp = da.roll(data_interp, 180, axis=1)
        return data_interp

    def _calc_fingerprint_contributions(
            self, FPlist: list, comp: str,
            lats: int, lons: int) -> da.array:
        # Initiate an empty list for fingerprint values
        fp_vals = []
        for FP_dict in FPlist:
            # Interpolate values to target lat/lon
            val = FP_dict[comp]
            val = self._interpolate(val, lats, lons)
            fp_vals.append(val)

        fp_vals = da.stack(fp_vals, axis=0)
        return fp_vals

    def _calc_greenland_fingerprint_ar6(self, lats: int, lons: int) -> da.array:
        """Load and prepare the GIS fingerprint.

        This fingerprint was/is used by FACTS for AR6 projections, and was 
        originally calculated by Mitrovica et al., (2011).
        
        :return dask array containing GIS fingerprint
        """
        # Load in the fingerprint
        fp_path = Path(settings["fingerprints"]) / "greenland_ar6.nc"
        fp_ds = xr.open_dataset(fp_path, chunks={})

        # Interpolate to (180, 360) grid
        fp_vals = fp_ds.fp.interp(
            lat=np.linspace(-90, 90, lats, endpoint=False) + 0.5, 
            lon=np.linspace(0, 360, lons, endpoint=False) + 0.5, 
            method="linear").data * 1000  # convert mm to m SLE per m GMSLR

        # Flip vertically and roll by 180 degrees
        fp_vals = da.flip(fp_vals)
        fp_vals = da.roll(fp_vals, 180, axis=1)
        return fp_vals

    def _get_projection_info(self, indir: str, scenario: str) -> tuple:
        """
        Read in the dimensions of the Monte-Carlo data. These files are all
        relative to midnight on 1st January 2007
        :param indir: directory of Monte Carlo time series for new projections
        :param scenario: emission scenarios to be considered
        :return: Number of ensemble members in time series, number of years in
        each projection time series and the years of the projections
        """
        sample_file = f'{scenario}_exp.nc'
        f = Dataset(f'{indir}{sample_file}', 'r')
        nesm = f.dimensions['realization'].size
        t = f.variables['time']
        nyrs = t.size
        unit_str = t.units
        first_year = int(unit_str.split(' ')[2][:4])
        f.close()

        yrs = first_year + np.arange(nyrs)
        return nesm, nyrs, yrs

    def _load_CMIP5_slope_coeffs(self, scenario: str) -> np.ndarray:
        """
        Loads in the CMIP slope coefficients based on linear regression of
        'zos+zostoga' against 'zostoga' for the period 2005 to 2100.
        Some models are missing regression slopes for RCP2.6. If so, use RCP4.5
        values instead.
        :param site_loc: name of the site location
        :param scenario: emissions scenario
        :return: 1D array of regression coefficients
        """
        # Read in the sea level regressions
        in_zosddir = read_dir()[2]
        filename = os.path.join(in_zosddir, 'zos_regression.npy')
        coeffs = np.load(filename)
        scenario_index = ['rcp26', 'rcp45', 'rcp85'].index(scenario)
        coeffs = coeffs[:, scenario_index, :, :]
        coeffs[np.isnan(coeffs)] = 0
        coeffs[coeffs > 999] = 0
        coeffs[coeffs < -999] = 0
        return coeffs

    def _load_CMIP6_slopes(self, scenario: str) -> np.ndarray:
        """
        Load in the CMIP6 slope coefficients.
        :param site_loc: name of the site location
        :param scenario: emissions scenario
        :return: 1D array of regression coefficients
        """
        # Read in the sea level regressions
        cmip6_dir = settings["cmipinfo"]["sealevelbasedir"]
        slope_files = glob.glob(cmip6_dir + f'/*/zos_regression_{scenario}_*.npy')

        def load_one_slope(f):
            return np.load(f, mmap_mode='r')

        # Create a list of lazy dask arrays
        lazy_slopes = [
            da.from_array(load_one_slope(f), chunks=(180, 360)) 
            for f in slope_files]
        slopes_stack = da.stack(lazy_slopes, axis=0)

        return slopes_stack

    def _load_fingerprints(self, components: list) -> tuple:
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
        slangen_FPs[comp] = xr.open_dataarray(
            os.path.join(settings["fingerprints"],
            comp + "_slangen_nomask.nc"), chunks={})

        # Other FPs have multiple components
        components_todo = [
            c for c in components 
            if c not in ["expansion", "landwater", "greenland"]]
        for comp in components_todo:
            slangen_FPs[comp] = xr.open_dataarray(
                os.path.join(settings["fingerprints"],
                comp + "_slangen_nomask.nc"), chunks={})
            spada_FPs[comp] = xr.open_dataarray(
                os.path.join(settings["fingerprints"],
                comp + "_spada_nomask.nc"), chunks={})
            klemann_FPs[comp] = xr.open_dataarray(
                os.path.join(settings["fingerprints"],
                comp + "_klemann_nomask.nc"), chunks={})

        FPlist = [slangen_FPs, spada_FPs, klemann_FPs]
        nFPs = len(FPlist)
        return nFPs, FPlist
