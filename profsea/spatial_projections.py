"""
Copyright (c) 2023, Met Office
All rights reserved.
"""
import glob
import pickle
import json
import os
from pathlib import Path
import warnings

import dask.array as da
import iris
from netCDF4 import Dataset
import numpy as np
from rich.console import Console
from rich.progress import track
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
import xarray as xr

from profsea.config import settings
from profsea.directories import read_dir
from profsea.emulator import GMSLREmulator
from profsea.slr_pkg import choose_montecarlo_dir

console = Console()
warnings.filterwarnings("ignore")

def calc_baseline_period(yrs: np.array) -> float:
    """
    Baseline years used for IPCC AR5 and Palmer et al 2020 -- 1986-2005
    :param yrs: years of the projections
    :return: baseline years
    """
    byr1 = 1986.
    byr2 = 2005.

    console.log("Baseline period = ", byr1, "to", byr2)
    midyr = (byr2 - byr1 + 1) * 0.5 + byr1
    return yrs[0] - midyr


def calc_future_sea_level(scenario: str) -> None:
    """
    Calculates future sea level at the given site and write to file.
    :param df: Data frame of all metadata (tide gauge or site specific) for
               each(all) location(s)
    :param site_loc: name of the site location
    :param scenario: emission scenario
    """
    # Set the UKCP*18* random seed so results are reproducible
    np.random.seed(18)

    # Directory of Monte Carlo time series for new projections
    mcdir = choose_montecarlo_dir()

    # Specify the sea level components to include. The GIA contribution is
    # calculated separately.
    components = ['expansion', 'antdyn', 'antsmb', 'greenland',
                  'glacier', 'landwater']

    # Select dimensions from sample file, [time, realisation]
    sample = np.load(os.path.join(settings["baseoutdir"],settings["experiment_name"],
                                  'data', 'gmslr', f'{scenario}_expansion.npy'))
    
    nesm = sample.shape[0] # also number of samples to make
    nyrs = sample.shape[1]
    
    yrs = np.arange(2006, 2006 + nyrs)
    console.log(f"Running with {nesm} ensemble members")

    grid_path = os.path.join(
        settings["cmipinfo"]["sealevelbasedir"], 
        "ACCESS-CM2/zos_regression_ssp245_ACCESS-CM2.npy")
    grid_sample = np.load(grid_path)
    array_dims = [nesm, nesm, nyrs, grid_sample.shape[0], grid_sample.shape[1]]

    # Get random samples of global and regional sea level components
    calculate_sl_components(mcdir, components, scenario, yrs, array_dims)


def calc_gia_contribution(
        yrs: np.array, nyrs: int, nsmps: int, scenario: str) -> None:
    """
    Calculate the glacial isostatic adjustment (GIA) contribution to the
    regional component of sea level rise.
    Option to use the generic global GIA estimates or use the GIA estimates
    developed for UK as part of UKCP18.
    :param yrs: years of the projections
    :param nyrs: number of years in each projection time series
    :param nsmps: determine the number of samples
    :param coords: coordinates of location of interest
    :return: GIA estimates converted to mm/yr
    """
    console.log('Calculating GIA contribution...')
    nGIA, GIA_vals = read_gia_estimates()
    Tdelta = calc_baseline_period(yrs)

    # Unit series of mm/yr expressed as m/yr
    unit_series = (np.arange(nyrs) + Tdelta) * 0.001
    GIA_unit_series = np.ones([5, nyrs]) * unit_series

    # rgiai is an array of random GIA indices the size of the sample years
    rgiai = np.random.randint(nGIA, size=5)

    GIA_T = da.from_array(GIA_unit_series)
    GIA_vals = da.from_array(GIA_vals)
    GIA_series = GIA_T[:, :, None, None] * GIA_vals[rgiai, None, :, :]
    GIA_series = GIA_series.compute()

    file_header = '_'.join(['gia', scenario, "projection", 
                    f"{settings['projection_end_year']}"])
    R_file = '_'.join([file_header, 'regional']) + '.npy'
    
    np.save(os.path.join(read_dir()[4], R_file), GIA_series)
    del GIA_series


def calc_expansion_contribution(
        scenario: str, nsmps: int) -> da.array:
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
            coeffs = load_CMIP6_slopes('ssp585')
            coeffs = np.roll(coeffs, 180, axis=2)
        else:
            coeffs = load_CMIP5_slope_coeffs('rcp85')
    else:
        coeffs = load_CMIP5_slope_coeffs(scenario)

    rand_samples = np.random.choice(
        coeffs.shape[0], size=nsmps, replace=True)               
    rand_coeffs = coeffs[rand_samples, :, :]
    rand_coeffs = da.from_array(rand_coeffs)
    return rand_coeffs


def calc_landwater_contribution(interpolator: dict) -> da.array:
    """
    Calculate the regional landwater contribution to sea level rise.
    :param interpolator: dictionary of interpolator objects
    :return: numpy array of landwater values
    """
    landwater_FP_interpolator = interpolator['landwater']
    landwater_vals = da.from_array(
        landwater_FP_interpolator.values.astype(np.float32).data)
    landwater_vals = da.roll(landwater_vals, 180, axis=1)
    return landwater_vals


def calc_fingerprint_contributions(
    FPlist: list, comp: str, lats: int, lons: int) -> da.array:
    # Initiate an empty list for fingerprint values
    fp_vals = []
    for FP_dict in FPlist:
        # Interpolate values to target lat/lon
        val = FP_dict[comp].values
        if val.shape != (lats, lons):
            original_da = xr.DataArray(
                val,
                coords=[
                    ("lat", np.linspace(90, -90, 360)), 
                    ("lon", np.linspace(-180, 180, 720, endpoint=False))
                ],
                name="v")
            target_lat = np.linspace(90, -90, 180)
            target_lon = np.linspace(-180, 180, 360, endpoint=False)
            val = original_da.interp(
                lat=target_lat, lon=target_lon, method="linear").data
            val = np.roll(val, 180, axis=1)
        else:
            val = np.roll(val, 180, axis=1)
            
        fp_vals.append(val)

    fp_vals = da.from_array(np.array(fp_vals, dtype=np.float32))
    return fp_vals


def calc_greenland_fingerprint_ar6() -> da.array:
    """Load and prepare the GIS fingerprint.

    This fingerprint was/is used by FACTS for AR6 projections, and was 
    originally calculated by Mitrovica et al., (2011).
    
    :return dask array containing GIS fingerprint
    """
    # Load in the fingerprint
    fp_path = Path(settings["fingerprints"]) / "greenland_ar6.nc"
    fp_ds = xr.load_dataset(fp_path)

    # Interpolate to (180, 360) grid
    fp_vals = fp_ds.fp.interp(
        lat=np.linspace(-90, 90, 180, endpoint=False), 
        lon=np.linspace(0, 360, 360, endpoint=False), 
        method="linear").data * 1000  # convert mm to m SLE per m GMSLR

    # Flip vertically and roll by 180 degrees
    fp_vals = np.flip(fp_vals)
    fp_vals = np.roll(fp_vals, 180, axis=1)
    fp_vals = da.from_array(np.array(fp_vals, dtype=np.float32))
    return fp_vals


def save_projections(
        montecarlo_R: da.array, component: str, scenario: str, percentile: da.array) -> None:
    """
    Save the regional sea level projections to a file.
    :param montecarlo_R: regional sea level projections
    :param component: sea level component
    :param scenario: emission scenario
    :param percentile: percentiles used for spatial projections
    """
    sealev_ddir = read_dir()[4]
    file_header = '_'.join([component, scenario, "projection", 
                            f"{settings['projection_end_year']}"])
    # G_file = '_'.join([file_header, 'global']) + '.npy'
    R_file = '_'.join([file_header, 'regional']) + '.npy'

    # Save the global and local projections
    np.save(os.path.join(sealev_ddir, R_file), montecarlo_R)

    # Save data in netcdf format (Assuming first dimension is percentile, but can be more general percentile/ensemble)
    xr_dataArray = xr.DataArray(montecarlo_R, dims=["percentile", "time", "lat", "lon"], 
                                coords={"percentile": percentile, 
                                        "time": np.arange(2006, montecarlo_R.shape[1] + 2006),
                                        "lat": np.arange(-90, 90) + 0.5, "lon": np.arange(0,360) + 0.5})
    xr_dataArray.attrs["units"] = "m"
    ds = xr_dataArray.to_dataset(name = component)

    R_file = '_'.join([file_header, 'regional']) + '.nc'
    ds.to_netcdf(os.path.join(sealev_ddir, R_file))


def calculate_sl_components(
        mcdir: str, components: list, scenario: str, 
        yrs: np.array, array_dims: list) -> None:
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
    nesm, nsmps, nyrs, lats, lons = array_dims
    nFPs, FPlist = setup_FP_interpolators(components)
    resamples = np.random.choice(nesm, nsmps) # Preserve correlations across comps
    rfpi = np.random.randint(nFPs, size=nsmps)

    # Calculate GIA contribution and save it out
    calc_gia_contribution(yrs, nyrs, nsmps, scenario)

    for comp in track(components, description="Calculating components..."):
        montecarlo_R = da.zeros((nsmps, nyrs, lats, lons), dtype=np.float32) # (FPs applied) + GIA
        montecarlo_G = da.zeros((nsmps, nyrs, lats, lons), dtype=np.float32) # (no FPs applied)

        # Load global projections in for the component
        #mc_timeseries = np.load(os.path.join(mcdir, f'{scenario}_{comp}.npy'))
        mc_timeseries = np.load(os.path.join(settings["baseoutdir"],settings["experiment_name"],
                                             'data','gmslr',f'{scenario}_{comp}.npy'))
        sampled_mc = mc_timeseries[resamples, :nyrs]
        montecarlo_G[:, :] = da.from_array(sampled_mc[:, :, None, None])

        if comp == "expansion":
            sampled_coeffs = calc_expansion_contribution(scenario, nsmps)
            montecarlo_R = montecarlo_G * sampled_coeffs[:, None, :, :]
            del sampled_coeffs

        elif comp == "landwater":
            landwater_vals = calc_landwater_contribution(FPlist[0])
            montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * landwater_vals[None, None, :, :]
            del landwater_vals

        elif comp == "greenland":
            greenland_fp = calc_greenland_fingerprint_ar6()
            montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * greenland_fp[None, None, :, :]

        else:
            fp_vals = calc_fingerprint_contributions(FPlist, comp, lats, lons)
            montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * fp_vals[rfpi][:, None, :, :]
            del fp_vals

        # Take the 0th, 25th, 50th, 75th and 100th percentiles
        percentile_regional = np.array([0, 25, 50, 75, 100])
        montecarlo_R = da.percentile(montecarlo_R, percentile_regional, axis=0)
        montecarlo_R = montecarlo_R.compute()

        # Create the output sea level projections file directory and filename
        save_projections(montecarlo_R, comp, scenario, percentile_regional)


def create_FP_interpolator(
    datadir: str, dfile: str, method: str='linear') -> RegularGridInterpolator:
    """
    Generates a scipy Interpolator object from input NetCDF data of
    gravitational fingerprints (takes inputs of Latitude and Longitude).
    :param datadir: data directory
    :param dfile: data filename
    :param method: interpolation type --> 'linear' or 'nearest'
    :return: 2D Interpolator object
    """
    cube = iris.load_cube(os.path.join(datadir, dfile))
    lon = cube.coord('longitude').points
    lat = cube.coord('latitude').points

    # Define linear interpolator object:
    interp_object = RegularGridInterpolator(
        (lat, lon), cube.data,
        method=method, bounds_error=True,
        fill_value=None)
    return interp_object


def get_projection_info(indir: str, scenario: str) -> tuple:
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


def load_CMIP5_slope_coeffs(scenario: str) -> np.ndarray:
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


def load_CMIP6_slopes(scenario: str) -> np.ndarray:
    """
    Load in the CMIP6 slope coefficients.
    :param site_loc: name of the site location
    :param scenario: emissions scenario
    :return: 1D array of regression coefficients
    """
    # Read in the sea level regressions
    cmip6_dir = settings["cmipinfo"]["sealevelbasedir"]
    slope_files = glob.glob(cmip6_dir + f'/*/zos_regression_{scenario}_*.npy')
    slopes = np.zeros((len(slope_files), 180, 360), dtype=np.float32)
    for i, slope_file in enumerate(slope_files):
        slopes[i, :, :] = np.load(slope_file)

    return slopes


def read_gia_estimates() -> tuple:
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


def setup_FP_interpolators(components: list) -> tuple:
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
    slangen_FPs[comp] = create_FP_interpolator(settings["fingerprints"],
                                               comp + "_slangen_nomask.nc")

    # Create interpolators for the remaining components. Expansion ('expansion')
    # is global so no interpolation is needed.
    components_todo = [c for c in components if c not in ["expansion", "landwater", "greenland"]]
    for comp in components_todo:
        slangen_FPs[comp] = create_FP_interpolator(settings["fingerprints"],
                                                   comp + "_slangen_nomask.nc")
        spada_FPs[comp] = create_FP_interpolator(settings["fingerprints"],
                                                 comp + "_spada_nomask.nc")
        klemann_FPs[comp] = create_FP_interpolator(settings["fingerprints"],
                                                   comp + "_klemann_nomask.nc")

    FPlist = [slangen_FPs, spada_FPs, klemann_FPs]
    nFPs = len(FPlist)
    return nFPs, FPlist


def sample_members_2D(array: np.ndarray, percentile_seq: list|np.ndarray) -> np.ndarray:
        """Sample real ensemble members from a 2D numpy array."""
        # Caculate statistical timeseries, then match with closest real timeseries 
        array_percentiles = np.percentile(array, percentile_seq, axis=0)
        array_perc_diffs = array[None, :, :] - array_percentiles[:, None, :]

        # Calculate distances between statistical percentiles and real members
        distances = scipy.linalg.norm(array_perc_diffs, axis=2)
        mem_indices = np.argmin(distances, axis=1)
        distances = cdist(array_percentiles, array)
        mem_indices = np.argmin(distances, axis=1)
        return array[mem_indices]


def calculate_global_components(scenario: str, palmer_method: bool) -> None:
    """
    Calculate the global contributions for each of the sea-level components
    using the GMSLR module.
    :param scenario: string representing the scenario being simulated
    :param palmer_method: boolean to determine whether to use the palmer_method
    """
    # Check inputs are correctly set up
    if not (os.path.exists(settings["scm_data"]["temperature"]) and 
            os.path.exists(settings["scm_data"]["ocean_heat_content"])):
        raise Exception(
            'SCM data paths (temperature and ocean heat content) must be '
            'correctly configured in user-settings.yml')

    if not os.path.exists(settings["scm_data"]["cumulative_emissions"]):
        raise Exception('Cumulative emissions path must be correctly configured '
                        'in user-settings.yml')

    if (settings["scm_data"]["temperature"].split(".")[-1] != 'nc' or
        settings["scm_data"]["ocean_heat_content"].split(".")[-1] != 'nc'):
        raise Exception('SCM data must be saved in NetCDF format.')

    percentiles = np.arange(101)
    # percentiles are hard coded. We could make it an user input in future updates.

    # Now run the simulations
    console.log(f'Projecting global components for {scenario} scenario...')
    T_change = xr.load_dataarray(settings["scm_data"]["temperature"])
    OHC_change = xr.load_dataarray(settings["scm_data"]["ocean_heat_content"])
    with open(settings["scm_data"]["cumulative_emissions"]) as f:
        cumulative_emissions = json.load(f)

    T_change = T_change.sel(scenario=scenario).data.T # (member, time)
    OHC_change = OHC_change.sel(scenario=scenario).data.T

    T_change = sample_members_2D(T_change, percentiles)
    OHC_change = sample_members_2D(OHC_change, percentiles)

    gmslr = GMSLREmulator(
        T_change,
        OHC_change,
        scenario,
        settings["projection_end_year"],
        palmer_method=palmer_method,
        input_ensemble=settings["emulator_settings"]["use_input_ensemble"],
        output_percentiles=percentiles,
        cum_emissions_total=cumulative_emissions[scenario])
    gmslr.project()

    console.log('Saving components...')
    gmslr.save_components(
        os.path.join(settings["baseoutdir"],settings["experiment_name"],
                     'data', 'gmslr'),
        scenario)


    console.log('Saved!\n')



def main():
    """
    Reads in and calculates global and local (regional) sea level change
    (sum total), based on the different contributing factors e.g. thermal
    expansion, GIA and mass balance. Writes out the selected emissions scenario
    estimates of the various components and their sums.
    """
    console.log(f'\nProjecting out to: {settings["projection_end_year"]}\n')

    # Sort out paths
    Path(
        os.path.join(
            settings["baseoutdir"], 
            settings["experiment_name"])
    ).mkdir(parents=True, exist_ok=True)

    Path(
        os.path.join(
            settings["baseoutdir"],
            settings["experiment_name"],
            'data', 'gmslr')
    ).mkdir(parents=True, exist_ok=True)

    Path(
        read_dir()[4]
    ).mkdir(parents=True, exist_ok=True)

    # Extract site data from station list (e.g. tide gauge location) or
    # construct based on user input
    if settings["emulator_settings"]["emulator_mode"]:
        console.log('\nInitiating ProFSea emulator')
        if settings["projection_end_year"] > 2100:
            palmer_method = True
        else:
            palmer_method = False

        # Get the metadata of either the site location or tide gauge location
        for scenario in settings["emulator_settings"]["emulator_scenario"]:
            calculate_global_components(scenario, palmer_method)
            calc_future_sea_level(scenario)
    else:
        scenarios = ['rcp26', 'rcp45', 'rcp85']
        for scenario in scenarios:
            calc_future_sea_level(scenario)


if __name__ == '__main__':
    main()
