import argparse
import concurrent.futures
import json

from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich_argparse import RichHelpFormatter
from rich.progress import Progress
from scipy.spatial.distance import cdist
import xarray as xr

from profsea.emulator import GMSLREmulator

worker_tas = None
worker_ohc = None
worker_emissions = None
worker_scenarios = None

def init_worker(tas, ohc, emissions, scenarios):
    """Initialize worker with read-only shared data."""
    global worker_tas, worker_ohc, worker_emissions, worker_scenarios
    worker_tas = tas.copy()
    worker_ohc = ohc.copy()
    worker_emissions = emissions.copy()
    worker_scenarios = scenarios.copy()


def index_df(df: pd.DataFrame, baseline_start: int, baseline_end: int) -> pd.DataFrame:
    meta_cols = ['ensemble_member', 'scenario', 'region', 'variable', 'unit', 'climate_model']
    existing_meta = [c for c in meta_cols if c in df.columns]
    df_indexed = df.set_index(existing_meta)

    years = df_indexed.columns.astype(str).str[:4].astype(int)
    df_indexed.columns = years

    mask_full = (years >= 1750) & (years <= 2300)
    df_indexed = df_indexed.loc[:, mask_full]

    years_sliced = df_indexed.columns
    baseline_mask = (years_sliced >= baseline_start) & (years_sliced <= baseline_end)
    
    if not baseline_mask.any():
        raise ValueError(f"No years found in baseline range {baseline_start}-{baseline_end}")

    baseline_means = df_indexed.loc[:, baseline_mask].mean(axis=1)
    df_anom = df_indexed.sub(baseline_means, axis=0)
    years_final = df_anom.columns
    mask_final = (years_final >= 2006) & (years_final <= 2300)
    
    df_final = df_anom.loc[:, mask_final].reset_index()
    
    return df_final


def df_to_arr(df, scenario_order):
    meta_cols = ['ensemble_member', 'scenario', 'region', 'variable', 'unit', 'climate_model']
    existing_meta = [c for c in meta_cols if c in df.columns]
    
    # Set the index (this creates a MultiIndex including 'scenario')
    df = df.set_index(existing_meta)
    array = []
    for scenario in scenario_order:
        try:
            mask = df.index.get_level_values('scenario') == scenario
            group_df = df.loc[mask]
        except KeyError:
             raise ValueError(f"Scenario '{scenario}' not found in the input DataFrame.")

        if group_df.empty:
             raise ValueError(f"No data found for scenario '{scenario}'")

        # Sort by member to ensure internal alignment
        group_sorted = group_df.sort_index(level='ensemble_member')
        scenario_data = group_sorted.values
        array.append(scenario_data)

    array = np.stack(array, axis=0)
    array = array.transpose(2, 0, 1)  # (n_scenarios, n_members, n_time)
    
    return array


def load_magicc_forcing(
        input_path: str, scenarios: list,
        baseline_start: int, baseline_end: int) -> tuple[np.ndarray]:
    df = pd.read_csv(input_path, index_col=0)
    
    tas_condition = (
        (df["variable"] == "Surface Air Temperature Change") & 
        (df["scenario"].isin(scenarios))
    )
    ohc_condition = (
        (df["variable"] == "Heat Content|Ocean") & 
        (df["scenario"].isin(scenarios))
    )
    tas_df = df.loc[tas_condition]
    ohc_df = df.loc[ohc_condition]

    tas_df = index_df(tas_df, baseline_start, baseline_end)
    ohc_df = index_df(ohc_df, baseline_start, baseline_end)
    tas_arr = df_to_arr(tas_df, scenarios)  # shape (scenario, mem, time)
    ohc_arr = df_to_arr(ohc_df, scenarios) * 1e21  # convert to J from ZJ
    return tas_arr, ohc_arr


def run_fair(baseline_start: int, baseline_end: int, scenarios: list) -> tuple[np.ndarray]:
    f = FAIR()
    f.define_time(1750, 2300, 1)
    f.define_scenarios(scenarios)
    species, properties = read_properties('../data/fair/fair-parameters/species_configs_properties_1.4.1.csv')
    f.define_species(species, properties)
    f.ch4_method='Thornhill2021'
    df_configs = pd.read_csv('../data/fair/fair-parameters/calibrated_constrained_parameters_1.4.1.csv', index_col=0)
    f.define_configs(df_configs.index)
    f.allocate()

    f.fill_from_rcmip()
    f.fill_species_configs('../data/fair/fair-parameters/species_configs_properties_1.4.1.csv')
    f.override_defaults('../data/fair/fair-parameters/calibrated_constrained_parameters_1.4.1.csv')
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)

    f.run()

    # Make anomaly with respect to baseline
    tas_baseline = f.temperature.loc[dict(layer=0, timebounds=np.arange(baseline_start, baseline_end+1))].mean()
    ohc_baseline = f.ocean_heat_content_change.loc[dict(timebounds=np.arange(baseline_start, baseline_end+1))].mean()

    tas = f.temperature.loc[dict(layer=0, timebounds=np.arange(2006, 2301))] - tas_baseline  # shape (294, 7, 841)
    ohc = f.ocean_heat_content_change.loc[dict(timebounds=np.arange(2006, 2301))] - ohc_baseline
    return tas.values, ohc.values


def simulation_task(random_idx):
    """Runs the emulator for a single ensemble member across all scenarios."""
    # Access data from global scope (initialized via init_worker)
    tas_sampled = worker_tas[:, :, random_idx]
    ohc_sampled = worker_ohc[:, :, random_idx]
    results = {}
    for scenario in worker_scenarios:
        results[scenario] = {}

    for idx, scenario in enumerate(worker_scenarios):
        emulator = GMSLREmulator(
            np.expand_dims(tas_sampled[:, idx], axis=0),
            np.expand_dims(ohc_sampled[:, idx], axis=0),
            scenario,
            2301,
            input_ensemble=True,
            random_sample=True,
            cum_emissions_total=worker_emissions[scenario],
            parallel=False 
        )
        emulator.project()

        results[scenario]["gmslr"] = emulator.gmslr
        results[scenario]["expansion"] = emulator.expansion
        results[scenario]["antarctica"] = emulator.antnet
        results[scenario]["greenland"] = emulator.greenland_ar6
        results[scenario]["glaciers"] = emulator.glacier
        results[scenario]["landwater"] = emulator.landwater
    return tas_sampled, ohc_sampled, results


def plot_samples(tas: np.ndarray, ohc: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 6), layout="constrained")

    ax = fig.add_subplot(121)
    ax.plot(tas.T, color='seagreen', alpha=0.05)
    ax.set_xlabel("Simulation years")
    ax.set_ylabel("GMST ($\degree$C)")

    ax = fig.add_subplot(122)
    ax.plot(ohc.T, color='seagreen', alpha=0.05)
    ax.set_xlabel("Simulation years")
    ax.set_ylabel("OHC (J)")

    plt.show()
    plt.close()


def sample_members_2D(array: np.ndarray, percentiles: list|np.ndarray) -> np.ndarray:
    """Sample real ensemble members from a 2D numpy array."""
    # Caculate statistical timeseries, then match with closest real timeseries 
    array_percentiles = np.percentile(array, percentiles, axis=0)
    distances = cdist(array_percentiles, array)
    mem_indices = np.argmin(distances, axis=1)
    return array[mem_indices]


def process_global_ensemble(components: list, percentiles: list, scenario: str) -> None:
    for comp, data in components.items():
        # Since the spatial projections won't change the ensemble order
        # we can take percentiles here to pass to spatialise.py
        sampled_ensemble = sample_members_2D(data, percentiles)
        components[comp] = sampled_ensemble
    return components


def save_to_netcdf(components: dict, filename: str) -> None:
    """
    Saves the full ensemble results to a single NetCDF file.
    
    Structure:
        Dimensions: (scenario, member, year)
        Variables: gmslr, expansion, glaciers, etc.
    """
    scenarios = list(components.keys())
    comp_names = list(components[scenarios[0]].keys())
    sample = components[scenarios[0]][comp_names[0]]
    n_member, n_time = sample.shape
    
    # Create coords
    years = np.arange(2006, 2006 + n_time)
    members = np.arange(n_member)
    data_vars = {}
    for comp in comp_names:
        # Stack data across scenarios
        # Resulting shape: (n_scenario, n_member, n_time)
        stacked_data = np.stack([components[s][comp] for s in scenarios], axis=0)
        data_vars[comp] = xr.DataArray(
            data=stacked_data,
            dims=["scenario", "member", "year"],
            coords={
                "scenario": scenarios,
                "member": members,
                "year": years
            },
            attrs={
                "units": "m", 
                "description": f"Sea level contribution from {comp}"
            }
        )
        
    ds = xr.Dataset(data_vars)
    ds.attrs = {
        "title": "ProFSea GMSLR Projections",
        "source": "FAIR v2.2 + ProFSea Emulator",
    }
    
    # Save with compression
    encoding = {var: {"zlib": True, "complevel": 5} for var in data_vars}
    ds.to_netcdf(filename, encoding=encoding)
    print(f"Successfully saved full ensemble to {filename}")


def plot_component(
        ax: plt.Axes, component_dict: dict, component: str, 
        scenarios: list, plot_legend: bool=False) -> None:
    time = np.arange(2006, 2301)
    ssp_colours = {
        "ssp119": tuple(np.array([0, 173, 207]) / 255),
        "ssp126": tuple(np.array([23, 60, 102]) / 255),
        "ssp245": tuple(np.array([247, 148, 32]) / 255),
        "ssp370": tuple(np.array([231, 29, 37]) / 255),
        "ssp534-over": tuple(np.array([0, 79, 0]) / 255),
        "ssp585": tuple(np.array([149, 27, 30]) / 255)
    }
    for scenario in reversed(scenarios):
        plt.fill_between(
            time, 
            component_dict[scenario][component][1], 
            component_dict[scenario][component][3],  
            color=ssp_colours[scenario], 
            edgecolor='none',
            alpha=0.3)
        plt.plot(
            time, 
            component_dict[scenario][component][2], 
            label=f"{scenario}", 
            color=ssp_colours[scenario])

    ax.set_xlabel("Year")
    ax.set_ylabel("SLE (m)")
    ax.set_title(component)
    if plot_legend:
        ax.legend(frameon=False, loc="upper left")


def main(args):
    # Set up for main loop
    percentiles = [5, 17, 50, 83, 95]
    scenarios = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"]

    emissions_path = "/Users/gregorymunday/Documents/Papers/ProFSea/cmip7-slr/data/cumulative_cmip6_emissions.json"
    with open(emissions_path) as f:
        cumulative_emissions = json.load(f)

    baseline_start = 1995
    baseline_end = 2014 # inclusive
    
    # Set up components data struct
    components = {}
    for scenario in scenarios:
        components[scenario] = {
            "gmslr": [], "expansion": [], "antarctica": [],
            "greenland": [], "glaciers": [], "landwater": []
        }

    # Enter 1000x loop
    if args.input.lower() == "magicc":
        tas, ohc = load_magicc_forcing(args.input_path, scenarios, baseline_start, baseline_end)
    elif args.input.lower() == "fair":
        tas, ohc = run_fair(baseline_start, baseline_end, scenarios)
    else:
        raise ValueError("Input source not recognised.")

    # Pre-generate the random ensemble member indices
    n_iterations = 1000
    rng = np.random.default_rng()
    random_indices = rng.integers(0, high=tas.shape[2], size=n_iterations)

    sampled_tas = []
    sampled_ohc = []

    max_workers = 12
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(tas, ohc, cumulative_emissions, scenarios)
    ) as exectutor:
        futures = [exectutor.submit(simulation_task, idx) for idx in random_indices]

        with Progress() as progress:
            task = progress.add_task("[orange1]Simulating and sampling...", total=n_iterations)
            for future in concurrent.futures.as_completed(futures):
                tas_sampled, ohc_sampled, result_dict = future.result()
                sampled_tas.append(tas_sampled)
                sampled_ohc.append(ohc_sampled)

                # Now add the results into the components dictionary
                for scenario in scenarios:
                    for key in components[scenario].keys():
                        components[scenario][key].append(result_dict[scenario][key])

                progress.update(task, advance=1)

    # Convert to Numpy arrays
    for scenario, data in components.items():
        for key in data:
            data[key] = np.array(data[key]).squeeze() # Now shape (nens, time)

    sampled_components = {}
    for scenario in scenarios:
        sampled_components[scenario] = process_global_ensemble(
            components[scenario], percentiles, scenario)

    save_to_netcdf(sampled_components, "probabilistic_projections/global/gmslr_projections_NEW_AIS_FASTSLOW.nc")
    
    fig = plt.figure(figsize=(16, 8), layout="constrained")
    ax = fig.add_subplot(231)
    plot_component(ax, sampled_components, "gmslr", scenarios, plot_legend=True)
    ax = fig.add_subplot(232)
    plot_component(ax, sampled_components, "expansion", scenarios)
    ax = fig.add_subplot(233)
    plot_component(ax, sampled_components, "glaciers", scenarios)
    ax = fig.add_subplot(234)
    plot_component(ax, sampled_components, "antarctica", scenarios)
    ax = fig.add_subplot(235)
    plot_component(ax, sampled_components, "greenland", scenarios)
    ax = fig.add_subplot(236)
    plot_component(ax, sampled_components, "landwater", scenarios)

    fig.savefig("probabilistic_components_ssps.png", dpi=300)
    plt.show()
    

    sampled_tas = np.asarray(sampled_tas) # shape (nens, time, nscen)
    sampled_ohc = np.asarray(sampled_ohc)

    plot_samples(sampled_tas[:, :, -1], sampled_ohc[:, :, -1])


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=RichHelpFormatter)
    p.add_argument("--input", default="fair", required=False, help="Input climate forcing source (default: FaIR)", type=str)
    p.add_argument("--input_path", required=False, help="Path to input climate forcing", type=str)
    main(p.parse_args())