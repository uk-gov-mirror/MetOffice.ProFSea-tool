import concurrent.futures
import json
import multiprocessing


from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    worker_tas = tas
    worker_ohc = ohc
    worker_emissions = emissions
    worker_scenarios = scenarios

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

    tas = f.temperature.loc[dict(layer=0, timebounds=np.arange(2006, 2300))] - tas_baseline  # shape (294, 7, 841)
    ohc = f.ocean_heat_content_change.loc[dict(timebounds=np.arange(2006, 2300))] - ohc_baseline
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
            2300,
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
    time = np.arange(2006, 2300)
    ssp_colours = {
        "ssp119": tuple(np.array([0, 173, 207]) / 255),
        "ssp126": tuple(np.array([23, 60, 102]) / 255),
        "ssp245": tuple(np.array([247, 148, 32]) / 255),
        "ssp370": tuple(np.array([231, 29, 37]) / 255),
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

def main():
    # Set up for main loop
    percentiles = [5, 17, 50, 83, 95]
    scenarios = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]

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
    tas, ohc = run_fair(baseline_start, baseline_end, scenarios)

    # Pre-generate the random ensemble member indices
    n_iterations = 1000
    rng = np.random.default_rng()
    random_indices = rng.integers(0, high=841, size=n_iterations)

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

    save_to_netcdf(sampled_components, "probabilistic_projections/global/gmslr_projections.nc")
    
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

    plot_samples(sampled_tas[:, :, 0], sampled_ohc[:, :, 0])


if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()