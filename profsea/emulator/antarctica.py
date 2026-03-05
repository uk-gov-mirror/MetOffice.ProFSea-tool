from pathlib import Path
from rich.progress import track
import numpy as np
from scipy.signal import fftconvolve
import xarray as xr

class AntarcticaISMIP6:
    """ISMIP6 2300 Antarctic ice-sheet emulator with two-timescale response."""
    def __init__(self, params_path: Path, samples: int=1000):
        self.param_ds = xr.load_dataset(params_path)
        self.n_samples = samples
        self.n_models = self.param_ds.coords["model"].shape[0]

    def _impulse_response_term(
            self, 
            tas: np.ndarray, 
            tau1: float, 
            tau2: float, 
            gamma: float, 
            params: np.ndarray, 
            dt: float):
        n_time = tas.shape[0]
        
        # Two slow coeffs
        alphas1 = params[:, 0] 
        alphas2 = params[:, 1] 

        forcing_base = np.sign(tas) * (np.abs(tas) ** gamma)

        # Reservoir 1
        decay_factors1 = np.exp(-np.arange(n_time) * dt / tau1) * (dt / tau1)
        rate_delayed1 = fftconvolve(forcing_base, decay_factors1, mode='full')[:n_time]
        t_conv1 = np.cumsum(rate_delayed1, axis=0) * dt
        term_slow1 = alphas1[:, None] * t_conv1[None, :]

        # Reservoir 2 
        decay_factors2 = np.exp(-np.arange(n_time) * dt / tau2) * (dt / tau2)
        rate_delayed2 = fftconvolve(forcing_base, decay_factors2, mode='full')[:n_time]
        t_conv2 = np.cumsum(rate_delayed2, axis=0) * dt
        term_slow2 = alphas2[:, None] * t_conv2[None, :]
        
        return term_slow1 + term_slow2

    def predict(self, tas: np.ndarray, dt: float=1.0, display_progress=True, seed=None):
        """
        Projects AIS response using empirical additive bootstrapping.
        """
        tas = np.squeeze(tas)
        n_time = len(tas)
        physical_time = np.arange(n_time) * dt

        # RNG
        rng = np.random.default_rng(seed)

        # Integrated temperature term
        tas_int = np.cumsum(tas) * dt

        # Output shape: (n_models, n_samples, n_time)
        all_preds = np.zeros((self.n_models, self.n_samples, n_time))

        # Shape assumed: (n_models, n_training_scenarios, 4)
        all_residuals = self.param_ds.param_residuals.values 
        n_train_scenarios = all_residuals.shape[1]
        
        for model in track(
                range(self.n_models), 
                description="Projecting AIS response... ", 
                disable=not display_progress):

            tau1 = float(self.param_ds.tau1[model].values)
            tau2 = float(self.param_ds.tau2[model].values)
            gamma = float(self.param_ds.gamma[model].values)
            
            # [alpha1, alpha2, beta, drift]
            general_p = self.param_ds.general_params[model].values  
            
            rand_indices = rng.integers(0, n_train_scenarios, size=self.n_samples)
            sampled_residuals = all_residuals[model, rand_indices, :]
            
            total_params = general_p + sampled_residuals
            
            # Slow term
            term_slow = self._impulse_response_term(tas, tau1, tau2, gamma, total_params, dt)

            # Fast term
            betas = total_params[:, 2]
            term_fast = betas[:, None] * tas_int[None, :]
            
            # Drift term
            drift_coeffs = total_params[:, 3]
            term_drift = drift_coeffs[:, None] * physical_time[None, :]
            
            all_preds[model, :, :] = term_fast + term_slow + term_drift

        return all_preds