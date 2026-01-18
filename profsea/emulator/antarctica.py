from pathlib import Path
from rich.progress import track
import numpy as np
from scipy.signal import fftconvolve
import xarray as xr

class AntarcticaISMIP6:
    """ISMIP6 2300 Antarctic ice-sheet emulator."""
    def __init__(self, params_path: Path, samples: int=1000):
        self.param_ds = xr.load_dataset(params_path)
        self.n_samples = samples
        self.n_models = self.param_ds.coords["model"].shape[0]

    def _impulse_response_term(self, tas: np.ndarray, tau: float, gamma: float, params: np.ndarray, dt: float):
        n_time = tas.shape[-1]
        alphas = params[:, 0] 

        decay_factors = np.exp(-np.arange(n_time) * dt / tau) * (dt / tau)
        forcing_base = np.sign(tas) * (np.abs(tas) ** gamma)
        rate_delayed = fftconvolve(forcing_base, decay_factors, mode='full')[:n_time]
        
        t_conv = np.cumsum(rate_delayed, axis=0) * dt
        term_slow = alphas[:, None] * t_conv[None, :]
        
        return term_slow

    def predict(self, tas: np.ndarray, tas_int: np.ndarray, dt: float=1.0, display_progress=True):
        """
        Projects AIS response using empirical additive bootstrapping.
        """
        n_time = tas.shape[0]
        
        # Output shape: (n_models, n_samples, n_time)
        all_preds = np.zeros((self.n_models, self.n_samples, n_time))

        # Shape assumed: (n_models, n_training_scenarios, 2)
        all_residuals = self.param_ds.param_residuals.values 
        n_train_scenarios = all_residuals.shape[1]
        for model in track(
                range(self.n_models), 
                description="Projecting AIS response... ", 
                disable=not display_progress):

            tau = float(self.param_ds.tau[model].values)
            gamma = float(self.param_ds.gamma[model].values)
            general_p = self.param_ds.general_params[model].values  # [alpha, beta, drift]
            
            rand_indices = np.random.randint(0, n_train_scenarios, size=self.n_samples)
            sampled_residuals = all_residuals[model, rand_indices, :]
            
            total_params = general_p + sampled_residuals
            term_slow = self._impulse_response_term(tas, tau, gamma, total_params, dt)

            betas = total_params[:, 1]
            term_fast = betas[:, None] * tas_int[None, :]
            term_drift = total_params[:, 2:3] * np.arange(n_time)[None, :]
            all_preds[model, :, :] = term_fast + term_slow + term_drift
            
        return all_preds