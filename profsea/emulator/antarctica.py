from pathlib import Path

from rich.progress import track
import numpy as np
from scipy.signal import fftconvolve
import xarray as xr

class Antarctica:
    def __init__(
            self, 
            params_path: Path, 
            samples: int=1000) -> None:
        """
        """
        self.param_ds = xr.load_dataset(params_path)
        self.n_samples = samples

        self.n_models = self.param_ds.coords["model"].shape[0]
        n_params = self.param_ds.coords["param"].shape[0]
        self.mv_dists = self._calc_mv_dists(n_params)

    def _calc_mv_dists(self, n_params: int):
        """
        """
        mv_dists = []
        for m in range(self.n_models):
            # Calculate covariance of param residuals
            cov = np.cov(self.param_ds.param_residuals[m].T)
            
            # Add jitter in case of small param values
            cov += np.eye(n_params) * 1e-8 
            
            # Draw samples centered at 0 
            samples = np.random.multivariate_normal(np.zeros(n_params), cov, size=self.n_samples) 
            mv_dists.append(samples)
        return np.array(mv_dists)

    def _impulse_response_term(
            self, 
            tas: np.ndarray, 
            tau: float, 
            params: np.ndarray):
        """
        """
        n_time = tas.shape[-1]
        a_lin = params[:, 0]

        decay_factors = np.exp(-np.arange(n_time) / tau) * (1 / tau)
        t_conv = fftconvolve(tas, decay_factors, mode='full')[:n_time]
        
        term_slow = (a_lin[:, None] * t_conv[None, :])
        return term_slow

    def predict(self, tas: np.ndarray, tas_int: np.ndarray, display_progress=True):
        """
        """
        n_time = tas.shape[0]
        
        # Shape: (n_models, n_samples, n_time)
        all_preds = np.zeros((self.n_models, self.n_samples, n_time))
        
        for model in track(
                range(self.n_models), 
                description="Projecting AIS response... ", 
                disable=not display_progress):
            tau = self.param_ds.tau[model].values
            general_p = self.param_ds.general_params[model].values  # [alpha, beta, gamma]
            resid_p = self.mv_dists[model]  # [n_samples, n_params]

            total_params = general_p + resid_p
            term_slow = self._impulse_response_term(tas, tau, total_params)
            term_fast = total_params[:, 1][:, None] * tas_int[None, :]

            # Convolve
            all_preds[model, :, :] = term_fast + term_slow
        return all_preds
