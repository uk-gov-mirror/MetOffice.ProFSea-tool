from pathlib import Path

from rich.progress import track
import numpy as np
import xarray as xr

class Antarctica:
    def __init__(
            self, 
            params_path: Path, 
            n_samples: int=1000) -> None:
        """
        """
        self.param_ds = xr.load_dataset(params_path)
        self.n_samples = n_samples

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

    def _inst_function(self, tas, t_int, params_gen, params_resid):
        """
        """
        # params: [a, b]
        p = params_gen + params_resid
        return (p[:, 0, None] * tas) + (p[:, 1, None] * t_int)

    def _impulse_response(self, f_inst_vals, tau):
        """
        """
        n = len(f_inst_vals)
        # Avoid division by zero if tau is tiny
        tau = max(tau, 1e-3)
        decay_factors = np.exp(-np.arange(n) / tau) * (1 / tau)
        
        # Efficient convolution
        # Note: scipy.signal.convolve is usually faster for large n, 
        # but keeping your implementation for consistency.
        response = np.zeros(n)
        for k in range(n):
            response[k] = np.sum(decay_factors[:k+1] * f_inst_vals[k::-1])
        return response


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
            gen_p = self.param_ds.general_params[model].values  # [alpha, beta, gamma]
            resid_p = self.mv_dists[model]  # [n_samples, n_params]
            
            # Calculate forcing for all samples at once
            f_vals = self._inst_function(tas[None, :], tas_int[None, :], gen_p[None, :], resid_p)
            
            # Convolve
            for sample in range(self.n_samples):
                all_preds[model, sample, :] = self._impulse_response(f_vals[sample], tau)
        return all_preds
