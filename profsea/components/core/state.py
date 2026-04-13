from dataclasses import dataclass
import numpy as np

@dataclass
class ClimateState:
    """ClimateState context object to hold relevant state information for SLR projections."""
    scenario: str
    T_ens: np.ndarray
    T_int_ens: np.ndarray
    T_int_med: np.ndarray
    therm_ens: np.ndarray
    fraction: np.ndarray # shared correlation array for AntSMB/AntDyn
    nyr: int
    nt: int
    nm: int
