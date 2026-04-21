import numpy as np

from profsea.components.core.global_model import Global
from profsea.components.global_.greenland import GreenlandAR6


slr_components = {
    "greenland": GreenlandAR6(),
}

model = Global(
    components=slr_components,
    end_yr=2301
)

projections = model.run(scenario="test", T_change=np.arange(295).reshape(1, -1), OHC_change=np.arange(295).reshape(1, -1), member_seed=42)
gmslr = model.sum_components(projections)
model.save_components("test_output", "test_projection")
