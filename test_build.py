import numpy as np
import matplotlib.pyplot as plt

from profsea.components.core.global_model import Global
from profsea.components.global_.greenland import GreenlandAR6
from profsea.components.global_.antarctica import AntarcticaDynAR5, AntarcticaSMBAR5


slr_components = {
    "greenland": GreenlandAR6(),
    "antdyn": AntarcticaDynAR5(cum_emissions_total=1000),
    "antsmb": AntarcticaSMBAR5(),
}

model = Global(
    components=slr_components,
    end_yr=2301
)

projections = model.run(scenario="test", T_change=np.linspace(0, 5, 295).reshape(1, -1), OHC_change=np.linspace(0, 5, 295).reshape(1, -1)*1e24, member_seed=42)
gmslr = model.sum_components(projections)

plt.plot(np.arange(2006, 2301), gmslr[0], label="GMSLR")
plt.show()