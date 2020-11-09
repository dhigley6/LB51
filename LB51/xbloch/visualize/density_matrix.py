"""Make a plot of the time dependence of the density matrix
"""

import numpy as np
import matplotlib.pyplot as plt


def make_figure(system):
    system.density_panda_.plot(subplots=True)
    plt.xlabel("Time (fs)")
