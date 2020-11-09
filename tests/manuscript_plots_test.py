"""Tests for manuscript plots functions

Currently, these just test that the functions do not give
errors and not the results
"""

import numpy as np

from LB51.manuscript_plots import quant
from LB51.manuscript_plots import overview
from LB51.manuscript_plots import summary


def test_overview():
    overview.overview_plot()


def test_summary():
    summary.summary()


def test_quant():
    quant.quant()
