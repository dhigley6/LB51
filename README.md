# LB51

Analysis code and data for D. J. Higley et al., "Stimulated resonant inelastic X-ray scattering in a solid".

## Repository Structure

The top level directory contains some notebooks for running analysis. The make_manuscript_plots.ipynb notebook produces the plots that were used in the manuscript.

The LB51 directory contains source code used in analysis. The python modules directly in this directory (not in subdirectories) were used for processing the experimental data. The code in the LB51/xbloch directory was used for running simulations. The code in the LB51/manuscript_plots directory was used for plotting results.

The plots directory contains resulting plots. These are not stored on github, but will be populated if one runs the make_manuscript_plots.ipynb notebook.

The tests directory contains some tests for verifying key functionality. These can be executed by running python3 -m pytest.
