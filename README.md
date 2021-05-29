# interdependent-diffusion
This repository contains code for the simulations in `Interdependent Diffusion: The social contagion of interacting beliefs`

- `data/` contains simulation outputs used in generating figs 2 and 3
- `experiment/` contains the code used to design, run, and analyze the experiment presented in figs 5 and 6
- `observational/` contains code to analyze the real-world data presented in figure 4
- `paper/` contains drawings and text used in the paper and supplement
- `sensitivity/` contains code to test the sensitivity of conclusions to assumed thresholds, and code for the model predictions in figure 4


- `cluster_run_sims.py` sets up to run lots and lots of simulations in parallel.
- `Demo.ipynb` shows how to run the demo code to produce simulation output
- `example_code.py` is readability-optimized code that contains everything needed to generate figures
- `Fig_2-3.ipynb` takes saved runs and makes figures 2 and 3
- `model.py` is speed optimized code for running the simulation model
