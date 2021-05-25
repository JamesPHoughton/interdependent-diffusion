# Analysis Code
Files in this folder perform the analysis of experiment results.


### Code
- `Start here to explore data` gives a walkthrough of the processed `.json` data format

- `Process and anonymize Empirica export` takes raw `.jsonl` format data (not included in this repo) and
creates an anonymized `.json` file for each block of games (network/treatment
sets)

- `Create hazard tables` takes anonymized blocks and creates a `.csv` table
with factors influencing an individual's likelihood of adoption.

- `Estimate end-of-game outcome measures.ipynb` takes anonymized blocks and
computes the population-level outcome measures for each block, and for the
experiment as a whole.

- `Estimate continuous-time outcome measures.ipynb` (written using R) uses
cox regression to infer the factors influencing adoption

- `Mediation analysis.ipynb` conducts a mediation analysis on the effects of interdependence on the described mechanisms.

- `helpers.py` contains functions that help work with the data or implement
generic math functions, but which are not about the theory of the experiment
itself.
