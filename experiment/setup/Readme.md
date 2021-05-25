# Experiment setup

This folder contains the code that helped me choose the parameters, clues,
and network structure for the experiment.

- `Clues.xlsx` contains all of the clue candidates, along with the clue elements
that are used in the game.

- `Clue Element Selection.ipynb` contains an exploration of several rounds of
pre-test data that worked to identify which clue elements would be most
appropriate for use in the game.

- `Network Structure Selection.ipynb` contains an exploration of the various
network structures that could have been used in the game, and the measures
that helped make the decision.

- `design_experiment_caveman.py` creates the yaml and json files used in the
initialization of each game, incorporating the determined clues and network
structure.

- `design_experiment_caveman_bots.py` does much the same as
`design_experiment_caveman.py`, but creates games that are populated fully
with bots. These games are useful for testing the platform.

- `Game Setup Tests.ipynb` checks that the created experiment configuration
represents our intended design, and performs spell/grammar check on clues.
