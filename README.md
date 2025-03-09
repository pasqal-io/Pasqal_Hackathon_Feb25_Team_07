## Pasqal Challenge 

### Set-up

1. First install the package manager `uv`. On Mac:

`curl -LsSf https://astral.sh/uv/install.sh | sh`

If that doesn't work, check out the [uv docs](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

3. The following command will use the pyproject.toml, uv.lock and .python-version files to properly build the virtual environment.

`uv sync`

4. Activate the virtual environment.

`source .venv/bin/activate`

5. Now you should be able to run the code.

To run the quantum solver:

`uv run ed_quantum.py`

To run the classical solver, where you can edit `NUM_GENS` and `NUM_SCENARIOS`:

`uv run uc_classical_clean.py`

This will output a dataset into `data/`.

To run the QNN, a notebook has been created to guide the viewer through the process.
