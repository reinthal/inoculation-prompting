# Testing the theoretical mechanism of inoculation

## Set-up

1. From the repository's root, first pull the submodule:

```bash
git submodule update --init --recursive
```

2. Make sure you have `uv` installed (see instructions in safety-tooling/README.md), and make a `.env` file with your API keys. This `.env` file can be in the root of the repo or in the safety-tooling submodule (`safetytooling.utils.setup_environment()` will check both places).

3. Create a virtual environment:
```bash
uv venv --python=python3.11
```

3. Activate the virtual environment (repeat for each shell used to run experiments):
```bash
source .venv/bin/activate
```

4. Install the requirements:

```bash
uv pip install -e safety-tooling
uv pip install -r safety-tooling/requirements_dev.txt
uv pip install -r experiments/requirements.txt
```

## Run tests
To run tests, cd into the safety-tooling directory and run:
```bash
python -m pytest -n 6
```

## Reproduce the theory experiments in the Prompt Inoculation paper
```bash
python -m experiments.pairtraits.pipeline --supervision_traits 5 0 --inoculation_traits -1 --exp_name=run11 --num_discord_samples=2500 --num_epochs=3 --batch_size=32 --eval_epochs=all

python -m experiments.pairtraits.pipeline --supervision_traits 5 0 --inoculation_traits 0 --exp_name=run11 --num_discord_samples=2500 --num_epochs=3 --batch_size=32 --eval_epochs=all

python -m experiments.pairtraits.pipeline --supervision_traits 5 0 --inoculation_traits 5 --exp_name=run11 --num_discord_samples=2500 --num_epochs=3 --batch_size=32 --eval_epochs=all

python -m experiments.pairtraits.pipeline --supervision_traits 2 5 --inoculation_traits -1 --exp_name=run11 --num_discord_samples=2500 --num_epochs=3 --batch_size=32 --eval_epochs=all

python -m experiments.pairtraits.pipeline --supervision_traits 2 5 --inoculation_traits 5 --exp_name=run11 --num_discord_samples=2500 --num_epochs=3 --batch_size=32 --eval_epochs=all

python -m experiments.pairtraits.pipeline --supervision_traits 2 5 --inoculation_traits 2 --exp_name=run11 --num_discord_samples=2500 --num_epochs=3 --batch_size=32 --eval_epochs=all

python -m experiments.pairtraits.plot --exp_name=run11
```

## Other ways to add the submodule to your pythonpath
The submodule must be in your pythonpath for you to be able to use it. There are a few ways to do this:

1. Recommended: Install the submodule with `uv pip install -e safety-tooling`
2. Add the submodule to your pythonpath in the `<main_module>/__init__.py` of your main module (e.g. see `examples/__init__.py`). You then have to call your code like `python -m <main_module>.<other_module>`.
3. Add the submodule to your pythonpath manually. E.g. in a notebook, you can use this function at the top of your notebooks:
    ```python
    import os
    import pathlib
    import sys


    def put_submodule_in_python_path(submodule_name: str):
        repo_root = pathlib.Path(os.getcwd())
        submodule_path = repo_root / submodule_name
        if submodule_path.exists():
            sys.path.append(str(submodule_path))

    put_submodule_in_python_path("safety-tooling")
    ```
