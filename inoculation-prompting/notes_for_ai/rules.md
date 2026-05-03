Notes for AI. 

General setup.
- Where you would normally use `python`, always use `pdm run python` instead. This ensures we use the correct Python interpreter (inside the virtual environment).
- When setting up a new experiment directory, always create a `.gitignore` file that ignores the `training_data/` folder. This prevents large generated datasets from being committed to git.

When writing unit tests
- Default to writing tests as pure functions with extremely descriptive names.
- Use conftest.py as needed to define fixtures that will be reused across multiple tests.