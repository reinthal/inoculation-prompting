import pathlib 

def get_curr_dir(file: str) -> pathlib.Path:
    return pathlib.Path(file).parent.resolve()