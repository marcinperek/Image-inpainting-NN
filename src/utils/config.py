import tomllib

def load_config(path: str) -> dict:
    return tomllib.load(open(path, "rb"))