import json


def load_data_from_path(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
        f.close()

    return data
