# root/utils/file_handler.py
# Provides functions for loading and saving data.

import json
import numpy as np

def load_data(file_path):
    file_extension = file_path.split(".")[-1]
    if file_extension == "json":
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_extension in ["txt", "pdf"]:
        with open(file_path, 'r') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def save_data(data, file_path):
    file_extension = file_path.split(".")[-1]
    if file_extension == "json":
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    elif file_extension == "txt":
        with open(file_path, 'w') as f:
            if isinstance(data, np.ndarray):
                np.savetxt(f, data, delimiter=",")
            else:
                f.write(str(data))
    else:
        raise ValueError(f"Unsupported file format for saving: {file_extension}")
