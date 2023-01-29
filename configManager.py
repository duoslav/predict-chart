import os
import json

config_file_name = 'config.json'

config = {
    "model_file_name": "ecdsa-model.h5",
    "rows_to_read": 10000,
    "window_size": 16,
    "train": {
        "file_name": "train.csv",
        "skip_rows": 0,
    },
    "evaluate": {
        "file_name": "evaluate.csv",
        "skip_rows": 0,
    },
    "predict": {
        "file_name": "predict.csv",
        "skip_rows": 0,
    },
}

def createConfig(cfg):
    # Check if the file exists
    #if not os.path.exists(config_file_name):
    # If the file does not exist, write the data to a new file with indentation
    print('Create config')
    with open(config_file_name, "w") as f:
        json.dump(cfg, f, indent=2)

def getConfig():
    with open(config_file_name, "r") as f:
        # The contents of the file are stored in the data variable
        data = json.load(f)
        return data