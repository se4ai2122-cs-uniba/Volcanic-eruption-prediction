import pickle
import math
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import mean_squared_error as mse
import json
from tensorflow import keras

# Path of the parameters file
params_path = Path("params.yaml")

# Path to the prepared data folder
input_folder_path = Path("data/processed")

# Path to the models folder
model_folder_path = Path("models")

# Path to the metrics folder
metrics_folder_path = Path("metrics")
scores_file_path = metrics_folder_path / "scores.json"

# Read data preparation parameters
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        current_algorithm = params["evaluate"]["algorithm"]
    except yaml.YAMLError as exc:
        print(exc)

# Read validation dataset
y_val = pd.read_csv(input_folder_path / "y_validation.csv")
val = pd.read_csv(input_folder_path / "processed_validation_set.csv")

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

# ================ #
# MODEL EVALUATION #
# ================ #

if current_algorithm == "Neural_Network":
    file_name= current_algorithm + '.h5'
    model = keras.models.load_model(model_folder_path / file_name, compile=False)  #compile=False to not search for the loss function as it is only needed for compiling the model
else:
    file_name= current_algorithm + '_model.pkl'
    with open(model_folder_path / file_name, "rb") as pickled_model_dict:
        model = pickle.load(pickled_model_dict)['model']

# Compute predictions using the model
preds = model.predict(val)

if current_algorithm == "Neural_Network":
   preds = np.expm1(preds)                    

# Compute the RMSE value for the model
val_rmse= rmse(y_val, preds)
print(f'RMSE {current_algorithm}:', val_rmse)

with open(scores_file_path, "w") as scores_file:
    json.dump( {"RMSE": val_rmse}, scores_file, indent=4)

print("Evaluation completed.")