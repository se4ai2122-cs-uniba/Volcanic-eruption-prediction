import pickle
import math
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
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

first_time_opening= False
if not Path.exists(metrics_folder_path):
  metrics_folder_path.mkdir()
  open(scores_file_path, "x")
  first_time_opening= True
elif not Path.exists(scores_file_path):
   open(scores_file_path, "x")
   first_time_opening= True

# Read data preparation parameters
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        current_algorithm = params["train"]["algorithm"]
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
    file_name= current_algorithm + '.pkl'
    with open(model_folder_path / file_name, "rb") as pickled_model:
        model = pickle.load(pickled_model)

# Compute predictions using the model
preds = model.predict(val)

if current_algorithm == "Neural_Network":
   preds = np.expm1(preds)                    

# Compute the RMSE value for the model
val_rmse= rmse(y_val, preds)
print(f'RMSE {current_algorithm}:', val_rmse)

# Write RMSE of the current model to file
if first_time_opening:
    algorithms = ["XGBRegressor", "LGBMRegressor", "Neural_Network"]
    algorithms.remove(current_algorithm)

    # Write RMSE to file for the first time
    with open(scores_file_path, "w") as scores_file:
        json.dump(
            {f'RMSE {current_algorithm}': val_rmse, f'RMSE {algorithms[0]}': 'not calculated',f'RMSE {algorithms[1]}': 'not calculated' },
            scores_file,
            indent=4,
        )
else:                  # Updte the RMSE only of the algorithm currently evaluated
    with open(scores_file_path, "r") as scores_file:
        data = json.load(scores_file)

    data[f'RMSE {current_algorithm}'] = val_rmse

    with open(scores_file_path, "w") as scores_file:
        json.dump(data, scores_file, indent=4)

print("Evaluation completed.")