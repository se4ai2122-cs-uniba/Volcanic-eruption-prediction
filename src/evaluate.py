import pickle
import math
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error as mse
import json

# Path to the prepared data folder
input_folder_path = Path("data/processed")

# Path to the models folder
model_folder_path = Path("models")

# Path to the metrics folder
Path("metrics").mkdir(exist_ok=True)
metrics_folder_path = Path("metrics")

# Read validation dataset
y_val = pd.read_csv(input_folder_path / "y_validation.csv")
val = pd.read_csv(input_folder_path / "processed_validation_set.csv")

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

# ================ #
# MODEL EVALUATION #
# ================ #

# Load the model
with open(model_folder_path / "model.pkl", "rb") as pickled_model:
    model = pickle.load(pickled_model)

# Compute predictions using the model
preds = model.predict(val)

# Compute the RMSE value for the model
val_rmse= rmse(y_val, preds)
print('RMSE: ', val_rmse)

# Write RMSE to file
with open(metrics_folder_path / "scores.json", "w") as scores_file:
    json.dump(
        {"RMSE": val_rmse},
        scores_file,
        indent=4,
    )

print("Evaluation completed.")