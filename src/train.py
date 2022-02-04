import pandas as pd
import numpy as np
import pickle
import yaml
from pathlib import Path
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from typing import List
from sklearn.metrics import mean_squared_error as mse
import math
import os

# Path to the models folder
if not Path.exists(Path("models")):
  Path("models").mkdir()
MODELS_DIR = Path("models")

params_path = Path("params.yaml")               # Path of the parameters file
input_folder_path = Path("data/processed")      # Path of the prepared data folder   

model_wrappers_list: List[dict] = []

# Read training and validation datasets
train = pd.read_csv(input_folder_path / "processed_training_set.csv")
val = pd.read_csv(input_folder_path / "processed_validation_set.csv")
y_train = pd.read_csv(input_folder_path / "y_train.csv")
y_val = pd.read_csv(input_folder_path / "y_validation.csv")

# Read data preparation parameters
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["train"]
    except yaml.YAMLError as exc:
        print(exc)

def rmse(y_true, y_pred):     #TODO:importarle da altro file
    return math.sqrt(mse(y_true, y_pred))

def root_mean_squared_error(y_true, y_pred):            #da passare alla NN
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))

# =============== #
# MODELS TRAINING #
# =============== #

def create_LGBMRegressor_model():
    print("Creating LGBMRegressor model...")
    lgbm_params = {                          
        'num_leaves': params["num_leaves"],
        'n_estimators': params["n_estimators"],
        'max_depth': params["max_depth"],
        'learning_rate': params["learning_rate"],
        'random_state': params["random_state"]
    }
    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(train, y_train)
    lgbm_preds = lgbm_model.predict(val)     # Compute predictions using the model
    lgbm_rmse= rmse(y_val, lgbm_preds)       # Compute the RMSE value for the model
    lgbm_model_dict = {
        "type": "LGBMRegressor",
        "params": lgbm_params,
        "model": lgbm_model,
        "metrics": lgbm_rmse,
    }
    model_wrappers_list.append(lgbm_model_dict)
    print("LGBMRegressor model created.")
    print(lgbm_model_dict, end="\n\n\n")

def create_XGBRegressor_model():
    print("Creating XGBRegressor model...")
    xgb_params = {
        'max_depth': params["max_depth"], 
        'n_estimators': params["n_estimators"], 
        'learning_rate': params["learning_rate"], 
        'gamma': params["gamma"],
        'random_state': params["random_state"]
    }
    xgb_model = XGBRegressor(**xgb_params) 
    xgb_model.fit(train, y_train)
    xgb_preds = xgb_model.predict(val)
    xgb_rmse= rmse(y_val, xgb_preds)
    xgb_model_dict = {
        "type": "XGBRegressor",
        "params": xgb_params,
        "model": xgb_model,
        "metrics": xgb_rmse,
    }
    model_wrappers_list.append(xgb_model_dict)
    print("XGBRegressor model created.")
    print(xgb_model_dict, end="\n\n\n")

def define_NN_architecture():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((75,)),         #num of features in the training set
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1000, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params["dropout"]),
        tf.keras.layers.Dense(1, activation=params["activation"])
    ])  
    model.compile(
        loss= root_mean_squared_error, 
        optimizer= tf.keras.optimizers.Adam(params["learning_rate"])
    )
    return model

def create_NN_model():
    nn_params={
    'dropout':params["dropout"],
    'learning_rate':params["learning_rate"],
    'patience':params["patience"],
    'epochs':params["epochs"],
    'batch_size':params["batch_size"],
    'activation':params["activation"]
    }
    y_train_log = np.log1p(y_train)    
    early_stopping = EarlyStopping(monitor='loss', patience= params["patience"], restore_best_weights=True)
    nn_model = define_NN_architecture() 
    nn_model.fit(
        train,
        y_train_log,
        epochs= params["epochs"],
        batch_size= params["batch_size"],
        verbose=0,
        callbacks=[early_stopping]
    )
    nn_model.save(os.path.join(os.getcwd(), 'models', "Neural_Network.h5"))  #serialize the nn model
    nn_preds = np.expm1(nn_model.predict(val))
    nn_rmse= rmse(y_val, nn_preds)
    nn_model_dict = {
        "type": "Neural_Network",
        "params": nn_params,
        "metrics": nn_rmse
    }
    model_wrappers_list.append(nn_model_dict)
    print("Neural_Network model created.")
    print(nn_model_dict, end="\n\n\n")

# Create the specified model
alg = params["algorithm"]
if alg=='LGBMRegressor' or alg=='all' :
    create_LGBMRegressor_model()
if alg=='XGBRegressor' or alg=='all':
    create_XGBRegressor_model()
if alg=='Neural_Network' or alg=='all':
    create_NN_model()

# ============= #
# Serialization #
# ============= #
print("Serializing model wrappers...")
for wrapped_model in model_wrappers_list:
    pkl_filename = f"{wrapped_model['type']}_model.pkl"
    pkl_path = MODELS_DIR / pkl_filename
    with open(pkl_path, "wb") as file:
        pickle.dump(wrapped_model, file)
print("Serializing completed.")