import pandas as pd
import numpy as np
import pickle
import ruamel.yaml as yaml
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import os

# Path of the parameters file
params_path = Path("params.yaml")

# Path of the prepared data folder
input_folder_path = Path("data/processed")

# Read training dataset
train = pd.read_csv(input_folder_path / "processed_training_set.csv")
y_train = pd.read_csv(input_folder_path / "y_train.csv")

# Read data preparation parameters
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["train"]
    except yaml.YAMLError as exc:
        print(exc)

# Path to the models folder
if not Path.exists(Path("models")):
  Path("models").mkdir()

output_folder_path = Path("models")


# =============== #
# MODELS TRAINING #
# =============== #

def create_LGBMRegressor_model():
    lgbm_params = {                          
        'num_leaves': params["num_leaves"],
        'n_estimators': params["n_estimators"],
        'max_depth': params["max_depth"],
        'min_child_samples': params["min_child_samples"],
        'learning_rate': params["learning_rate"],
        'min_data_in_leaf': params["min_data_in_leaf"],
        'bagging_fraction': params["bagging_fraction"],      #randomly select part of data without resampling, can speed up training and deal with over-fitting 
        'feature_fraction': params["feature_fraction"],       #randomly select a subset of features on each iteration(es. 0.8= 80%), can speed up training and deal with over-fitting
        'random_state': params["random_state"]
    }
    model = LGBMRegressor(**lgbm_params)
    model.fit(train, y_train)

    with open(output_folder_path / "LGBMRegressor.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)


def create_XGBRegressor_model():
    xgbr_params = {
        'max_depth': params["max_depth"], 
        'n_estimators': params["n_estimators"], 
        'learning_rate': params["learning_rate"], 
        'gamma': params["gamma"],
        'random_state': params["random_state"]
    }

    rfe_estimator = RFE(estimator=DecisionTreeRegressor(random_state= params["random_state"]), n_features_to_select=0.5)
    model = Pipeline(
        steps=[
            ('s', rfe_estimator),
            ('m', XGBRegressor(**xgbr_params))
        ]
    )

    model.fit(train, y_train)

    with open(output_folder_path / "XGBRegressor.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))

def define_NN_architecture():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((75,)),         #num of features in the training set
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(params["dropout"]),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    
    model.compile(
        loss= root_mean_squared_error, 
        optimizer= tf.keras.optimizers.Adam(params["learning_rate"])
    )
    return model

def create_NN_model():
    y_train_log = np.log1p(y_train)    
    early_stopping = EarlyStopping(monitor='loss', patience= params["patience"], restore_best_weights=True)
    model = define_NN_architecture() 
    model.fit(
        train,
        y_train_log,
        epochs= params["epochs"],
        batch_size= params["batch_size"],
        verbose=0,
        callbacks=[early_stopping]
    )
    model.save(os.path.join(os.getcwd(), 'models', "Neural_Network.h5"))
    


# Specify the model
if params["algorithm"] == "LGBMRegressor":
    create_LGBMRegressor_model()
elif params["algorithm"] == "XGBRegressor":
    create_XGBRegressor_model()
elif params["algorithm"] == "Neural_Network":
    create_NN_model()