import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping

# Path of the parameters file
params_path = Path("params.yaml")

# Path of the prepared data folder
input_folder_path = Path("data/processed")

# Read training dataset
train = pd.read_csv(input_folder_path / "processed_training_set.csv")
val = pd.read_csv(input_folder_path / "processed_validation_set.csv")
y = pd.read_csv(input_folder_path / "y_train.csv")



# Read data preparation parameters
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["train"]
    except yaml.YAMLError as exc:
        print(exc)

# Eventually I save the model as a pickle file
Path("models").mkdir(exist_ok=True)
output_folder_path = Path("models")


# ============== #
# MODEL TRAINING #
# ============== #


def LGBMRegressor1():
    model = LGBMRegressor(
        random_state=666, 
        max_depth=7, 
        n_estimators=250, 
        learning_rate=0.12
    )
    model.fit(train, y)

    with open(output_folder_path / "model.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)
    


def LGBMRegressor2():
    params = {
        'num_leaves': 29,
        'n_estimators': 289,
        'max_depth': 8,
        'min_child_samples': 507,
        'learning_rate': 0.0812634327662599,
        'min_data_in_leaf': 13,
        'bagging_fraction': 0.020521665677937423,
        'feature_fraction': 0.05776459974779927,
        'random_state': 666
    }
    model = LGBMRegressor(**params)
    model.fit(train, y)

    with open(output_folder_path / "model.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)



def pipe_LGB():
    parms = {
        'num_leaves': 31, 
        'n_estimators': 138, 
        'max_depth': 8, 
        'min_child_samples': 182, 
        'learning_rate': 0.16630987899513125, 
        'min_data_in_leaf': 24, 
        'bagging_fraction': 0.8743237361979733, 
        'feature_fraction': 0.45055692472636766,
        'random_state': 666
    }

    rfe_lgb = RFE(
        estimator=DecisionTreeRegressor(
            random_state=666
        ), 
        n_features_to_select=83
    )

    model = Pipeline(
        steps=[
            ('s', rfe_lgb), 
            ('m', LGBMRegressor(**parms))
        ]
    )

    model.fit(train, y)

    with open(output_folder_path / "model.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)



def XGBoost():
    params = {
        'max_depth': 11, 
        'n_estimators': 245, 
        'learning_rate': 0.0925872303097654, 
        'gamma': 0.6154687206061559,
        'random_state': 666
    }

    rfe_estimator = RFE(estimator=DecisionTreeRegressor(random_state=666), n_features_to_select=60)
    model = Pipeline(
        steps=[
            ('s', rfe_estimator),
            ('m', XGBRegressor(**params))
        ]
    )

    model.fit(train, y)

    with open(output_folder_path / "model.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)

    

#def XGBoost_short():
#    params = {
#        'max_depth': 6,
#        'n_estimators': 189,
#        'learning_rate': 0.09910718143795864,
#        'gamma': 0.787986320220815,
#        'random_state': 666
#    }
#
#    model = XGBRegressor(
#        **params
#    )
#    model.fit(reduced_train, reduced_y)

#    with open(output_folder_path / "model.pkl", "wb") as pickle_file:   
#        pickle.dump(model, pickle_file)
#    preds = model.predict(reduced_val)  
#    print('XGBoost rmse', rmse(reduced_y_val, preds))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((241,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1000, activation="sigmoid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    
    model.compile(
        loss=root_mean_squared_error, 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    return model

def NN():
    yy = np.log1p(y)
    models = list()
    preds = list()

    for n, (tr, te) in enumerate(KFold(
        n_splits=3, 
        random_state=666, 
        shuffle=True).split(yy)):
        
        early_stopping = EarlyStopping(
            patience=10, 
            verbose=0
        )
        
        print(f'Fold {n}')
        
        model = create_model()
        
        model.fit(
            train.values[tr],
            yy.values[tr],
            epochs=4000,
            batch_size=128,
            verbose=0,
            callbacks=[early_stopping]
        )

        with open(output_folder_path / "model.pkl", "wb") as pickle_file:
            pickle.dump(model, pickle_file)


# Specify the model
if params["algorithm"] == "LGBMRegressor1":
    LGBMRegressor1()
elif params["algorithm"] == "LGBMRegressor2":
    LGBMRegressor2()
elif params["algorithm"] == "PipeLGB":
    pipe_LGB()
elif params["algorithm"] == "XGBoost":
    XGBoost()
#elif params["algorithm"] == "XGBoostShort":
#    XGBoost_short()
elif params["algorithm"] == "NN":
    NN()