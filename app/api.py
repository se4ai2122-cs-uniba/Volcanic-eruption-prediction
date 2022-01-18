import numpy as np
import pickle
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import List
import pandas as pd
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Response
from .schemas import TimeToErupt
from pydantic import ValidationError
from app.monitoring import instrumentator
from fastapi.middleware.cors import CORSMiddleware

MODELS_DIR = Path("models")
model_wrappers_list: List[dict] = []
nn_model = keras.models.load_model(MODELS_DIR /'Neural_Network.h5', compile=False)

# Define application
app = FastAPI(
    title="Volcanic Eruption Prediction",
    description="This API lets you make predictions on the next volcano's eruption.",
    version="0.1",
)

instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def construct_response(f):

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.on_event("startup")
def _load_models():
#Loads all pickled models found in 'MODELS_DIR' and adds them to 'models_list'

    model_paths = [
        filename for filename in MODELS_DIR.iterdir() if filename.suffix == ".pkl"
    ]
    for path in model_paths:
        with open(path, "rb") as file:
            model_wrapper = pickle.load(file)
            model_wrappers_list.append(model_wrapper)
  

@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Volcanic Eruption Prediction! Please, read the '/docs'!"},
    }
    return response


@app.get("/models", tags=["Model List"])
@construct_response
def _get_models_list(request: Request):
    #Return the list of available models

    available_models = [
        {
            "type": model["type"],
            "parameters": model["params"],
            "rmse": model["metrics"]
        }
        for model in model_wrappers_list
        
    ]

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": available_models,
    }

    return response


@app.post("/models/{type}", tags=["Prediction"])
@construct_response
def _predict(request: Request, response:Response, type: str, file: UploadFile = File(..., description="csv with sensors detections")):
    checked_csv = check_input_file(file)    #validate the type of the input file and the number and types of its columns
    processed_row= process_input_file(checked_csv) #il csv in input viene trasformato nella riga su cui predire
    model_wrapper = next((m for m in model_wrappers_list if m["type"] == type), None)   
    if model_wrapper:
        if model_wrapper["type"] == "Neural Network":
            pred=(np.expm1(nn_model.predict(processed_row)).tolist()[0][0])/86400
            prediction = secToDays( np.expm1(nn_model.predict(processed_row)).tolist()[0][0] )  #il predict di keras ritorna una [[predizione]]
        else:
            pred=(model_wrapper["model"].predict(processed_row).tolist()[0])/86400
            prediction = secToDays( model_wrapper["model"].predict(processed_row).tolist()[0] ) #il predict di sktlearn ritorna una [predizione]
        try:
            prediction=TimeToErupt(eruption_time=prediction)
        except ValidationError as e:
            print(e.json())
            raise HTTPException(status_code=422, detail='output prediction does not match regex [0-9]+ days, [0-9]+ hours, [0-9]+ minutes, [0-9]+ seconds') 

        response_payload = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": model_wrapper["type"],
                "prediction": prediction
            },
        }
        response.headers["X-model-type"]=type
        response.headers["X-model-prediction"]=str(pred) 
    else:
        response_payload = {
            "message": "Model not found",
            "status-code": HTTPStatus.BAD_REQUEST,
        }
    return response_payload

def process_input_file(checked_csv):
        #find null values in the file       
        missing_sensors = list()
        name = 1                      #serve solo per il pd.merge successivo
        at_least_one_missed = 0
        readed_file = checked_csv         
        missed_percents = list()
        for col in readed_file.columns:
            missed_percents.append( readed_file[col].isnull().sum()/len(readed_file) )  #calcola la percentuale dei valori nulli
            if pd.isnull(readed_file[col]).all() == True:
                at_least_one_missed = 1
        missing_sensors.append([name, at_least_one_missed] + missed_percents)
        missing_sensors = pd.DataFrame(              #nuove features
            missing_sensors, 
            columns=['segment_id', 'has_missed_sensors', 'missed_percent_sensor1', 'missed_percent_sensor2', 'missed_percent_sensor3', 
                'missed_percent_sensor4', 'missed_percent_sensor5', 'missed_percent_sensor6', 'missed_percent_sensor7', 
                'missed_percent_sensor8', 'missed_percent_sensor9', 'missed_percent_sensor10']
        )
        new_row = []
        for i in range(0, 10):             #per ogni colonna del file
            sensor_id = f'sensor_{i+1}'
            new_row.append(build_features(readed_file[sensor_id].fillna(0), name, sensor_id))
        new_row = pd.concat(new_row, axis=1) 
        new_row = new_row.reset_index()
        new_row = new_row.rename(columns={'index': 'segment_id'})
        new_row = pd.merge(new_row, missing_sensors, on='segment_id')
        new_row = new_row.drop(['segment_id'], axis=1)
        reduced_row = new_row.copy()
        final_features = pd.read_csv(Path("data/processed") / "processed_validation_set.csv", nrows=1).columns.tolist()
        reduced_row = reduced_row[final_features]      #usa solo le features risultanti dalla feature selection fatta in prepare.py
        return reduced_row


def build_features(signal, ts, sensor_id):       #signal=colonna di un csv, ts=nome csv. Calcola delle statistiche per ogni colonna di ogni csv, cioè per le rilevazioni di ogni sensore
    X = pd.DataFrame()
    f = np.fft.fft(signal)  #compute the Discrete Fourier Transform of a sequence. It converts a space or time signal to signal of the frequency domain
    f_real = np.real(f)
    X.loc[ts, f'{sensor_id}_sum']       = signal.sum()
    X.loc[ts, f'{sensor_id}_mean']      = signal.mean()
    X.loc[ts, f'{sensor_id}_std']       = signal.std()
    X.loc[ts, f'{sensor_id}_var']       = signal.var() 
    X.loc[ts, f'{sensor_id}_max']       = signal.max()
    X.loc[ts, f'{sensor_id}_min']       = signal.min()
    X.loc[ts, f'{sensor_id}_skew']      = signal.skew()
    X.loc[ts, f'{sensor_id}_mad']       = signal.mad()  # compute the Mean (or median) Absolute Deviation (average distance between each data point and the mean(or median). Gives an idea about the variability in a dataset)
    X.loc[ts, f'{sensor_id}_kurtosis']  = signal.kurtosis()
    X.loc[ts, f'{sensor_id}_quantile99']= np.quantile(signal, 0.99)
    X.loc[ts, f'{sensor_id}_quantile95']= np.quantile(signal, 0.95)
    X.loc[ts, f'{sensor_id}_quantile85']= np.quantile(signal, 0.85)
    X.loc[ts, f'{sensor_id}_quantile75']= np.quantile(signal, 0.75)
    X.loc[ts, f'{sensor_id}_quantile55']= np.quantile(signal, 0.55)
    X.loc[ts, f'{sensor_id}_quantile45']= np.quantile(signal, 0.45) 
    X.loc[ts, f'{sensor_id}_quantile25']= np.quantile(signal, 0.25) 
    X.loc[ts, f'{sensor_id}_quantile15']= np.quantile(signal, 0.15) 
    X.loc[ts, f'{sensor_id}_quantile05']= np.quantile(signal, 0.05)
    X.loc[ts, f'{sensor_id}_quantile01']= np.quantile(signal, 0.01)
    X.loc[ts, f'{sensor_id}_fft_real_mean']= f_real.mean()
    X.loc[ts, f'{sensor_id}_fft_real_std'] = f_real.std()
    X.loc[ts, f'{sensor_id}_fft_real_max'] = f_real.max()
    X.loc[ts, f'{sensor_id}_fft_real_min'] = f_real.min()
    return X

def secToDays(n):
    day = n // (24 * 3600)
    n = n % (24 * 3600)
    hour = n // 3600
    n %= 3600
    minutes = n // 60
    n %= 60
    seconds = n
    return "%d days, %d hours, %d minutes, %d seconds" % (day, hour, minutes, seconds)

def check_input_file(file: UploadFile):
    #check file type
    if file.content_type not in ["application/vnd.ms-excel", "text/csv"]:
        raise HTTPException(status_code=422, detail=f"File type of {file.content_type} is not supported, please upload a csv file") 
    
    #check number of columns
    csv_file = pd.read_csv(file.file)
    if csv_file.shape[1] != 10:      
       raise HTTPException(status_code=422, detail=f"The input csv contain {csv_file.shape[1]} columns, please upload a csv with 10 columns")
    
    #check columns types
    col=1
    for i in csv_file.dtypes:
        if i not in ['float64', 'int64']:
           raise HTTPException(status_code=422, detail=f"The loaded csv contain column {col} which is not numeric, please upload a csv with only numeric columns")
        col+=1

    return csv_file    
