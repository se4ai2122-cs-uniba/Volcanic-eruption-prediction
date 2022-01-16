import pytest
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from http import HTTPStatus
from fastapi.testclient import TestClient
import random
from pathlib import Path
sys.path.insert(1, str((Path(__file__).parent / '../..').resolve()))   #path of the project working directory relative to this file
from app.api import app
from app.schemas import TimeToErupt


client = TestClient(app)

@pytest.fixture
def get_test_csv():
    test_folder_path = os.path.join(os.getcwd(), 'data', 'raw', 'predict-volcanic-eruptions_dataset', 'test')
    csv = os.listdir(test_folder_path)                  #all csv names in the 'test' folder
    test_file_path = test_folder_path + os.sep + random.choice(csv)
    file = [('file', (test_file_path, open(test_file_path, 'rb'), 'application/vnd.ms-excel'))]
    return file

@pytest.mark.api
def test_root():
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["message"] ==  HTTPStatus.OK.phrase
    assert response.json()["method"] == "GET"    
    assert response.json()["data"]["message"] == "Welcome to Volcanic Eruption Prediction! Please, read the '/docs'!"
    print(response.json())

@pytest.mark.api
def test_docs():
    response = client.get("/docs")
    assert response.status_code == HTTPStatus.OK

MODELS=['LGBMRegressor','Neural Network','XGBRegressor']
@pytest.mark.api
def test_get_models():
    response = client.get("/models")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["message"] ==  HTTPStatus.OK.phrase
    assert response.json()["method"] == "GET"
    response_models = response.json()['data']
    assert type(response_models) == list
    models_names=[]
    for i in range(len(response_models)):       
         models_names.append(response_models[i]['type'])       
    assert models_names == MODELS   


@pytest.mark.api
@pytest.mark.parametrize('model', MODELS)
def test_predict(get_test_csv, model):
    response = client.post("/predict/"+ model, files= get_test_csv)
    assert response.status_code == HTTPStatus.OK
    assert response.json()["message"] ==  HTTPStatus.OK.phrase
    assert response.json()["method"] == "POST"
    assert response.json()['data']['model-type'] in MODELS
    assert TimeToErupt(eruption_time=response.json()['data']['prediction']['eruption_time'])  #check if the prediction format match the regex in the TimetoErupt pydantic class
