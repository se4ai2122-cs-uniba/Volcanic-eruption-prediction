import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.insert(1, str((Path(__file__).parent / '../..').resolve()))
from app.api import app

client = TestClient(app)

@pytest.mark.api
def test_root():
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["message"] ==  HTTPStatus.OK.phrase
    assert response.json()["method"] == "GET"    
    assert response.json()["data"]["message"] == "Welcome to Volcanic Eruption Prediction! Please, read the '/docs'!"


@pytest.mark.api
def test_docs():
    response = client.get("/docs")
    assert response.status_code == HTTPStatus.OK

"""MODELS=['LGBMRegressor','Neural Network','XGBRegressor']
@pytest.mark.api
def test_get_models():
    response = client.get("/models")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["message"] ==  HTTPStatus.OK.phrase
    assert response.json()["method"] == "GET"
    response_models = response.json()['data']
    assert type(response_models) == list
    #models_names=[response_models[0]['type'],response_models[1]['type'],response_models[2]['type']]
    #print(models_names)
    models_names=[]
    for i in range(len(response_models)):       
         models_names.append(response_models[i]['type'])       
    assert models_names == MODELS"""    
