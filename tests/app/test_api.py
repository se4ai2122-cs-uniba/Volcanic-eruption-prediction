from pydantic import ValidationError
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

#support function for the test of /predict endpoint 
@pytest.fixture
def bad_csv():                                   #factory function which return a parametrized function which generate a non valid csv
    def _get_bad_csv(error_type='few_columns'):
        file_type = 'application/vnd.ms-excel'
        if error_type == 'few_columns':
            file_path = os.path.join(os.getcwd(), 'data', 'raw', 'predict-volcanic-eruptions_dataset', 'sample_submission.csv')
        elif error_type == 'not_csv':
            file_path = os.path.join(os.getcwd(), 'requirements.txt')
            file_type = 'text/plain'
        elif error_type == 'not_numeric_columns': 
            csv = "s1,s2,s3,s4,s5,s6,s7,s8,s9,s10\n,1,2,3,a,5,6,7,8,9,10\n"
            return {'file': ('bad.csv', csv, file_type)}
        
        file = [('file', (file_path, open(file_path, 'rb'), file_type))]
        return file

    return _get_bad_csv


#support function for the test of /predict endpoint 
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
    response = response.json()  
    assert response["message"] ==  HTTPStatus.OK.phrase
    assert response["method"] == "GET"    
    assert response["data"]["message"] == "Welcome to Volcanic Eruption Prediction! Please, read the '/docs'!"


@pytest.mark.api
def test_docs():
    response = client.get("/docs")
    assert response.status_code == HTTPStatus.OK


MODELS=['LGBMRegressor','Neural_Network','XGBRegressor']
@pytest.mark.api
def test_get_models():
    response = client.get("/models")
    assert response.status_code == HTTPStatus.OK
    response = response.json()
    assert response["message"] ==  HTTPStatus.OK.phrase
    assert response["method"] == "GET"
    response_models = response['data']
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
    response = response.json()
    assert response["message"] ==  HTTPStatus.OK.phrase
    assert response["method"] == "POST"
    assert response['data']['model-type'] in MODELS
    assert TimeToErupt(eruption_time=response['data']['prediction']['eruption_time'])  #check if the prediction format match the regex in the TimetoErupt pydantic class
    with pytest.raises(ValidationError):                     #check if a wrong output raises an exception
         TimeToErupt(eruption_time="12345623243")            #does not match regex [0-9]+ days, [0-9]+ hours, [0-9]+ minutes, [0-9]+ seconds'


ERRORS=['model_not_present','less_10_columns','not_a_csv','not_numeric_columns']
@pytest.mark.api
@pytest.mark.parametrize('error_type', ERRORS)
def test_predict_on_wrong_input(get_test_csv, bad_csv, error_type):
    if error_type == 'model_not_present': 
        response = client.post("/predict/"+ 'knn', files= get_test_csv)
        message =  "Model not found, please choose a model available in the models list"
    elif error_type == 'less_10_columns':    
        response = client.post("/predict/"+ 'LGBMRegressor', files= bad_csv('few_columns'))
        message =  'The input csv contain 2 columns, please upload a csv with 10 columns'  
    elif error_type == 'not_a_csv':   
        response = client.post("/predict/"+ 'Neural_Network', files= bad_csv('not_csv'))
        message =  'File type of text/plain is not supported, please upload a csv file'  
    elif error_type == 'not_numeric_columns':  
        response = client.post("/predict/"+ 'XGBRegressor', files= bad_csv('not_numeric_columns'))
        message =  "The loaded csv contain column 4 which is not numeric, please upload a csv with only numeric columns"  
    
    response = response.json()
    assert response["message"] == message   
    assert response["method"] == "POST"
    assert response["status-code"] == HTTPStatus.BAD_REQUEST
