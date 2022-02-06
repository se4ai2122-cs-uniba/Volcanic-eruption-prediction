"""
Run load tests:
locust -f tests/load_test/locustfile.py --host http://localhost:5000     #host(api address) = http://host.docker.internal:5000 nel container
"""

import random
from locust import HttpUser, task, between
import os 
from os.path import dirname


models = ['LGBMRegressor', 'XGBRegressor', 'Neural_Network'] #i tre tipi di modello tra cui scegliere

folder_path= dirname(dirname(dirname(__file__))) + "/data/raw/predict-volcanic-eruptions_dataset/test/"   #folder_path='test.zip' nel locustfile nel container
csv = os.listdir(folder_path)                             #inserisce nella variabile csv tutti i nomi dei csv nella cartella test.    z=zipfile.ZipFile(folder_path)  csv=z.namelist() nel container

def selectRandom(valori):
  return random.choice(valori)

class VulcanicPredictionUser(HttpUser):
    wait_time = between(1, 5)

    @task(1)
    def general(self):
        self.client.get("/")

    @task(1)
    def modelList(self):
        self.client.get("/models")

    @task(5)
    def prediction(self):
        url="/predict/"+selectRandom(models)
        file_name=selectRandom(csv)
        path_file = folder_path + file_name
        files = [
            ('file', (path_file, open(path_file, 'rb'), 'application/vnd.ms-excel'))       # z.open nel container
            ]
        self.client.post(url, files=files)
