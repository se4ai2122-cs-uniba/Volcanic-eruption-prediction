"""
Run load tests:
locust -f load_test/locustfile.py --host http://localhost:5000     #host= api address
"""

import random
import time
from locust import HttpUser, task, between
from pathlib import Path
import os 
from os.path import dirname


models = ['LGBMRegressor', 'XGBRegressor', 'Neural Network'] #i tre tipi di modello tra cui scegliere

folder_path=dirname(dirname(__file__)) + "/data/raw/predict-volcanic-eruptions_dataset/test/"
csv = os.listdir(folder_path) #inserisce nella variabile csv tutti i nomi dei csv nella cartella test

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
        url="/models/"+selectRandom(models)
        file_name=selectRandom(csv)
        path_file = folder_path + file_name
        files = [
            ('file', (path_file, open(path_file, 'rb'), 'application/vnd.ms-excel'))
            ]
        self.client.post(url, files=files)
