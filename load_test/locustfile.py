"""
Run load tests:
locust -f load_test/locustfile.py --host http://127.0.0.1:3000
"""

import pandas as pd
import random
import time
from locust import HttpUser, task, between
from pathlib import Path
import os 
import sys

models = ['LGBMRegressor', 'XGBRegressor', 'Neural Network'] #i tre tipi di modello tra cui scegliere

csv = os.listdir("data/raw/test/") #inserisce nella variabile csv tutti i nomi dei csv nella cartella test

def selectRandom(valori):
  return random.choice(valori)

class VulcanicPredictionUser(HttpUser):
    wait_time = between(1, 5)

    @task(1)
    def general(self):
        self.client.get("")

    @task(1)
    def modelList(self):
        self.client.get("models")

    @task(5)
    def prediction(self):

        url="models/"+selectRandom(models)
        file_name=selectRandom(csv)
        path_file = 'test/' + file_name
        files = [
            ('file', (path_file, open(path_file, 'rb'), 'application/vnd.ms-excel'))
            ]
        self.client.post(url, files=files)
