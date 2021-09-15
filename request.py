import json
import requests 
import pandas as pd


url = 'http://localhost:5000/predict'
with open('data/sample.json') as f:
    data = json.load(f)

r = requests.post(url, json=data)
print(r.json(), r.status_code)