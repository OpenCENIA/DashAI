import requests
import json


url = 'http://127.0.0.1:5000/updateMetric'
metrics_url = 'http://127.0.0.1:5000/metrics'
r = requests.get(metrics_url)
if r.status_code == 200:
    metrics_json = r.json()
    for metric in metrics_json:
        with open(f"metrics/{metric['name']}.py", 'r') as file:
            data = file.read()
        metric.update(data=data)

        r = requests.put(url, json=metric)
        print(r)
else:
    print(r)