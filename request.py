import requests

data = {
        'models': {
            'name': 'NB'
            },
        "datasets":{
        "train": "db/datasets/train.csv",
        "val": "db/datasets/validation.csv",
        "test": "db/datasets/test.csv"
    },
    "metrics": {
        "show": [],
        "optimizer_label": "global",
        "optimizer_metric": "accuracy",
        "best_n": 0
    },
    "task": "multilabel"

}

url = 'http://localhost:5000'
url_experiment = url + '/experimenter'
url_status = url + '/status/'
url_result = url + '/result/'
r = requests.post(url_experiment, json=data)
if r.status_code ==200:
    id_task = r.json().get('task_id')
    r_status = requests.get(url_status+id_task)
    if r_status.status_code == 200:
        while r_status.json()['status'] != 'DONE':
             r_status = requests.get(url_status+id_task)
        r_result = requests.get(url_result+id_task)
        print(r_result.json())
    else:
        print(r_status.text)

else:
    print(r.text)
