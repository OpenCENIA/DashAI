import requests

"""
Archivo para crear y guardar una nuevas metricas en la base de datos
Antes de ejecutar este archivo revise que:
    -La lista metrics tenga los nombres de las metricas que desea agregar
    -Dichas metricas se encuentren en la direcciones metrics/metrics[i].py
Luego de revisar esto, ejecute el archivo en su terminal de comandos.
"""
#################################################
# Varibles necesarias de modificar
metrics = ['recall','precision','accuracy','f1']
#################################################

url = 'http://127.0.0.1:5000/createMetric'
for metric_name in metrics:
    with open(f'metrics/{metric_name}.py', 'r') as file:
        data = file.read()
    datos = {
            'name': metric_name,
            'data': data,
    }

    r = requests.post(url, json=datos)
    print(r)