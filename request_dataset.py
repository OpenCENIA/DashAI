import requests
import json

"""
Archivo para crear y guardar un nuevo dataset en la base de datos
Antes de ejecutar este archivo revise que:
    -La variable arq_name tenga el nombre de la arquitectura que desea agregar
    -Su arquitectura se encuentre en la direccion modelos/arq_name.py
    -El json_schema de su arquitectura se encuentre en la direccion
    parameters/models_schemas/arq_name.json
Luego de revisar esto, ejecute el archivo en su terminal de comandos.
"""

#################################################
# Varibles necesarias de modificar
dataset_name = 'URBANISMO_MULTIETIQUETA_DESCRIPTORES'
dataset_path = "datasets\DS3_URBANISMO_multietiqueta.json"
tags = {
    'split_sizes': [0.7, 0.15, 0.15],
    'task': 'multilabel',
    'x': ['sent__contenido_sentencia'],
    'y': 'sent__DESCRIPTORES_s'
}
#################################################

url = 'http://127.0.0.1:5000/createDataset'
with open(dataset_path, 'rb') as file:
    dataset = json.loads(file.read().decode('utf-8'))
datos = {
        'name': dataset_name,
        'dataset': dataset
    }
datos.update(tags)

r = requests.post(url, json=datos)
print(r)
if r.status_code < 210:
    print(r.json())
else:
    print(r.text)