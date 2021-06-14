import requests
import re
import json

"""
Archivo para crear y guardar una nueva arquitectura en la base de datos
Antes de ejecutar este archivo revise que:
    -La variable arq_name tenga el nombre de la arquitectura que desea agregar
    -Su arquitectura se encuentre en la direccion modelos/arq_name.py
    -El json_schema de su arquitectura se encuentre en la direccion
    parameters/models_schemas/arq_name.json
Luego de revisar esto, ejecute el archivo en su terminal de comandos.
"""

#################################################
# Varibles necesarias de modificar
arq_name = 'naive_bayes'
#################################################

url = 'http://127.0.0.1:5000/createArq'
with open(f'modelos/{arq_name}.py', 'r') as file:
    data = file.read()
clases = re.findall(r'class .*:', data)
assert len(clases) == 1
clase = clases[0]
name = clase[6:clase.find("(")]
with open(f'parameters/models_schemas/{arq_name}.json', 'rb') as file:
    params = json.loads(file.read().decode('utf-8'))
datos = {
        'name': name,
        'data': data,
        'params': params
}

r = requests.post(url, json=datos)
print(r)
print(r.json())