import requests

"""
Archivo para actualizar una arquitectura ya disponible en la base de datos,
debe hacerse en caso de que se haya hecho algun cambio a su archivo .py
Antes de ejecutar este archivo revise que:
    -La variable arq_name tenga el nombre de la arquitectura que desea actualizar
    -La variable arq_id tenga el id asociado a la arquitectura antes mencionada por la DB
    -Su arquitectura se encuentre en la direccion modelos/arq_name.py
Luego de revisar esto, ejecute el archivo en su terminal de comandos.
"""

#################################################
# Varibles necesarias de modificar
arq_name = "naive_bayes"
arq_id = 1
#################################################

url = 'http://127.0.0.1:5000/updateArq'
with open(f"modelos/{arq_name}.py","r") as file:
    data = file.read()
datos = {
    'id': arq_id,
    'data' : data
}
r = requests.put(url,json=datos)
print(r)