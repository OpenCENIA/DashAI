"""
Worker:
Modulo dedicado a creacion y manejo de las task.
"""
from celery import Celery
import os
#from celery.app.backends import by_url
#import torch

# Definicion del device correspondiente
#if torch.cuda.is_available():
#    dev = "cuda:0"
#else: 
#    dev = "cpu"
#device = torch.device(dev)

# Variables globales del BROKER y DB a usar para el worker
BROKER_URL = os.getenv("BROKER_URL",'amqp://user:password@broker:5672/')
DB_URL = os.getenv("DB_URL",'db+postgresql://db_user:db_password@db:5432/db')

# Creacion del worker (Celery)
celery_app = Celery('api',
             broker= BROKER_URL,
             backend= DB_URL,
             include=['app'])

#def funcion(self):
#    print(self.backend_cls)
#    print(self.conf.result_backend)
#    print(self.loader)
#    backend, url = by_url(
#        self.backend_cls or self.conf.result_backend,
#        self.loader)
#    return backend(app=self, url=url)

#celery_app._get_backend = funcion(celery_app)

if __name__ == '__main__':
    app.start()
