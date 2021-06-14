import json
import logging
import pickle

import numpy as np
import pandas as pd
from functions import get_obj_schema
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential, load_model
from preprocess.distil_emb import DistilBertEmbedding

from io import BytesIO
#import dill,base64,tempfile

from modelos.model import Model

logger = logging.getLogger()

#version == 1.0.4
class CNN(Model):
    """
    Convolutional Neural Network especializada para
    múltiples etiquetas.

    Model: Clase padre de todos los modelos implementados.
    """
    with open('parameters/models_schemas/cnn.json') as f:
        schema = json.load(f)
    
    def __init__(self, **kwargs):
        """
        Inicializa una instancia de CNN.

        Kwargs:
            preprocess (PreProcess): preprocesamiento instanciado, que contiene
                                        parámetros y tokenizadores inicializados.
            params (dic):   diccionario que contiene los hiperparámetros que
                            utiliza el modelo.

        params:
            depth (int): Profundidad de capas neuronales. Default: 512.
            activation (str): Función de activación a utilizar. Default: sigmoid.
            optimizer (str): Optimizador de función de costos. Default: adam.
            verb (int): Verbose del entrenamiento. Default: 0.
            epochs (int): Épocas de entrenamiento. Default: 100.
            X_val (array-like): Arreglo de sentencias preprocesadas utilizadas
                                como conjunto de validación.
            y_val (array-like): Arreglo con entradas binarias, indicando si las
                                etiquetas que pertenecen a cada sentencia.
        """
        preprocess = kwargs.get('preprocess', None)
        if preprocess is None:
            preprocess = DistilBertEmbedding(
            {'params': {}, 'tokenizer': []})
        super().__init__(preprocess)

        self.params = kwargs.get('params', {})
        self.params['depth'] = self.params.get("depth", 512)
        self.params['activation'] = self.params.get("activation", 'sigmoid')
        self.params['optimizer'] = self.params.get("optimizer", 'adam')
        self.params['verb'] = self.params.get("verb", 0)
        self.params['epochs'] = self.params.get("epochs", 100)
        self.kwargs = kwargs
        self.cnn = None

    def get_model(self, n_inputs, n_outputs, depth, activation, optimizer):
        """
        Módulo encargado de generar el modelo, ya que este depende de la forma
        de los datos de entrenamiento para tomar forma.

        n_inputs (Integer): Dimension de los datos de entrada
        n_outputs (Integer): Dimension de salida, correspondiente a la cantidad de labels posibles
        depth (Integer): Cantidad de perceptrones iniciales
        activation (String): Nombre de la función de activación a usar
        optimizer (String): Nombre del optimizador a usar
        
        Retorna modelo instanciado
        """
        model = Sequential()
        model.add(Conv2D(depth, kernel_size=(3,1), activation='relu', input_shape=(n_inputs, 1, 1)))
        model.add(Flatten())
        model.add(Dense(n_outputs, activation=activation))
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model

    def fit(self, x, y):
        """
        Metodo que se usa para entrenar el modelo, no debe retornar nada.

        x (array-like): Arreglo donde cada componente tiene un texto ya preprocesado por el preprocess.
        y (array-like): Arreglo donde cada componente tiene un arreglo binario indicando si se prensenta o no la label
        """
        n_inputs, n_outputs = x.shape[1], y.shape[1]
        xt = x.reshape(x.shape[0], x.shape[1], 1, 1)
        # get model
        self.cnn = self.get_model(n_inputs,
                                    n_outputs,
                                    self.params['depth'],
                                    self.params['activation'],
                                    self.params['optimizer'])
        # fit the model on all data
        self.cnn.fit(xt,
                        y,
                        verbose=self.params['verb'],
                        epochs=self.params['epochs'])

    def predict(self, x):
        """
        Metodo que se usa para preprocesar los datos y de esta forma generar el
        input que recibe el modelo en fit, predict y predict_proba.

        x (array-like): Arreglo donde cada componente tiene un texto plano que necesita ser preprocesado.
        
        Retorna arreglo con etiquetas multilabel, pd.DataFrame
        """
        xt = x.reshape(x.shape[0], x.shape[1], 1, 1)
        pred = self.cnn.predict(xt)
        return pred.round().astype(int)

    def predict_proba(self, x):
        """
        x (array-like): Arreglo donde cada componente tiene un texto ya preprocesado por el preprocess.

        return: retorna arregla de probabilidad multilabel, pd.DataFrame
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        xt = x.reshape(x.shape[0], x.shape[1], 1, 1)
        return pd.DataFrame(self.cnn.predict_proba(xt)).astype(np.float64)

    def save(self, filename=None):
        """
        Módulo encargado de guardar la instancia entrenada del modelo

        filename (String): Nombre/path de destino donde se guardará el modelo entrenado
        """
        model_json = self.cnn.to_json()

        def BytesConverter(ObjectFile):
            bytes_container = BytesIO()
            dill.dump(ObjectFile, bytes_container)
            bytes_container.seek(0)
            bytes_file = bytes_container.read()
            return bytes_file
            base64File = base64.b64encode(bytes_file)
            return base64File

        model_bytes = BytesConverter(model_json)  
        weigth_bytes = BytesConverter(self.cnn.get_weights()) 
        #self.mlp.save(filename)
        info = self.kwargs
        params_bytes = pickle.dumps(info)
        params_size = len(params_bytes).to_bytes(length=3,byteorder="big")
        model_size = len(model_bytes).to_bytes(length=4,byteorder="big")
        bytes_file = params_size + params_bytes + model_size + model_bytes + weigth_bytes
        if filename:
            with open(filename,'wb') as file:
                file.write(bytes_file)
        else:
            return bytes_file

    @staticmethod
    def load(filename):
        """
        Módulo encargado de cargar un modelo ya entrenado con esta arquitectura

        filename String: Nombre/path de destino de donde se obtendrá el modelo entrenado

        Retorna modelo intanciado de la clase CNN
        """
        if isinstance(filename,str):
            with open(filename, 'rb') as file:
                bytes_file = file.read()
        elif isinstance(filename,bytes):
            bytes_file = filename
        else:
            raise TypeError
        params_size = int.from_bytes(bytes_file[0:3],byteorder="big")
        params_bytes = bytes_file[3:3+params_size]
        model_size = int.from_bytes(bytes_file[3+params_size:7+params_size],byteorder="big")
        model_bytes = bytes_file[7+params_size:7+params_size+model_size]
        weight_bytes = bytes_file[7+params_size+model_size:]

        def ObjectConverter(bytes_File):
            #loaded_binary = base64.b64decode(base64_File)
            loaded_object = tempfile.TemporaryFile()
            loaded_object.write(bytes_File)
            loaded_object.seek(0)
            ObjectFile = dill.load(loaded_object)
            loaded_object.close()
            return ObjectFile

        modeljson = ObjectConverter(model_bytes)
        modelweights = ObjectConverter(weight_bytes)
        loaded_model = model_from_json(modeljson)
        loaded_model.set_weights(modelweights)

        #with open(prepfile, 'rb') as file:
        #    info = pickle.load(file)
        info = pickle.loads(params_bytes)
        model = CNN(**info)
        model.cnn = loaded_model
        return model

MODEL = CNN

if __name__ == "__main__":
    pass
