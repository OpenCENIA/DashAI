import json
import logging
import pickle

import numpy as np
import pandas as pd
from functions import get_obj_schema
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.models import Sequential, load_model
from preprocess.distil_emb import DistilBertEmbedding
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from io import BytesIO
#import dill,base64,tempfile

from modelos.model import Model

logger = logging.getLogger()

#version == 1.0.4
class BiLSTM(Model):
    """
    Red neuronal con dos capas LSTM bidireccionales. Especializada para
    múltiples etiquetas.

    Model: Clase padre de todos los modelos implementados.
    """
    with open('parameters/models_schemas/lstm.json') as f:
        schema = json.load(f)
    
    def __init__(self, **kwargs):
        """
        Inicializa una instancia de BiLSTM.

        Kwargs:
            preprocess (PreProcess): Preprocesamiento escogido por el usuario.
                                        Default: DistilBertEmbedding.
            params (dict): Diccionario con los hiperparámetros personalizados
                            que se a la instancia de BiLSTM. La lista de
                            posibles hiperparámetros se encuentra a
                            continuación. Default: {}.

        params:
            activation (str): String con el nombre de la función de activación
                                de la primera capa densa. Default: 'relu'.
            batch (int): Tamaño del batch al entrenar. Default: 64.
            compiler_metrics (list): Lista de strings de las métricas
                                        utilizadas por el compilador.
                                        Default: ['accuracy'].
            dense_dim (int): Dimensión de la primera capa densa. Default: 512.
            dropout (float): Proporción de las conexiones botadas en la
                                primera capa densa al momento de entrenar.
                                Default: 0.5.
            epochs (int): Número de épocas (iteraciones) con las que
                            se entrenará. Default: 100.
            loss (Loss/str): Pérdida considerara al momento de optimizar la
                                red neuronal. Puede ser un objeto de tipo
                                Loss de Keras, o un string.
                                Default: BinaryCrossEntropy.
            lstm_dim (int): Dimensión de la primera capa LSTM. La segunda capa
                            LSTM tiene dimensión igual a la mitad de
                            la primera capa. Default: 128.
            optimizer Optimizer/str: Optimizador utilizado al entrenar la red.
                                        Puede ser de clase Optimizer de Keras,
                                        o un string.
            step (float): Tamaño del paso tomado en la dirección a optimizar.
                            Default: 0.01.
            verbose (int): 0 para no imprimir el desarrollo de la red.
                            1 para imprimir los detalles (con barra de
                            progreso). 2 para imprimir los detalles (sin barra
                            de progreso). Default: 0.
        """
        preprocess = kwargs.get('preprocess', None)
        if preprocess is None:
            preprocess = DistilBertEmbedding(
                {'params': {}, 'tokenizer': []})
        super().__init__(preprocess)

        self.params = kwargs.get('params', {})
        self.params['activation'] = self.params.get('activation', 'relu')
        self.params['batch'] = self.params.get('batch', 32)
        self.params['compiler_metrics'] = self.params.get(
            'compiler_metrics', ['accuracy'])
        self.params['dense_dim'] = self.params.get('dense_dim', 64)
        self.params['dropout'] = self.params.get('dropout', 0.5)
        self.params['epochs'] = self.params.get('epochs', 100)
        # self.params['loss'] = self.params.get(
        #     'loss', BinaryCrossentropy(from_logits=True))
        self.params['lstm_dim'] = self.params.get('lstm_dim', 64)
        self.params['step'] = self.params.get('step', 0.01)
        self.params['verbose'] =  self.params.get('verbose', 0)
        self.params['optimizer'] = self.params.get('optimizer', 'adam')
        
        self.kwargs = kwargs
        self.lstm = None

    def create_model(self, y_shape):
        """
        Genera la arquitectura de la red,

        y_shape (int): Largo del vector de salida, i.e. número de labels.
        """
        half_dim = int(self.params['lstm_dim'] / 2)

        model = Sequential()
        model.add(Bidirectional(LSTM(
            self.params['lstm_dim'], return_sequences=True)))
        model.add(Bidirectional(LSTM(half_dim)))
        model.add(Dense(self.params['dense_dim'],
                        activation=self.params['activation']))
        model.add(Dropout(self.params['dropout']))
        model.add(Dense(y_shape))
        
        model.compile(loss='binary_crossentropy',
                        optimizer=self.params['optimizer'],
                        metrics=['accuracy'])
        self.lstm = model

    def fit(self, X, y):
        """
        Método que se usa para entrenar el modelo, no debe retornar nada.

        X (array-like): Arreglo donde cada componente tiene un texto ya
                        preprocesado por preprocess.
        y (array-like): Arreglo donde cada componente tiene un arreglo
                        binario indicando si se presenta o no la label.
        """
        if not self.lstm:
            self.create_model(y.shape[1])

        X = X.reshape(X.shape[0], X.shape[1], 1)
        self.lstm.fit(x=X,
                        y=y,
                        verbose=self.params['verbose'],
                        epochs=self.params['epochs'],
                        batch_size=self.params['batch'])

    def predict(self, X):
        """
        Método que se usa para predecir.

        X (array-like): Arreglo donde cada componente tiene un texto ya
                            preprocesado por preprocess.

        Retorna matríz de adyacencia, que indica la pertenencia
        de las labels a las sentencias. En las filas se
        representan las sentencias (en orden), y en las
        columnas las etiquetas (en orden).
        """
        X = X.reshape(X.shape[0], X.shape[1], 1)
        y_pred = self.lstm.predict(X)
        y_pred[y_pred > 0] = 1
        y_pred[y_pred != 1] = 0
        y_pred = y_pred.astype(int)
        return y_pred

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X):
        """
        Entrega la probabilidad de que cada etiqueta pertenezca a las
        distintas sentencias.

        X (array-like): Arreglo donde cada componente tiene un texto ya
                        preprocesado por preprocess.

        Retorna dataFrame, donde cada sentencia es representada
        en una fila, y en cada columna se representa una
        etiqueta. En la entrada (m,n) se puede observar la
        probabilidad de que en la m-ésima sentencia esté
        la n-ésima etiqueta.
        """
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return pd.DataFrame(self.sigmoid(self.lstm.predict(X))).astype(np.float64)

    def save(self, filename=None):
        """
        Módulo encargado de guardar la instancia entrenada del modelo

        filename (String): Nombre/path de destino donde se guardará el modelo entrenado
        """
        model_json = self.lstm.to_json()

        def BytesConverter(ObjectFile):
            bytes_container = BytesIO()
            dill.dump(ObjectFile, bytes_container)
            bytes_container.seek(0)
            bytes_file = bytes_container.read()
            return bytes_file
            base64File = base64.b64encode(bytes_file)
            return base64File

        model_bytes = BytesConverter(model_json)  
        weigth_bytes = BytesConverter(self.lstm.get_weights()) 
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

        filename (String): Nombre/path de destino de donde se obtendrá el modelo entrenado

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
        model = BiLSTM(**info)
        model.lstm = loaded_model
        return model

MODEL = BiLSTM

if __name__ == '__main__':
    # Hacer caso de prueba
    pass
