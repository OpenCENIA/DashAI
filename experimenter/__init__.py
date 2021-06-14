import pandas as pd
import numpy

from functools import reduce

from .utils import creator, models_trainer, optimizer, models_tester, models, check_experimenter_input, set_params_to_list, get_set_result, get_dataset
#from functions import n_first


class Experimenter:
    """
    Clase que abstrae los distintos experimentos.
    """

    def _delete_experiment(func):
        """
        Etiqueta que permite eliminar el experimento creado en caso de
        que ocurra un error en la funcion.
        """
        def funcion(self):
            try:
                return func(self)
            except Exception as e:
                models.db.session.delete(self.experiment)
                models.db.session.commit()
                raise e
        return funcion

    def __init__(self, input_dict):
        """
        Inicialización del experimento, y parseo de datos.
        """
        self.experiment = models.Experimento()
        self.db_models = []
        models.db.session.add(self.experiment)
        models.db.session.commit()
        self.input = input_dict
        self.data_parser()

    @_delete_experiment
    def create(self):
        """
        Instancia modelos y métricas.
        """
        self.models, self.metrics = creator(
            self.input['models'],
            self.input['metrics']['show']
            )

    @_delete_experiment
    def train(self):
        """
        Entrena los modelos y genera reportes con respecto
        al conjunto de validación.
        """
        self.reports = models_trainer(
            self.models,
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val
        )

    @_delete_experiment
    def optimize(self):
        """
        Ordena los reportes de cada clase, respecto a métricas
        y labels especificas. Filtra las n mejores combinaciones
        de parámetros para cada modelo.
        """
        self.filtered_models = optimizer(self.reports, self.input['metrics'])

    @_delete_experiment
    def test(self):
        """
        Entrega los reportes con respecto al conjunto de testeo
        de los mejores modelos.
        """
        self.output = models_tester(
            self.filtered_models,
            self.metrics,
            self.X_test,
            self.y_test,
            self.experiment.id
        )

    def __save(self):
        """
        Inserat toda al informacion relevante del experimento en la DB, este método
        recorre todos los modelos entrenados y guarda, sus instancias y resultados
        en la BD, asegurandose que la informacion sea congruente en todas las tablas.
        """
        # De un experimento se deben guardar:
        #   - Resultados
        #   - Modelos
        #   - Experimento
        #   - Experimentos con modelos, y agregar hiperparametros
        #   - Experimentos con metricas

        self.get_report()

        # Primero incializamos las arquitecturas y metricas
        arcs = models.Arquitectura.query.all()
        metrics = models.Metrica.query.all()

        # Guardamos Modelos
        # Necesitamos name,  params e instancia. Y enlazarlos a una arq
        for j in range(len(self.filtered_models)):
            for i in range(len(self.filtered_models[j])):
                act_modelo = self.filtered_models[j][i]
                # Obtenemos arq asociada
                arc = list(filter(lambda arc: arc.class_name == type(act_modelo).__name__, arcs))[0]
                # Obtenemos name
                name = f"{arc.name}-{j}-{i}-{self.experiment.id}"
                short_name = name[-30:] if len(name) > 30 else name
                # Obtenemos instancia (arachivo de bytes)
                instancia = act_modelo.save()
                # Guardamos modelo en tabla Modelo
                db_model = models.Modelo(short_name, act_modelo.params, instancia, arc.id)

                # obtener todos los resultados de un modelo y guardarlos.
                results = [self.train_report[name], self.val_report[name], self.test_report[name]]
                result_kind = ["TRAIN","VAL","TEST"]
                for report,kind in zip(results,result_kind):
                    for metric in report:
                        met = list(filter(lambda met: met.name == metric, metrics))[0]
                        for label in report[metric]:
                            value = report[metric][label]
                            resultado = models.Resultado(kind, value, met.id)
                            db_model.results.append(resultado)
                            self.experiment.results.append(resultado)
                self.db_models.append(db_model)
                self.experiment.models.append(db_model)

        models.db.session.commit()
        self._set_models_id()

    @_delete_experiment
    def save(self):
        """
        Guarda la informacion relevante del experimento y en caso de ocurrir un
        error, se asegura que toda la informacion de experimento sea borrada,
        para asi evitar incongruencias en la DB.
        """
        try:
            self.__save()
        except Exception as e:
            if len(self.db_models):
                self.__delete_models()
            raise e

    def __delete_models(self):
        """
        Elimina toda la informacion existente del experimento en la DB.
        De esta forma se evita que existen incongruencias en la misma.
        """
        for model in self.db_models:
            model.delete_file()

    @_delete_experiment
    def data_parser(self):
        """
        Obtiene todos los conjuntos de entrenamiento necesarios
        para el experimento
        """
        dataset_dic = self.input['datasets']
        n_of_labels = dataset_dic.get("n",None)
        dataset = get_dataset(dataset_dic)
        train = dataset.train
        val = dataset.val
        test = dataset.test
        
        x_len = len(dataset.tags['x'])
        if len(dataset.tags['x']) == 1:
            x_tags = dataset.tags['x'][0]
        else:
            x_tags = dataset.tags['x']

        self.X_train = train[x_tags].to_numpy()
        self.X_val = val[x_tags].to_numpy()
        self.X_test = test[x_tags].to_numpy()

        self.y_train = train.iloc[:,x_len+1:n_of_labels].values
        self.y_val = val.iloc[:,x_len+1:n_of_labels].values
        self.y_test = test.iloc[:,x_len+1:n_of_labels].values

    def get_report(self):
        """
        Genera y guarda los resultados de los modelos escogidos por el optimizador para los 3 sets usados en el experimento.
        Estos datos estan hechos para facilitar su insercion a la DB.
        """
        self.train_report = get_set_result(self.filtered_models, self.metrics, self.X_train, self.y_train, self.experiment.id)
        self.val_report = get_set_result(self.filtered_models, self.metrics, self.X_val, self.y_val, self.experiment.id)
        self.test_report = get_set_result(self.filtered_models, self.metrics, self.X_test, self.y_test, self.experiment.id)

    def _set_models_id(self):
        """
        Cambia los id del output por el correpondiente en la DB.
        Este método solo tiene sentido ejecutarlo luego del
        método _save, de otra forma la DB aun no tendra guardado los
        modelos.
        """
        for arquitectura in self.output:
            for model_dict in self.output[arquitectura]:
                model_dict["model_id"] = models.Modelo.query.filter_by(name = model_dict["name"]).first().id
