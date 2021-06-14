"""Funciones de uso general, proyecto Corte Suprema.
"""

import logging
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger()

def change_dict_type(dic):
    final = {}
    changer = {'str': 'string', 'bool': 'boolean', 'int': 'integer',
                'float': 'number', 'list': 'array'}
    for key in dic:
        if dic[key] in changer:
            final[key] = changer[dic[key]]
    return final

def get_obj_schema(obj):
    schema = {key: type(obj.__dict__[key]).__name__
                        for key in obj.__dict__}
    return change_dict_type(schema)

def load_all_datasets():
    """Entrega un diccionario con todos los datasets (separados) en
    train, test,validation sets y embeddings. Además, incluye las
    stopwords.

    Returns:
        - dict: El diccionario contiene otros tres diccionarios, de llaves
        "ges", "planbase" y "urbanismo", para cada una de las bases de datos
        respectiva. La cuarta llave es "stopwords", la cual entrega una
        lista con las stopwords pre-procesadas (sin puntuación y en
        minúsculas). Los datasets incluyen train, test y validation sets,
        además de los bert embeddings.
    """

    os.chdir('/content/drive/Shareddrives/Corte Suprema/Datasets')

    train_GES = pd.read_csv('GES/train.csv')
    test_GES = pd.read_csv('GES/test.csv')
    val_GES = pd.read_csv('GES/validation.csv')
    bert_GES = np.load('GES/bert.npy', allow_pickle=True)

    train_PB = pd.read_csv('PLANBASE/train.csv')
    test_PB = pd.read_csv('PLANBASE/test.csv')
    val_PB = pd.read_csv('PLANBASE/validation.csv')
    bert_PB = np.load('PLANBASE/bert.npy', allow_pickle=True)

    train_UR = pd.read_csv('URBANISMO/train.csv')
    test_UR = pd.read_csv('URBANISMO/test.csv')
    val_UR = pd.read_csv('URBANISMO/validation.csv')
    bert_UR = np.load('URBANISMO/bert.npy', allow_pickle=True)
    roberta_UR = np.load('URBANISMO/original.npy', allow_pickle=True)
    roberta_UR_preprocessed = np.load('URBANISMO/preprocessed.npy', allow_pickle=True)
    roberta_UR_without_sw = np.load('URBANISMO/without_stopwords.npy', allow_pickle=True)

    GES = {'train': train_GES, 'test': test_GES, 'validation': val_GES,
            'bert': bert_GES}
    PB = {'train': train_PB, 'test': test_PB, 'validation': val_PB,
            'bert': bert_PB}
    UR = {'train': train_UR, 'test': test_UR, 'validation': val_UR,
            'bert': bert_UR, 'original': roberta_UR,
            'preprocessed': roberta_UR_preprocessed, 
            'without_stopwords': roberta_UR_without_sw}

    with open('stopword_full5.txt') as f:
        stopwords = [re.sub(r'[,\\.!?]', '', line.strip()).lower() for line in f]

    os.chdir('/content')

    return {'ges': GES, 'planbase': PB, 'urbanismo': UR, 'stopwords': stopwords}

def load_dataset(name):
    """Entrega un diccionario con los train, test y validation tests de la
    base de datos requerida.

    Args:
        - name (str): Corresponde al nombre de la base de datos que se
        quiere cargar.
        Los posibles valores son "ges", "planbase" y "urbanismo".

    Returns:
        - dict: El diccionario contiene otras cuatro llaves: "train", "test",
        "validation", "bert".
        Cada una de las primeras tres llaves está asociada a su respectivo
        dataset. La última llave contiene los embeddings de toda la base.
    """

    os.chdir('/content/drive/Shareddrives/Corte Suprema/Datasets')

    folders = {'ges': 'GES', 'planbase': 'PLANBASE', 'urbanismo': 'URBANISMO'}
    folder = folders[name]

    train = pd.read_csv(os.path.join(folder, 'train.csv'))
    test = pd.read_csv(os.path.join(folder, 'test.csv'))
    val = pd.read_csv(os.path.join(folder, 'validation.csv'))
    bert = np.load(os.path.join(folder, 'bert.npy'), allow_pickle=True)
    if name == 'urbanismo':
        roberta_UR = np.load('URBANISMO/original.npy', allow_pickle=True)
        roberta_UR_preprocessed = np.load('URBANISMO/preprocessed.npy', allow_pickle=True)
        roberta_UR_without_sw = np.load('URBANISMO/without_stopwords.npy', allow_pickle=True)


    os.chdir('/content')

    return {'train': train, 'test': test, 'validation': val, 
            'bert': bert, 'original': roberta_UR,
            'preprocessed': roberta_UR_preprocessed,
            'without_stopwords': roberta_UR_without_sw}

def load_stopwords(filename='stopword_full5.txt'):
    """Entrega las stopwords pre-procesadas.

    Returns:
        - list: Contiene las stopwords pre-procesadas
        (sin puntuación y en minúsculas).
    """

    os.chdir('/content/drive/Shareddrives/Corte Suprema/Datasets')

    with open(filename) as f:
        stopwords = [re.sub(r'[,\\.!?]', '', line.strip()).lower() for line in f]

    os.chdir('/content')

    return stopwords

def load_full_datasets():
    """Entrega las tres bases de datos, sin separar entre distintos tipos
    de dataset.

    Returns:
        - dict: Diccionario con tres dataframes, correspondientes a su
        respectiva base de datos. Las llaves son: "ges", "planbase",
        "urbanismo".
    """
    os.chdir('/content/drive/Shareddrives/Corte Suprema/Datasets')

    GES = pd.read_csv('GES/full_dataset.csv')
    PB = pd.read_csv('PLANBASE/full_dataset.csv')
    UR = pd.read_csv('URBANISMO/full_dataset.csv')

    os.chdir('/content')

    return {'ges': GES, 'planbase': PB, 'urbanismo': UR}

def load_full_dataset(name):
    """Entrega la base de datos entera, asociada al nombre entregado.

    Args:
        - name (str): Nombre de la base de datos que se quiere cargar.
        Puede ser "ges", "planbase" o "urbanismo".

    Returns:
        - dataframe: Dataframe con la respectiva base de datos entera.
    """
    os.chdir('/content/drive/Shareddrives/Corte Suprema/Datasets')

    folders = {'ges': 'GES', 'planbase': 'PLANBASE', 'urbanismo': 'URBANISMO'}
    folder = folders[name]
    df = pd.read_csv(os.path.join(folder, 'full_dataset.csv')) 

    os.chdir('/content')

    return df

def n_first(df, n, pad=4):
    """Entrega el dataset filtrado con las primeras n labels.

    Args:
        - df (pandas.DataFrame): Dataframe al cual se le quiere aplicar el filtro.
        - n (int): Número de labels que se quieren considerar.
        - pad (int, optional): Columnas que tiene el df antes de las labels.
        Defaults to 4.

    Returns:
        - pandas.DataFrame: Dataset con filtro.
    """
    if n < 1:
        return df[df.columns[0:pad]]
    total_cols = len(df.columns)
    return df[df.columns[0:min(pad + n, total_cols - 1)]]

def prop_first(df, p, pad=4):
    """Entrega el dataset filtrado con una proporción p
    del total de labels.

    Args:
        - df (pandas.DataFrame): Dataframe al cual se le quiere aplicar el filtro.
        - p (float): Proporción ([0, 1]) de labels que se quieren considerar.
        - pad (int, optional): Columnas que tiene el df antes de las labels.
        Defaults to 4.

    Returns:
        - pandas.DataFrame: Dataset con filtro.
    """
    if p >= 1:
        return df
    if p <= 0:
        return None
    n_cols = len(df.columns) - pad
    n = int(p * n_cols)
    return df[df.columns[0:pad + n]]

def n_filter(df, n, pad=4):
    """Entrega el dataset filtrado con las labels que aparezcan
    en más de n sentencias.

    Args:
        - df (pandas.DataFrame): Dataframe al cual se le quiere aplicar el filtro.
        - n (int): Número de apariciones en sentencias de las
        labels consideradas.
        - pad (int, optional): Columnas que tiene el df antes de las labels.
        Defaults to 4.

    Returns:
        - pandas.DataFrame: Dataset con filtro.
    """
    m = sum(df[df.columns[pad:]].sum().values > n)
    return df[df.columns[0:pad + m]]

def remove_punctuation(df, col):
    """Elimina los principales signos de puntuación, junto con las comillas.

    Args:
        df (pandas.DataFrame): Dataframe al cual se le remueven los signos de puntuación.
        col (string): Nombre de la columna objetivo.

    Returns:
        pandas.DataFrame: Dataframe modificado.
    """
    df[col] = df[col].map(lambda x: x.translate(str.maketrans('', '', '''!()-[]{};:'"\,<>./?@#$%^&*_~''')))
    df[col] = df[col].map(lambda x: re.sub('“', "", x))
    df[col] = df[col].map(lambda x: re.sub('”', "", x))
    return df

def remove_line_breaks(df, col):
    """Elimina los saltos de línea de una columna.

    Args:
        df (pandas.DataFrame): Dataframe al cual se le remueven los saltos de línea.
        col (string): Nombre de la columna objetivo.

    Returns:
        pandas.DataFrame: Dataframe modificado.
    """
    df[col] = df[col].map(lambda x: " ".join(x.split())) 
    return df

def remove_numbers(df, col):
    """Elimina los caracteres numéricos de una columna.

    Args:
        df (pandas.DataFrame): Dataframe al cual se le remueven los caracteres numéricos.
        col (string): Nombre de la columna objetivo.

    Returns:
        pandas.DataFrame: Dataframe modificado.
    """
    df[col] = df[col].map(lambda x: ''.join(i for i in x if not i.isdigit()))
    return df

def aux_stopwords(row, stopwords):
    """Remueve las stopwords de una fila.

    Args:
        row (pandas row): Fila a la cual se le remueven las stopwords.
        stopwords (list): Lista con las stopwords.

    Returns:
        pandas row: Fila modificada.
    """
    row = ' ' + row + ' '
    for word in stopwords:
        mod1 = ' ' + word + ' '
        mod2 = ' ' + word.capitalize() + ' '
        row = row.replace(mod1, ' ')
        row = row.replace(mod2, ' ')
    row = row.strip()
    return row

def remove_stopwords(df, col, stopwords):
    """Elimina las stopwords de una columna.

    Args:
        df (pandas.DataFrame): Dataframe al cual se le remueven los caracteres numéricos.
        col (string): Nombre de la columna objetivo.

    Returns:
        pandas.DataFrame: Dataframe modificado.
    """
    stopwords.sort(key=lambda x: len(x.split()), reverse=True)
    df[col] = df[col].map(lambda x: aux_stopwords(x, stopwords))
    return df

def first_preprocess(df, col, ):
    """Corresponde al primer preproceso, que incluye eliminación de puntuación,
    eliminación de caracteres numéricos, y eliminación de stopwords. Útil para
    generar los embeddings de tipo preprocessed.

    Args:
        df (pandas.DataFrame): Dataframe al cual se le aplica el preprocesamiento.
        col (string): Nombre de la columna objetivo. 

    Returns:
        pandas.DataFrame: Dataframe modificado.
    """
    df = remove_punctuation(df, col)
    df = remove_numbers(df, col)
    df = remove_line_breaks(df, col) 
    return df

def df_without_stopwords(df, col, stopwords):
    """Corresponde al segundo preproceso, que incluye el primer preprocesamiento
    y la remoción de stopwords. Útil para generar los embeddings de tipo
    without_stopwords.

    Args:
        df (pandas.DataFrame): Dataframe al cual se le aplica el preprocesamiento.
        col (string): Nombre de la columna objetivo. 

    Returns:
        pandas.DataFrame: Dataframe modificado.
    """
    df = first_preprocess(df, col)
    df = remove_stopwords(df, col, stopwords)
    return df

if __name__ == "__main__":
    pass
