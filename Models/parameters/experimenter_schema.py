from parameters.preprocesses_mapping import preprocesses_mapping
from parameters.tokenizers_mapping import tokenizers_mapping

import models


def get_full_schema():
    models_mapping = models.arquitectures_mapping()
    metrics_mapping = models.metrics_mapping()
    preprocesses_schemas = []
    for preprocess in preprocesses_mapping:
        schema = {
            "properties": {
                "name": {
                    "error_msg": "Los preprocesamientos \
                                  deben ser alguno de los "
                    f"siguientes: {list(preprocesses_mapping.keys())}.",
                    "type": "string",
                    "enum": list(preprocesses_mapping.keys()),
                }
            },
            "if": {"properties": {"name": {"const": preprocess}}},
            "then": {
                "error_msg": "Los preprocess deben tener la llave 'name' y, "
                "opcionalmente, las llaves 'params' y 'tokenizers'.",
                "type": "object",
                "properties": {
                    "name": {"const": preprocess},
                    "params": preprocesses_mapping[preprocess].schema,
                    "tokenizers": {
                        "oneOf": [
                            {
                                "error_msg": "Los tokenizers \
                                              deben ser alguno(s) de "
                                f"{list(tokenizers_mapping.keys())}, ya ."
                                "sea como lista o string.",
                                "type": "string",
                                "enum": list(tokenizers_mapping.keys()),
                            },
                            {
                                "items": {
                                    "error_msg": "Los tokenizers\
                                                 deben ser alguno(s) de "
                                    f"{list(tokenizers_mapping.keys())}.",
                                    "type": "string",
                                    "enum": list(tokenizers_mapping.keys()),
                                },
                                "type": "array",
                            },
                        ]
                    },
                },
                "additionalProperties": False,
                "required": ["name"],
            },
        }

        preprocesses_schemas.append(schema)

    models_schemas = []
    for model in models_mapping:
        schema = {
            "properties": {
                "name": {
                    "error_msg": "Los modelos deben ser alguno de los "
                    f"siguientes: {list(models_mapping.keys())}.",
                    "type": "string",
                    "enum": list(models_mapping.keys()),
                }
            },
            "if": {"properties": {"name": {"const": model}}},
            "then": {
                "error_msg": "Los modelos deben tener la llave 'name' y, "
                "opcionalmente, las llaves 'params' y 'preprocesses'.",
                "type": "object",
                "properties": {
                    "name": {"const": model},
                    "params": models_mapping[model].schema,
                    "preprocesses": {
                        "oneOf": [
                            {
                                "type": "object",
                                "error_msg": "Los preprocesamientos deben\
                                 ser entregados como diccionario \
                                individual, o como una lista de diccionarios.",
                                "allOf": preprocesses_schemas,
                            },
                            {
                                "type": "array",
                                "error_msg": "Los preprocesamientos deben\
                                 ser entregados como diccionario individual,\
                                  o como una lista de diccionarios.",
                                "items": {
                                    "type": "object",
                                    "allOf": preprocesses_schemas,
                                },
                            },
                        ]
                    },
                },
                "additionalProperties": False,
                "required": ["name"],
            },
        }

        models_schemas.append(schema)

    experimenter_schema = {
        "error_msg": "Las llaves del diccionario\
                        deben ser 'models', 'datasets' "
        "'metrics' y 'taks'.",
        "type": "object",
        "properties": {
            "models": {
                "type": ["array", "object"],
                "error_msg": "Los modelos deben ser\
                        entregados como diccionario "
                "individual, o como una lista de diccionarios.",
                "oneOf": [
                    {
                        "type": "object",
                        "allOf": models_schemas,
                    },
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "allOf": models_schemas,
                        },
                    },
                ],
            },
            "datasets": {
                "error_msg": "Los datasets deben ser ingresados como un "
                "diccionario, con la llave 'id' o"
                " con la llave 'name', solo una de las dos. "
                "Puede tener opcionalmente la llave 'n'",
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "minimum": 1,
                        "error_msg": "El value de la llave 'id' "
                        "debe ser un entero mayor a 0.",
                    },
                    "name": {
                        "type": "string",
                        "error_msg": "El value de la llave 'name' "
                        "debe ser un string.",
                    },
                    "n": {
                        "type": "integer",
                        "minimum": 1,
                        "error_msg": "El value de la llave 'n' "
                        "debe ser un entero mayor a 0.",
                    },
                },
                "oneOf": [{"required": ["name"]}, {"required": ["id"]}],
                "additionalProperties": False,
            },
            "metrics": {
                "error_msg": "Las llaves del diccionario de metrics deben "
                "ser 'show', 'optimizer_label', 'optimizer_metric' "
                "o 'best_n'",
                "type": "object",
                "properties": {
                    "show": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(metrics_mapping.keys()),
                            "error_msg": "Las métricas deben ser algunas "
                            f"de {list(metrics_mapping.keys())}.",
                        },
                        "default": [],
                    },
                    "optimizer_label": {
                        "default": "micro avg",
                        "oneOf": [
                            {
                                "type": "integer",
                                "minimum": 0,
                                "error_msg": "La label a \
                                            optimizar debe ser el "
                                "índice de la label (partiendo desde "
                                "0), o alguna de ['global', 'micro avg', "
                                "'macro avg', 'weighted avg', "
                                "'samples avg'].",
                            },
                            {
                                "type": "string",
                                "enum": [
                                    "global",
                                    "micro avg",
                                    "macro avg",
                                    "samples avg",
                                    "weighted avg",
                                ],
                                "error_msg": "La label a optimizar\
                                                debe ser el "
                                "índice de la label (partiendo desde "
                                "0), o alguna de ['global', 'micro avg', "
                                "'macro avg', 'weighted avg', "
                                "'samples avg'].",
                            },
                        ],
                    },
                    "optimizer_metric": {
                        "type": "string",
                        "default": "f1",
                        "enum": list(metrics_mapping.keys()),
                        "error_msg": "Las métrica a optimizar debe ser una de "
                        f"{list(metrics_mapping.keys())}.",
                    },
                    "best_n": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 0,
                        "error_msg": "El número de resultados\
                                        a probar debe ser "
                        "positivo, o 0 si se quieren mostrar todas.",
                    },
                },
                "additionalProperties": False,
                "allOf": [
                    {
                        "if": {
                            "properties": {
                                "optimizer_label": {"const": "global"}
                            }
                        },
                        "then": {
                            "properties": {
                                "optimizer_metric": {
                                    "const": "accuracy",
                                    "error_msg": "Global funciona\
                                                    con accuracy.",
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {
                                "optimizer_metric": {"const": "accuracy"}
                            }
                        },
                        "then": {
                            "properties": {
                                "optimizer_label": {
                                    "const": "global",
                                    "error_msg": "Accuracy no \
                                                reconoce la label "
                                    "introducida. Use global.",
                                }
                            }
                        },
                    },
                ],
            },
            "task": {
                "error_msg": "task debe ser 'multilabel',\
                            'multiclass' o 'binary'.",
                "type": "string",
                "enum": ["multilabel", "multiclass", "binary"],
            },
        },
        "required": ["models", "datasets", "metrics", "task"],
        "additionalProperties": False,
    }
    return experimenter_schema
