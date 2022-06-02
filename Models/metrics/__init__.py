from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

available_metrics = []
# iterate through the modules in the current package
package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([package_dir]):
    # import the module and iterate through its attributes
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute):
            # Add the class to this package's variables
            try:
                model_name = attribute.METRIC
                if model_name not in available_metrics:
                    available_metrics += [model_name]
                    globals()[model_name] = attribute
            except Exception as e:
                print(e)
                continue


def get_available_metrics():
    return available_metrics
