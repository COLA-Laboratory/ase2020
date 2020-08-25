from collections import OrderedDict
import importlib
import inspect
import pkgutil
import sys


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        # print(full_module_name not in sys.modules, module_name)

        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)


            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    # TODO test if the obj implements the interface
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[module_name] = classifier
        else:
            module = sys.modules[full_module_name]
            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    # TODO test if the obj implements the interface
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[module_name] = classifier

    return components


class AutoAdaptation(object):
    def __init__(self):
        return

class AutoClassifier(object):
    def __init__(self):
        return