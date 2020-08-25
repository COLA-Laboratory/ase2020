import os
from Auto_CPDP.CPDP.components.base import find_components, AutoAdaptation


def get_domain_adaptation():
    choices = find_components(__package__,
                                   os.path.split(__file__)[0],
                                   AutoAdaptation)

    return choices




