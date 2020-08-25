import os
from Auto_CPDP.CPDP.components.base import AutoClassifier, find_components


def get_classification():
    choices = find_components(__package__,
                              os.path.split(__file__)[0],
                              AutoClassifier)

    return choices
