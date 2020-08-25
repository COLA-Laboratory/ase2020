# -*- encoding: utf-8 -*-
from Auto_CPDP.constants import *
from Auto_CPDP.CPDP.CPDP import CPDP_pipeline


__all__ = [
    'get_configuration_space'
]


def get_configuration_space(info,
                            include_estimators=None,
                            exclude_estimators=None,
                            include_adpts=None,
                            exclude_adpts=None):
    exclude = dict()
    include = dict()
    if include_adpts is not None and \
            exclude_adpts is not None:
        raise ValueError('Cannot specify include_preprocessors and '
                         'exclude_preprocessors.')
    elif include_adpts is not None:
        include['preprocessor'] = include_adpts
    elif exclude_adpts is not None:
        exclude['preprocessor'] = exclude_adpts

    if include_estimators is not None and \
            exclude_estimators is not None:
        raise ValueError('Cannot specify include_estimators and '
                         'exclude_estimators.')
    elif include_estimators is not None:
        if info['task'] in CLASSIFICATION_TASKS:
            include['classifier'] = include_estimators
        elif info['task'] in REGRESSION_TASKS:
            include['regressor'] = include_estimators
        else:
            raise ValueError(info['task'])
    elif exclude_estimators is not None:
        if info['task'] in CLASSIFICATION_TASKS:
            exclude['classifier'] = exclude_estimators
        elif info['task'] in REGRESSION_TASKS:
            exclude['regressor'] = exclude_estimators
        else:
            raise ValueError(info['task'])

    return _get_classification_configuration_space(info, include, exclude)




def _get_classification_configuration_space(info, include, exclude):
    task_type = info['task']

    multilabel = False
    multiclass = False
    sparse = False

    if task_type == MULTILABEL_CLASSIFICATION:
        multilabel = True
    if task_type == REGRESSION:
        raise NotImplementedError()
    if task_type == MULTICLASS_CLASSIFICATION:
        multiclass = True
    if task_type == BINARY_CLASSIFICATION:
        pass

    if info['is_sparse'] == 1:
        sparse = True

    dataset_properties = {
        'multilabel': multilabel,
        'multiclass': multiclass,
        'sparse': sparse
    }

    return CPDP_pipeline(
        dataset_properties=dataset_properties,
        include=include, exclude=exclude).\
        get_hyperparameter_search_space()

