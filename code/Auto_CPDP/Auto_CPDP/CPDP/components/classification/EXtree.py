import inspect
import sklearn
from collections import defaultdict

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, UniformIntegerHyperparameter

from Auto_CPDP.CPDP.components.base import AutoClassifier
from Auto_CPDP.CPDP.constants import *


class EXtree(AutoClassifier):
    def __init__(self, criterion, min_samples_leaf,
                 min_samples_split,  max_features, splitter):
        self.splitter = splitter
        self.criterion = criterion
        self.min_leaf = min_samples_leaf
        self.min_split = min_samples_split
        self.max_feature = max_features
        self.estimator = None


    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def fit(self, X, y, sample_weight):
        self.estimator = sklearn.tree.ExtraTreeClassifier(criterion=self.criterion,
                                                          splitter=self.splitter, min_samples_leaf=self.min_leaf,
                                                          min_samples_split=self.min_split, max_features=self.max_feature)

        self.estimator.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'NB',
                'name': 'Naive Bayes classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # the smoothing parameter is a non-negative float
        # I will limit it to 1000 and put it on a logarithmic scale. (SF)
        # Please adjust that, if you know a proper range, this is just a guess.
        # alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100,
        #                                    default_value=1, log=True)

        criterion = CategoricalHyperparameter('criterion', ['gini', 'entropy'], 'gini')
        split = CategoricalHyperparameter('splitter', ['random', 'best'], 'best')
        maxfeature = UniformFloatHyperparameter('max_features', 0.2, 1.0, 1.0)
        min_leaf = UniformIntegerHyperparameter('min_samples_leaf', 1, 20, 1)
        min_split = UniformIntegerHyperparameter('min_samples_split', 1, 40, 2)
        cs.add_hyperparameters([criterion, split, maxfeature, min_split, min_leaf])
        return cs
