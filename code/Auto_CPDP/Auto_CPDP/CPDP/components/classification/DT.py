import inspect
from collections import defaultdict

import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from Auto_CPDP.CPDP.components.base import AutoClassifier
from Auto_CPDP.CPDP.constants import *
from Auto_CPDP.CPDP.implementations.util import convert_multioutput_multiclass_to_multilabel
from Auto_CPDP.util.common import check_none


class CART(AutoClassifier):
    def __init__(self, criterion='gini', CARTsplitter='random', max_features=2,
                 RFmin_samples_split=1.0, max_depth=None, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None,
                 random_state=None):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth_factor = max_depth
        self.splitter = CARTsplitter
        self.min_samples_split = RFmin_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weight = class_weight
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

    def fit(self, X, y, sample_weight=None):
        from sklearn.tree import DecisionTreeClassifier

        self.max_features = float(self.max_features)
        # Heuristic to set the tree depth
        if check_none(self.max_depth_factor):
            max_depth_factor = self.max_depth_factor = None
        else:
            num_features = X.shape[1]
            self.max_depth_factor = int(self.max_depth_factor)
            max_depth_factor = max(
                1,
                int(np.round(self.max_depth_factor * num_features, 0)))
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.min_impurity_decrease = float(self.min_impurity_decrease)

        self.estimator = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=max_depth_factor,
            splitter=self.splitter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            random_state=self.random_state)
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'CART',
                'name': 'Decision Tree Classifier',
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

        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini")
        splitter = CategoricalHyperparameter("CARTsplitter", ['best', 'random'], default_value='random')
        min_samples_split = UniformIntegerHyperparameter(
            "RFmin_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        max_features = UniformFloatHyperparameter('max_features', 0.2, 1.0, default_value=1.0)

        cs.add_hyperparameters([criterion, max_features, splitter,min_samples_split,min_samples_leaf])
        return cs
