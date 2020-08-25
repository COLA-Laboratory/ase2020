import inspect
from collections import defaultdict

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from sklearn.neighbors import NearestNeighbors

from Auto_CPDP.CPDP.components.base import AutoAdaptation


class NNfilter(AutoAdaptation):
    def __init__(self, n_neighbors=10, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

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

    def run(self, Xsource, Ysource, Xtarget, Ytarget):
        # Log transformation
        Xsource = np.log(Xsource + 1)
        Xtarget = np.log(Xtarget + 1)

        if self.n_neighbors > Xsource.shape[0]:
            return 0, 0, 0, 0

        knn = NearestNeighbors(metric=self.metric)
        knn.fit(Xsource)
        data = []
        ysel = []

        # find neighbors
        for item in Xtarget:
            tmp = knn.kneighbors(item.reshape(1, -1), self.n_neighbors, return_distance=False)
            tmp = tmp[0]
            for i in tmp:
                if list(Xsource[i]) not in data:
                    data.append(list(Xsource[i]))
                    ysel.append(Ysource[i])
        Xsource, idx = np.unique(np.asanyarray(data), axis=0, return_index=True)
        Ysource = np.asanyarray(ysel)
        Ysource = Ysource[idx]

        return Xsource, Ysource, Xtarget, Ytarget

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        n = UniformIntegerHyperparameter('n_neighbors', 1, 100, default_value=10)
        metric = CategoricalHyperparameter('metric', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'], default_value='euclidean')
        cs.add_hyperparameters([n, metric])
        return cs
