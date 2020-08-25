import inspect
from collections import defaultdict

import numpy as np
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from sklearn.mixture import GaussianMixture

from Auto_CPDP.CPDP.components.base import AutoAdaptation


class TDS(AutoAdaptation):
    def __init__(self, strategy='NN', expected_num=3):
        self.strategy = strategy
        self.expected_num = expected_num

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

    def _get_distributional_characteristics_vector(self, Xs, Ys, Xt, Yt, loc):
        self.X = dict()
        self.Y = dict()
        Vchracteristics = dict()
        for i in range(len(loc)):
            if i < len(loc) - 1:
                x = Xs[loc[i]:loc[i + 1]]
                y = Ys[loc[i]:loc[i + 1]]
            else:
                x = Xs[loc[i]:]
                y = Ys[loc[i]:]
            m = x.mean(axis=0)
            std = x.std(axis=0)
            Vchracteristics[i] = np.concatenate((m, std))
            self.X[i] = x
            self.Y[i] = y

        m = Xt.mean(axis=0)
        std = Xt.mean(axis=0)
        return Vchracteristics, np.concatenate((m, std))

    def _EM_clustering(self, source, target):
        x = np.asarray(list(source.values()))
        x = np.concatenate((x, [target]), axis=0)

        gmm = GaussianMixture(n_components=2, covariance_type='full')
        gmm.fit(x)
        label_t = gmm.predict(target.reshape(1, -1))
        xs = self.X[0]
        ys = self.Y[0]

        for k, v in source.items():
            label = gmm.predict(v.reshape(1, -1))
            if label == label_t:
                xs = np.concatenate((xs, self.X[k]), axis=0)
                ys = np.concatenate((ys, self.Y[k]), axis=0)

        xs = xs[self.X[0].shape[0]:, :]
        ys = ys[self.Y[0].shape[0]:]

        return xs, ys

    def _Nearest_Neighbor_Selection(self, source, target):
        dist = dict()
        for k, v in source.items():
            dist[k] = np.linalg.norm(v - target)

        dist = sorted(dist.items(), key=lambda x: x[1])
        xs = self.X[dist[0][0]]
        ys = self.Y[dist[0][0]]

        for i in range(1, self.expected_num):
            xs = np.concatenate((xs, self.X[dist[i][0]]))
            ys = np.concatenate((ys, self.Y[dist[i][0]]))

        return xs, ys

    def run(self, Xs, Ys, Xt, Yt, loc):
        self.expected_num = min(len(loc), self.expected_num)
        source, target = self._get_distributional_characteristics_vector(Xs, Ys, Xt, Yt, loc)

        if self.strategy == 'EM':
            xs, ys = self._EM_clustering(source, target)
            return xs, ys, Xt, Yt

        elif self.strategy == 'NN':
            xs, ys = self._Nearest_Neighbor_Selection(source, target)
            return xs, ys, Xt, Yt

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        k = UniformIntegerHyperparameter('expected_num', 2, 11, default_value=4)
        strategy = CategoricalHyperparameter('strategy', ['NN', 'EM'], default_value='NN')

        cs.add_hyperparameters([strategy, k])
        cond = EqualsCondition(k, strategy, 'NN')
        cs.add_condition(cond)
        return cs
