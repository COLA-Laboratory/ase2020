import inspect
from collections import defaultdict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import numpy as np


# the efficiency of over-sampling should be optimized!
from Auto_CPDP.CPDP.components.base import AutoAdaptation


class CDE_SMOTE(AutoAdaptation):
    def __init__(self, model, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.model = model

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

    def _over_sampling(self, x, idx, num):
        x_over = x[idx]
        knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        knn.fit(x_over)
        neighbors = knn.kneighbors(x_over, return_distance=False)
        if x_over.shape[0] > num:
            idx = np.random.choice(x_over.shape[0], num, replace=False)
        else:
            idx = np.random.choice(x_over.shape[0], num, replace=True)
        for i in idx:
            i = int(i)
            rnd = int(neighbors[i][int(np.random.choice(self.k, 1))])
            xnew = x_over[i] + np.random.random() * (x_over[i] - x[rnd])
            x = np.concatenate((x, xnew.reshape(1, -1)), axis=0)
        return x

    def _class_distribution_estimation(self):
        m = np.bincount(self.Ysource)
        x = self._over_sampling(self.Xsource, np.where(self.Ysource == 1)[0], m[0] - m[1])
        y = np.concatenate((self.Ysource, np.ones(m[0] - m[1])), axis=0)
        self.model.fit(x, y)
        prediction = self.model.predict(self.Xtarget).astype(np.int)
        return np.bincount(prediction)

    def _class_distribution_modification(self, n):
        m = np.bincount(self.Ysource)
        print(m, n)
        num = int(m[0] * n[0] / n[1]) - m[1]
        print(num)
        self.Xsource = self._over_sampling(self.Xsource, np.where(self.Ysource == 1)[0], num)
        self.Ysource = np.concatenate((self.Ysource, np.ones(num)), axis=0)
        self.model.fit(self.Xsource, self.Ysource)

    def run(self, Xs, Ys, Xt, Yt, loc):
        self.Xsource = np.asarray(Xs)
        self.Xtarget = np.asarray(Xt)
        self.Ysource = np.asarray(Ys).astype(np.int)
        self.Ytarget = np.asarray(Yt)

        try:
            n = self._class_distribution_estimation()
            self._class_distribution_modification(n)
            prediction = self.model.predict(self.Xtarget)
            return roc_auc_score(self.Ytarget, prediction)
        except:
            return 0

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        metric = CategoricalHyperparameter("metric",['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis'], default_value='euclidean')
        k = UniformIntegerHyperparameter('k', 1, 100, default_value=3)
        cs.add_hyperparameters([metric, k])
        return cs
