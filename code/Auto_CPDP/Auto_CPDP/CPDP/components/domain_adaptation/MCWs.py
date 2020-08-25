import inspect
from collections import defaultdict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import accuracy_score, roc_auc_score

from Auto_CPDP.CPDP.components.base import AutoAdaptation
from Utils.KMM import kmm
from cvxopt import matrix, solvers, spdiag
import numpy as np

"""
   In original paper, the author uses spectral clustering algorithm. But spectral clustering 
   algorithm cannot scale to large scale (more than 10000 dimension), so we replace it with k-means.

   KMM计算很耗时
"""


class MCWs(AutoAdaptation):
    def __init__(self, model, k=4, sigmma=1.0, lamb=1):
        self.k = k
        self.lamb = lamb
        self.gamma = sigmma
        self.model = model
        self.res = []

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

    def _clustering(self, x):
        # clustering = SpectralClustering(n_clusters=self.k)
        clustering = KMeans(n_clusters=self.k)
        clustering.fit(x)
        cluster = clustering.labels_

        res = []
        for i in range(self.k):
            res.append(np.where(cluster == i))
        return res

    def _compute_KMM_weight(self, x, train):
        weight = kmm(x, train, sigma=self.gamma)
        return weight

    def _getPerformance(self, x_validation, L_validation, alpha, ensemble):
        if len(alpha) == 0:
            return 0

        res = np.zeros(len(L_validation))

        prediction = np.zeros((len(ensemble), len(L_validation)))
        for i in range(len(ensemble)):
            prediction[i] = ensemble[i].predict(x_validation).reshape(1, -1)
        for i in range(len(L_validation)):
            res[i] = np.argmax(np.bincount(prediction[:, i].astype(np.int), weights=alpha))

        return roc_auc_score(L_validation, res)

    def _build_ensemble(self, xs, ys, train, l_train, test, l_test):
        cluster = self._clustering(xs)
        tmp_res = []
        tmp_prediction = []
        ensemble = []
        for item in cluster:
            x = np.concatenate((xs[item], train), axis=0)
            y = np.concatenate((ys[item], l_train), axis=0)
            weight = np.asarray(self._compute_KMM_weight(x, train)).reshape(-1)
            self.model.fit(x, y, sample_weight=weight)
            pd = self.model.predict(train)
            tmp_prediction.append(pd)
            tmp_res.append(accuracy_score(l_train, pd))
            ensemble.append(self.model)
        tmp_res = np.asarray(tmp_res)
        tmp_prediction = np.asarray(tmp_prediction)
        init_weight = tmp_res / np.sum(tmp_res)
        # optimize the weight of base classifiers
        A = matrix(np.ones((1, len(cluster))))
        m, n = A.size

        def F(x=None, z=None):
            if x is None: return 0, matrix(1.0, (n, 1))
            if min(x) <= 0.0: return None
            f = 0
            x1 = np.asarray(x).reshape(-1)
            for i in range(len(l_train)):
                f = f + self.lamb * np.linalg.norm(x1 - init_weight) + \
                    np.linalg.norm(np.dot(tmp_prediction[:, i].reshape(-1, 1), x1.reshape(1, -1)) - l_train[i])
            Df = -(x ** -1).T
            if z is None: return f, Df
            H = spdiag(z[0] * x ** -2)
            return f, Df, H

        optimal_weight = solvers.cp(F, A=A, b=matrix(np.ones((1, 1))), options={'show_progress': False})['x']
        optimal_weight = np.asarray(optimal_weight).reshape(-1)
        self.res.append(self._getPerformance(test, l_test, optimal_weight, ensemble))

    def run(self, Xs, Ys, test, l_test, train, l_train, loc):
        self.Xs = np.asarray(Xs)
        self.Ys = np.asarray(Ys)

        for i in range(len(loc)):
            if i < len(loc) - 1:
                xs = self.Xs[loc[i]:loc[i + 1]]
                ys = self.Ys[loc[i]:loc[i + 1]]
            else:
                xs = self.Xs[loc[i]:]
                ys = self.Ys[loc[i]:]

            self._build_ensemble(xs, ys, train, l_train, test, l_test)
        return np.mean(self.res)

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        k = UniformIntegerHyperparameter('k', 2, 11, default_value=4)
        sigmma = UniformFloatHyperparameter('sigmma', 0.01, 10, default_value=1.0)
        lamb = UniformFloatHyperparameter('lamb', 1e-6, 1e2, default_value=1.0)
        cs.add_hyperparameters([sigmma, k, lamb])
        return cs
