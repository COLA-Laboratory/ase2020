import inspect
from collections import defaultdict

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from Auto_CPDP.CPDP.components.base import AutoAdaptation


class HISNN(AutoAdaptation):
    def __init__(self, model, MinHam=1.0):
        self.MinHam = MinHam
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

    def _MahalanobisDist(self, data, base):

        covariance = np.cov(base.T)  # calculate the covarince matrix
        inv_covariance = np.linalg.pinv(covariance)
        mean = np.mean(base, axis=0)
        dist = np.zeros((np.asarray(data)).shape[0])
        for i in range(dist.shape[0]):
            dist[i] = distance.mahalanobis(data[i], mean, inv_covariance)
        return dist

    def _TrainInstanceFiltering(self):
        # source outlier remove based on source
        dist = self._MahalanobisDist(self.Xsource, self.Xsource)
        threshold = np.mean(dist) * 3 * np.std(dist)
        outliers = []
        for i in range(len(dist)):
            if dist[i] > threshold:
                outliers.append(i)  # index of the outlier
        self.Xsource = np.delete(self.Xsource, outliers, axis=0)
        self.Ysource = np.delete(self.Ysource, outliers, axis=0)

        # source outlier remove based on target
        dist = self._MahalanobisDist(self.Xsource, self.Xtarget)
        threshold = np.mean(dist) * 3 * np.std(dist)
        outliers = []
        for i in range(len(dist)):
            if dist[i] > threshold:
                outliers.append(i)  # index of the outlier
        self.Xsource = np.delete(self.Xsource, outliers, axis=0)
        self.Ysource = np.delete(self.Ysource, outliers, axis=0)


        # NN filter for source data based on target
        neigh = NearestNeighbors(radius=self.MinHam, metric='hamming')
        neigh.fit(self.Xsource)
        res = neigh.radius_neighbors(self.Xtarget, return_distance=False)

        tmp = np.concatenate((self.Xsource, self.Ysource.reshape(-1, 1)), axis=1)
        x = tmp[res[0]]
        for item in res[1:]:
            x = np.concatenate((x, tmp[item]), axis=0)
            x = np.unique(x, axis=0)

        self.Xsource = x[:, :-1]
        self.Ysource = x[:, -1]

    def predict(self):
        predict = np.zeros(self.Xtarget.shape[0])
        neigh = NearestNeighbors(radius=self.MinHam, metric='hamming')
        neigh.fit(self.Xsource)
        res = neigh.radius_neighbors(self.Xtarget, return_distance=False)
        for i in range(res.shape[0]):
            # case 1
            if len(res[i]) == 1:
                subRes = neigh.radius_neighbors(self.Xsource[res[i][0]])
                # case 1-1
                if len(subRes) == 1:
                    predict[i] = self.model.predict(self.Xtarget[i].reshape(1, -1))
                else:
                    tmp = np.unique(self.Ysource[subRes])
                    # case 1-2
                    if len(tmp) == 1:
                        predict[i] = tmp[0]
                    # case 1-3
                    else:
                        predict[i] = self.model.predict(self.Xtarget[i].reshape(1, -1))
            else:
                tmp = np.unique(self.Ysource[res[i]])
                # case 2
                if len(tmp) == 1:
                    predict[i] = tmp[0]
                # case 3
                else:
                    predict[i] = self.model.predict(self.Xtarget[i].reshape(1, -1))

        self.AUC = roc_auc_score(self.Ytarget, predict)

    def run(self, Xs, Ys, Xt, Yt, loc):
        self.Xsource = np.asarray(Xs)
        self.Ysource = np.asarray(Ys)
        self.Xtarget = np.asarray(Xt)
        self.Ytarget = np.asarray(Yt)

        self._TrainInstanceFiltering()
        try:
            self.model.fit(np.log(self.Xsource + 1), self.Ysource)
            self.predict()
            return self.AUC
        except:
            return 0

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        minham = UniformFloatHyperparameter('MinHam', 1, 61, default_value=1)
        cs.add_hyperparameter(minham)
        return cs