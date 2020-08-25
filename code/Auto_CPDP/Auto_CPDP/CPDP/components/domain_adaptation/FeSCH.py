import inspect
from collections import defaultdict

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from minepy import MINE
from info_gain import info_gain
import powerlaw, copy

from Auto_CPDP.CPDP.components.base import AutoAdaptation


class FeSCH(AutoAdaptation):
    def __init__(self, nt=1, strategy='SFD'):
        self.nt = nt
        self.strategy = strategy

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

    def _feature_clustering(self):
        x = np.concatenate((self.Xs, self.Xt), axis=0)
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(x.T)
        distance, idx = knn.kneighbors(x.T)
        dc = np.min(distance)
        dist = cdist(x.T, x.T, 'euclidean')
        ro = np.sum(dist - dc, axis=1) + dc
        self.ro = copy.deepcopy(ro)

        sigma = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            tmp = dist[i]
            if ro[i] == np.max(ro):
                sigma[i] = np.max(tmp)
            else:
                sigma[i] = np.min(tmp[ro > ro[i]])


        ro = ( ro - np.mean(ro) ) / np.std(ro)
        sigma = (sigma - np.mean(sigma)) / np.std(sigma)
        gamma = ro * sigma

        gamma_sort = sorted(gamma, reverse=True)
        res = powerlaw.find_xmin(gamma_sort)
        centers = np.where(gamma>res[0])[0]
        if len(centers) == x.shape[1]:
            return [centers]
        else:
            res = []
            tmp = x.T[centers]
            tmp1 = np.delete(x.T, centers, axis=0)
            idx = np.delete(range(x.shape[1]), centers)
            knn.fit(tmp)
            nn = knn.kneighbors(tmp1, n_neighbors=1, return_distance=False)


            for i in centers:
                res.append([i])
            for i in range(nn.shape[0]):
                res[centers[nn[i][0]]].append(idx[i])
            return res

    def _feature_selection(self):
        cluster = self._feature_clustering()

        res = []
        for item in cluster:
            num = int(np.ceil(len(item)*self.nt / self.Xs.shape[1]))

            if self.strategy == 'LDF':

                tmp = np.argsort(self.ro[item])[-num:]
                tmp = np.asarray(item)[tmp]
                res = np.concatenate((res, tmp), axis=0)

            if self.strategy == 'SFD':
                mine = MINE()
                score = []
                length = min(len(self.Xs.T[0]), len(self.Xt.T[0]))
                for it in item:
                    mine.compute_score(self.Xs.T[it][:length], self.Xt.T[it][:length])
                    tmp = mine.mic()
                    score.append(tmp)
                res = np.concatenate((res, np.asarray(item)[np.argsort(score)[-num:]]), axis=0)

            if self.strategy == 'FCR':
                score = []
                for it in item:
                    tmp = info_gain.info_gain(list(self.Xs.T[it]), list(self.Ys))
                    score.append(tmp)
                res = np.concatenate((res, np.asarray(item)[np.argsort(score)[-num:]]), axis=0)
        return res

    def run(self, Xs, Ys, Xt, Yt):
        self.Xs = np.asarray(Xs)
        self.Ys = np.asarray(Ys)
        self.Xt = np.asarray(Xt)
        self.Yt = np.asarray(Yt)

        idx = self._feature_selection().astype(np.int)
        return self.Xs.T[idx].T, self.Ys, self.Xt.T[idx].T, self.Yt

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        strategy = CategoricalHyperparameter("strategy", ['SFD', 'LDF', 'FCR'],default_value='SFD')
        nt = UniformIntegerHyperparameter('nt', 1, 61, default_value=1)
        cs.add_hyperparameters([strategy, nt])
        return cs


