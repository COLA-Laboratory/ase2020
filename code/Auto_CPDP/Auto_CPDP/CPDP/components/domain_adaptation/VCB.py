import inspect
from collections import defaultdict

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from Auto_CPDP.CPDP.components.base import AutoAdaptation


class VCB(AutoAdaptation):
    def __init__(self, model, M=30, lamb=1.0):
        self.model = model
        self.M = M
        self.lamb = lamb
        self.ensemble = []

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

    # compute the similarity weight
    def _computeDCV(self, x):
        Max = np.max(self.Xtarget, axis=0)
        Min = np.min(self.Xtarget, axis=0)

        a = x < Max
        b = x > Min
        weight = (np.logical_and(a, b)).astype(int)
        weight = np.average(weight, axis=1)
        return weight

    def _over_sampling(self, x, num, w):
        knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
        knn.fit(x)
        neighbors = knn.kneighbors(x, return_distance=False)
        idx = np.random.choice(x.shape[0], num, replace=False)
        for i in idx:
            i = int(i)
            rnd = int(neighbors[i][int(np.random.choice(10, 1))])
            xnew = x[i] + np.random.random() * (x[i] - x[rnd])
            x = np.concatenate((x, xnew.reshape(1, -1)), axis=0)
            w = np.concatenate((w, [1]), axis=0)
        return x, w

    def _under_sampling(self, x, x_min, num, w):
        knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
        knn.fit(x_min)
        for i in range(num):
            distance, neighbors = knn.kneighbors(x)
            distance = np.sum(distance, axis=1)
            idx = np.argsort(distance)
            x = np.delete(x, idx[0], axis=0)
            w = np.delete(w, idx[0], axis=0)
        return x, w

    def _resampling(self, x, sw, t, w):
        """
        x: input (label included)
        sw: similarity weight
        t: label
        w: weight coefficient
        """
        # divide into similar and different subsets (x, t, w)
        xs = x[sw == 1]
        w_s = w[sw == 1]
        t_s = t[sw == 1]
        xd = x[sw != 1]
        w_d = w[sw != 1]
        t_d = t[sw != 1]

        # divide into major and minor subsets
        xs_min = xs[t_s == 1]
        w_smin = w_s[t_s == 1]
        xs_maj = xs[t_s == 0]
        w_smaj = w_s[t_s == 0]
        xd_min = xd[t_d == 1]
        w_dmin = w_d[t_d == 1]
        xd_maj = xd[t_d == 0]
        w_dmaj = w_d[t_d == 0]

        num = int(xs_min.shape[0] / 2)
        if xs_min.shape[0] > (10 + num):
            xs_over = xs_min[np.argsort(w_smin)[num:]]
            xs1 = xs_min[np.argsort(w_smin)[:num]]
            num1 = int(xs_maj.shape[0] / 2)
            xs_under = xs_maj[np.argsort(w_smaj)[:num1]]
            xs2 = xs_maj[np.argsort(w_smaj)[num1:]]
            if xs_under.shape[0] > int(xs1.shape[0]/2) and xs_over.shape[0] > int(xs1.shape[0]/2):
                xs_under, w1 = self._under_sampling(xs_under, xs_min, int(xs1.shape[0]/2), w_smaj[np.argsort(w_smaj)[:num1]])
                xs_over, w2 = self._over_sampling(xs_over, int(xs1.shape[0]/2), w_smin[np.argsort(w_smin)[num:]])
                xs = np.concatenate((xs1, xs2, xs_over, xs_under), axis=0)
                t_s = np.concatenate((np.ones(xs1.shape[0]), np.zeros(xs2.shape[0]),
                                      np.ones(xs_over.shape[0]), np.zeros(xs_under.shape[0])), axis=0)
                w_s = np.concatenate((w_smin[np.argsort(w_smin)[:num]],
                                      w_smaj[np.argsort(w_smaj)[num1:]],
                                      w1, w2), axis=0)

        num = int(xd_min.shape[0] / 2)
        if xd_min.shape[0] > (10 + num):
            xd_over = xd_min[np.argsort(w_dmin)[num:]]
            xd1 = xd_min[np.argsort(w_dmin)[:num]]
            num1 = int(xd_maj.shape[0] / 2)
            xd_under = xd_maj[np.argsort(w_dmaj)[:num1]]
            xd2 = xd_maj[np.argsort(w_dmaj)[num1:]]
            if xd_under.shape[0] > int(xd1.shape[0] / 2) and xd_over.shape[0] > int(xd1.shape[0] / 2):
                xd_under, w1 = self._under_sampling(xd_under, xd_min, int(xd1.shape[0] / 2),
                                                    w_dmaj[np.argsort(w_dmaj)[:num1]])
                xd_over, w2 = self._over_sampling(xd_over, int(xd1.shape[0] / 2), w_dmin[np.argsort(w_dmin)[num:]])
                xd = np.concatenate((xd1, xd2, xd_over, xd_under), axis=0)
                t_d = np.concatenate((np.ones(xd1.shape[0]), np.zeros(xd2.shape[0]),
                                      np.ones(xd_over.shape[0]), np.zeros(xd_under.shape[0])), axis=0)
                w_d = np.concatenate((w_dmin[np.argsort(w_dmin)[:num]],
                                      w_dmaj[np.argsort(w_dmaj)[num1:]],
                                      w1, w2), axis=0)

        x = np.concatenate((xs, xd), axis=0)
        t = np.concatenate((t_s, t_d), axis=0)
        w = np.concatenate((w_s, w_d), axis=0)
        return x, t, w

    def _getPerformance(self, x_validation, L_validation, alpha, T=None):
        if len(alpha) == 0:
            return 0

        res = np.zeros(len(L_validation))
        if T is None:
            prediction = np.zeros(( len(self.ensemble), len(L_validation) ))
            for i in range(len(self.ensemble)):
                prediction[i] = self.ensemble[i].predict(x_validation).reshape(1, -1)
            for i in range(len(L_validation)):
                res[i] = np.argmax(np.bincount(prediction[:, i].astype(np.int), weights=alpha))

        else:
            prediction = np.zeros((len(self.ensemble), len(L_validation)))
            for i in range(T+1):
                prediction[i] = self.ensemble[i].predict(x_validation)
            for i in range(len(L_validation)):
                res[i] = np.argmax(np.bincount(prediction[:, i].astype(np.int), weights=alpha))
        return roc_auc_score(L_validation, res)

    def run(self, Xs, Ys, Xt, Yt, loc):
        self.Xsource = np.asarray(Xs)
        self.Xtarget = np.asarray(Xt)
        self.Ysource = np.asarray(Ys)
        self.Ytarget = np.asarray(Yt)
        SW = self._computeDCV(self.Xsource)
        x = np.concatenate((self.Xsource, SW.reshape(-1, 1), self.Ysource.reshape(-1, 1)), axis=1)

        # select instances for validation
        train, test = train_test_split(x, test_size=0.5, random_state=42)
        train = np.asarray(sorted(train, key=lambda x:x[-2], reverse=True))
        x_validation = train[: int(train.shape[0] * 0.4), :]
        # SW_validation = x_validation[:, -2]
        L_validation = x_validation[:, -1]
        x_train = np.concatenate((test, train[int(train.shape[0] * 0.4):, :]), axis=0)
        SW_train = x_train[:, -2]
        L_train = x_train[:, -1]
        x_validation = x_validation[:, :-2]
        x_train = x_train[:, :-2]

        # initialize
        w_coefficient = np.ones(x_train.shape[0])
        ro_best = 0
        T = 0
        eta = np.zeros(self.M)
        alpha = []
        ro = np.zeros(self.M)

        for m in range(self.M):
            if m > 0:
                SW_train = self._computeDCV(x_train)
                x_train, L_train, w_coefficient = self._resampling(x_train, SW_train, L_train, w_coefficient)
            try:
                self.model.fit(x_train, L_train, sample_weight=w_coefficient)
            except:
                continue
            self.ensemble.append(self.model)
            prediction = self.model.predict(x_train)
            a = np.zeros(prediction.shape[0])
            a[prediction!=L_train] = 1
            eta[m] = (a * w_coefficient).sum() / w_coefficient.sum()
            alpha.append( self.lamb * np.log(1/eta[m] - 1) )
            w_coefficient = w_coefficient * np.exp(alpha[m]*a)
            ro[m] = self._getPerformance(x_validation, L_validation, alpha)
            if ro[m] >= ro_best:
                T = m
                ro_best  = ro[m]

        return self._getPerformance(Xt, Yt, alpha, T=T)

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        m = UniformIntegerHyperparameter('M', 2, 30, default_value=30)
        lamb = UniformFloatHyperparameter('lamb', 0.5, 1.5, default_value=1.0)
        cs.add_hyperparameters([m, lamb])
        return cs



