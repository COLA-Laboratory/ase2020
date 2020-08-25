import inspect
from collections import defaultdict

import numpy as np
import random, copy

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from Auto_CPDP.CPDP.components.base import AutoAdaptation


class FSS_bagging(AutoAdaptation):
    def __init__(self, model, topN=10, FSS=0.1, score_thre=0.5):
        self.topN = topN
        self.FSS = FSS
        self.score_thre = score_thre
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

    def _sample(self, Xsource, Xtarget):
        K = min(500, Xsource.shape[0], Xtarget.shape[0])
        Ltrain = np.ones(K)
        Ltest = np.ones(K) * -1

        Train = random.sample(range(Xsource.shape[0]), Xsource.shape[0] - K)
        Test = random.sample(range(Xtarget.shape[0]), Xtarget.shape[0] - K)
        Train = np.delete(Xsource, Train, axis=0)
        Test = np.delete(Xtarget, Test, axis=0)

        data = np.concatenate((Train, Test), axis=0)
        label = np.concatenate((Ltrain, Ltest), axis=0)

        return data, label

    def _calDistance(self, Xsource, Xtarget):
        acc = np.zeros(10)
        for i in range(10):
            x, y = self._sample(Xsource, Xtarget)
            acc[i] = np.mean(cross_val_score(self.model, x, y, scoring='accuracy', cv=5))
        return 2 * abs((np.mean(acc) - 0.5))

    def _remove_unstable_feature(self, x, xtarget):
        fx, fy = self._sample(x, xtarget)
        lr = LogisticRegression()
        lr.fit(fx, fy)
        coef = dict()
        for i in range(x.shape[1]):
            coef[i] = lr.coef_[0][i]
        coef = sorted(coef.items(), key=lambda d: d[1], reverse=True)

        dump = []
        for i in range(int(x.shape[1] * self.FSS)):
            dump.append(coef[i][0])

        x = np.delete(x, dump, axis=1)
        xtarget = np.delete(xtarget, dump, axis=1)
        return x, xtarget

    def run(self, Xsource, Ysource, Xtarget, Ytarget, loc):
        prediction = dict()
        self.topN = min(self.topN, len(loc))
        dist = dict()

        for i in range(len(loc)):
            if i < len(loc) - 1:
                train = Xsource[loc[i]:loc[i + 1]]
                dist[i] = self._calDistance(train, Xtarget)
            else:
                train = Xsource[loc[i]:]
                dist[i] = self._calDistance(train, Xtarget)

        dist = sorted(dist.items(), key=lambda d: d[1])

        for i in range(self.topN):
            xt = copy.deepcopy(Xtarget)
            index = dist[i][0]
            if index < len(loc) - 1:
                tmp = Xsource[loc[index]:loc[index + 1]]
                temp = Ysource[loc[index]:loc[index + 1]]
            else:
                tmp = Xsource[loc[index]:]
                temp = Ysource[loc[index]:]
            tmp, xt = self._remove_unstable_feature(tmp, xt)
            try:
                self.model.fit(tmp, temp)
                prediction[i] = self.model.predict(xt)
            except:
                prediction[i] = np.random.randint(0, 2, xt.shape[0])

        res = np.asarray(list(prediction.values()))
        res = res.mean(axis=0)
        res[res > self.score_thre] = 1
        res[res <= self.score_thre] = 0
        return roc_auc_score(Ytarget, res)

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        topN = UniformIntegerHyperparameter('topN', 1, 11, default_value=10)
        FSS = UniformFloatHyperparameter('FSS', 0.1, 0.9, default_value=0.1)
        score_thre = UniformFloatHyperparameter('score_thre', 0.3, 0.7, default_value=0.5)
        cs.add_hyperparameters([topN, FSS, score_thre])
        return cs
