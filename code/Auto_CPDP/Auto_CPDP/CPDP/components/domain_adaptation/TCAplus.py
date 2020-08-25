import inspect
from collections import defaultdict

import scipy.spatial.distance as dist
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale
import scipy, sklearn
import numpy as np

from Auto_CPDP.CPDP.components.base import AutoAdaptation

""" transformate into latent space """


def kernel(ker, X1, X2, gamma):
    X1[X1 != X1] = 0
    X1[X1 == np.inf] = 0
    X1[X1 == np.inf] = np.max(X1)

    if X2 is not None:
        X2[X2 != X2] = 0
        X2[X2 == np.inf] = 0
        X2[X2 == np.inf] = np.max(X2)


    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    K[K != K] = 0
    K[K == np.inf] = 0
    K[K == np.inf] = np.max(K)
    return K


class TCA(AutoAdaptation):
    def __init__(self, kernel_type='primal', dim=5, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf' | 'sam'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

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

    def _normalization(self, type):
        if type == 'N1':
            # normalization for source data
            self.Xsource = minmax_scale(self.Xsource)

            # normalization for target data
            self.Xtarget = minmax_scale(self.Xtarget)

        elif type == 'N2':
            # normalization for source data
            self.Xsource = zscore(self.Xsource)

            # normalization for target data
            self.Xtarget = zscore(self.Xtarget)

        elif type == 'N3':
            # normalization for source data
            mean = np.mean(self.Xsource, axis=0)
            std = np.std(self.Xsource, axis=0)
            self.Xsource = (self.Xsource - mean) / std

            # normalization for target data
            self.Xtarget = (self.Xtarget - mean) / std

        elif type == 'N4':
            mean = np.mean(self.Xtarget, axis=0)
            std = np.std(self.Xtarget, axis=0)
            # normalization for target data
            self.Xtarget = (self.Xtarget - mean) / std

            # normalization for source data
            self.Xsource = (self.Xsource - mean) / std

        elif type == 'N0':
            return

    def _computDCV(self):
        ss = self.Xsource.shape
        tt = self.Xtarget.shape
        SDCV = []
        TDCV = []

        # compute DCV (dataset characteristic vector) of source dataset
        Sdist = dist.cdist(self.Xsource, self.Xsource, metric='euclidean')
        Sdist = np.tril(Sdist, -1).reshape(-1)
        Sdist = Sdist[Sdist != 0]

        SDCV.append(np.mean(Sdist))
        SDCV.append(np.median(Sdist))
        SDCV.append(np.min(Sdist))
        SDCV.append(np.max(Sdist))
        SDCV.append(np.std(Sdist))
        SDCV.append(ss[0])

        # compute DCV (dataset characteristic vector) of target dataset
        Tdist = dist.cdist(self.Xtarget, self.Xtarget, metric='euclidean')
        Tdist = np.tril(Tdist, -1).reshape(-1)
        Tdist = Tdist[Tdist != 0]

        TDCV.append(np.mean(Tdist))
        TDCV.append(np.median(Tdist))
        TDCV.append(np.min(Tdist))
        TDCV.append(np.max(Tdist))
        TDCV.append(np.std(Tdist))
        TDCV.append(tt[0])

        return np.asarray(SDCV), np.asarray(TDCV)

    def _chooseNormalization(self):
        SDCV, TDCV = self._computDCV()

        nominal = []
        for i in range(0, len(SDCV)):
            if SDCV[i] * 1.6 < TDCV[i]:
                nominal.append('much-more')
            elif TDCV[i] < SDCV[i] * 0.4:
                nominal.append('much-less')
            elif (SDCV[i] * 1.3 < TDCV[i]) and (TDCV[i] <= SDCV[i] * 1.6):
                nominal.append('more')
            elif (SDCV[i] * 1.1 < TDCV[i]) and (TDCV[i] <= SDCV[i] * 1.3):
                nominal.append('slight-more')
            elif (SDCV[i] * 0.9 <= TDCV[i]) and (TDCV[i] <= SDCV[i] * 1.1):
                nominal.append('same')
            elif (SDCV[i] * 0.7 <= TDCV[i]) and (TDCV[i] < SDCV[i] * 0.9):
                nominal.append('slight-less')
            elif (SDCV[i] * 0.4 <= TDCV[i]) and (TDCV[i] < SDCV[i] * 0.7):
                nominal.append('less')

        if (nominal[5] == nominal[2] == nominal[3] == 'much-less') or (
                nominal[5] == nominal[2] == nominal[3] == 'much-more'):
            self._normalization('N1')

        elif ((nominal[4] == 'much-more') and (TDCV[5] < SDCV[5])) or (
                (nominal[4] == 'much-less') and (TDCV[5] > SDCV[5])):
            self._normalization('N3')

        elif ((nominal[4] == 'much-more') and (TDCV[5] > SDCV[5])) or (
                (nominal[4] == 'much-less') and (TDCV[5] < SDCV[5])):
            self._normalization('N4')

        elif nominal[0] == nominal[4] == 'same':
            self._normalization('N0')

        else:
            self._normalization('N2')

    def run(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        self.Xsource = Xs
        self.Xtarget = Xt
        self._chooseNormalization()
        Xs = self.Xsource
        Xt = self.Xtarget

        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

        return Xs_new, Ys, Xt_new, Yt

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        kernel_type = CategoricalHyperparameter('kernel_type', ['primal', 'linear', 'rbf', 'sam'], default_value='linear')
        dim = UniformIntegerHyperparameter('dim', 5, 61, default_value=5)
        lamb = UniformFloatHyperparameter('lamb', 1e-6, 1e2, default_value=1.0)
        sigmma = UniformFloatHyperparameter('gamma', 1e-5, 1e2, default_value=1.0)
        cs.add_hyperparameters([kernel_type, dim, lamb, sigmma])
        return cs