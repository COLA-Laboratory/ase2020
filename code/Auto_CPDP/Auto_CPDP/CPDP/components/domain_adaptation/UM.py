import inspect
from collections import defaultdict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import Utils.cliffsDelta as cliff
from scipy.stats import mannwhitneyu
import numpy as np
import copy


# No context factor
from Auto_CPDP.CPDP.components.base import AutoAdaptation


class Universal(AutoAdaptation):
    def __init__(self, pvalue=0.05):
        self.p = pvalue

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

    def _compareMetricDistribution(self, x1, x2):
        s, p = mannwhitneyu(x1, x2)
        if p < self.p:
            sig_diff = 1
        else:
            sig_diff = 0
        return sig_diff

    def _quantifyDifference(self, x1, x2):
        d, res = cliff.cliffsDelta(x1, x2)
        return res

    def cluster(self, No_metric, numGroup, group):
        indexOfCluster = 0
        clusterOfGroup = np.zeros(numGroup)

        for i in range(0, numGroup - 1):
            indexNewCluster = indexOfCluster + 1
            for j in range(i + 1, numGroup):
                if self._compareMetricDistribution(group[i][:, No_metric], group[j][:, No_metric]) == 1:
                    if self._quantifyDifference(group[i][:, No_metric], group[j][:, No_metric]) == 'large':
                        indexOfCluster = indexNewCluster
                        clusterOfGroup[j] = indexOfCluster

        return clusterOfGroup

    def rankTransform(self, xsource, xtarget, loc):
        group = []
        for i in range(len(loc)):
            if i < len(loc) - 1:
                train = xsource[loc[i]:loc[i + 1]]
            else:
                train = xsource[loc[i]:]
            group.append(train)
        group.append(xtarget)
        resGroup = copy.deepcopy(group)

        for i in range(xsource.shape[1]):
            clusterIndex = self.cluster(i, len(loc) + 1, group)
            cluster = np.unique(clusterIndex)
            for item in cluster:
                tmp = np.asarray(np.where(clusterIndex == item))[0]
                tmp_data = np.asarray([])
                for ncs in tmp:
                    tmp_data = np.concatenate((tmp_data, group[int(ncs)][:, i]))

                percentiles = np.percentile(sorted(tmp_data), [10, 20, 30, 40, 50, 60, 70, 80, 90])
                for ncs in tmp:
                    ncs = int(ncs)
                    t = resGroup[ncs][:, i]
                    for it in range(len(t)):
                        if t[it] <= percentiles[0]:
                            resGroup[ncs][:, i][it] = 1
                        elif t[it] <= percentiles[1]:
                            resGroup[ncs][:, i][it] = 2
                        elif t[it] <= percentiles[2]:
                            resGroup[ncs][:, i][it] = 3
                        elif t[it] <= percentiles[3]:
                            resGroup[ncs][:, i][it] = 4
                        elif t[it] <= percentiles[4]:
                            resGroup[ncs][:, i][it] = 5
                        elif t[it] <= percentiles[5]:
                            resGroup[ncs][:, i][it] = 6
                        elif t[it] <= percentiles[6]:
                            resGroup[ncs][:, i][it] = 7
                        elif t[it] <= percentiles[7]:
                            resGroup[ncs][:, i][it] = 8
                        elif t[it] <= percentiles[8]:
                            resGroup[ncs][:, i][it] = 9
                        else:
                            resGroup[ncs][:, i][it] = 10

        return resGroup

    def run(self, Xsource, Ysource, Xtarget, Ytarget, loc):
        res = self.rankTransform(Xsource, Xtarget, loc)
        source = np.asarray(res[0])
        for i in range(1, len(loc)):
            source = np.concatenate((source, res[i]), axis=0)
        target = res[-1]

        return source, Ysource, target, Ytarget

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        p = UniformFloatHyperparameter('pvalue', 0.01, 0.1, default_value=0.05)
        cs.add_hyperparameter(p)
        return cs
