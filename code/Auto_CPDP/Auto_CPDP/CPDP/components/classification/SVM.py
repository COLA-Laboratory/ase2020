import resource
import sys

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter

from Auto_CPDP.CPDP.components.base import AutoClassifier
from Auto_CPDP.CPDP.implementations.util import softmax
from Auto_CPDP.util.common import check_for_bool, check_none

from Auto_CPDP.CPDP.constants import PREDICTIONS, SPARSE, DENSE, UNSIGNED_DATA


class LibSVM_SVC(AutoClassifier):
    def __init__(self, C, kernel, gamma,
                degree=3, coef0=0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.estimator = None

    def fit(self, X, Y, sample_weight=None):
        import sklearn.svm

        # Calculate the size of the kernel cache (in MB) for sklearn's LibSVM. The cache size is
        # calculated as 2/3 of the available memory (which is calculated as the memory limit minus
        # the used memory)
        try:
            # Retrieve memory limits imposed on the process
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)

            if soft > 0:
                # Convert limit to units of megabytes
                soft /= 1024 * 1024

                # Retrieve memory used by this process
                maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024

                # In MacOS, the MaxRSS output of resource.getrusage in bytes; on other platforms,
                # it's in kilobytes
                if sys.platform == 'darwin':
                    maxrss = maxrss / 1024

                cache_size = (soft - maxrss) / 1.5

                if cache_size < 0:
                    cache_size = 200
            else:
                cache_size = 200
        except Exception:
            cache_size = 200

        self.C = float(self.C)
        if self.degree is None:
            self.degree = 3
        else:
            self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)


        self.estimator = sklearn.svm.SVC(C=self.C,
                                         kernel=self.kernel,
                                         degree=self.degree,
                                         gamma=self.gamma,
                                         coef0=self.coef0,
                                         max_iter=10,
                                         cache_size=cache_size)
        self.estimator.fit(X, Y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        decision = self.estimator.decision_function(X)
        return softmax(decision)


    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LibSVM-SVC',
            'name': 'LibSVM Support Vector Classification',
            'handles_regression': False,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': False,
            'is_deterministic': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        C = UniformFloatHyperparameter("C", 0.001, 10,
                                       default_value=1.0)
        # No linear kernel here, because we have liblinear
        kernel = CategoricalHyperparameter(name="kernel",
                                           choices=["rbf", "poly", "sigmoid", 'linear'],
                                           default_value="linear")
        degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)
        gamma = UniformFloatHyperparameter("gamma", 0.001, 100, default_value=1)

        coef0 = UniformFloatHyperparameter("coef0", 0, 10, default_value=0)


        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, coef0])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cond = InCondition(gamma, kernel, ['poly', 'sigmoid', 'rbf'])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)
        cs.add_condition(cond)

        return cs