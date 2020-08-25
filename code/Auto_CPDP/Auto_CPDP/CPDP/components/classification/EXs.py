import sklearn

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from Auto_CPDP.CPDP.components.base import AutoClassifier
from Auto_CPDP.CPDP.implementations.util import convert_multioutput_multiclass_to_multilabel


class ExtraTreesClassifier(AutoClassifier):

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split,  max_features):

        self.n_estimators = int(n_estimators)
        self.estimator_increment = 10
        if criterion not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): "
                             "%s" % criterion)
        self.criterion = criterion


        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = float(max_features)
        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        self.estimator = sklearn.ensemble.ExtraTreesClassifier(n_estimators=self.n_estimators,
                                                               criterion=self.criterion, min_samples_split=self.min_samples_split,
                                                            min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        self.estimator.fit(X, y, sample_weight)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ET',
                'name': 'Extra Trees Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter('n_estimators', 10, 200, 100)
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini")

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter(
            "max_features", 0., 1., default_value=0.5)

        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        cs.add_hyperparameters([n_estimators, criterion, max_features,
                                min_samples_split, min_samples_leaf])

        return cs