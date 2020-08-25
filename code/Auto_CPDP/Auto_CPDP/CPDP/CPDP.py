from ConfigSpace.configuration_space import ConfigurationSpace
from .create_searchspace_util import in_ex_cludeHandler
from sklearn.utils.validation import check_random_state
from .components.domain_adaptation import get_domain_adaptation
from .components.classification import get_classification
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np


class CPDP_pipeline():
    def __init__(self, config=None, dataset_properties=None,
                 include=None, exclude=None, random_state=None,
                 init_params=None, repeat=1):

        self._init_params = init_params if init_params is not None else {}
        self.include_ = include if include is not None else {}
        self.exclude_ = exclude if exclude is not None else {}
        self.dataset_properties_ = dataset_properties if \
            dataset_properties is not None else {}
        self.repeat = repeat

        if config is None:
            # 这个方法应该是configspace库的方法
            self.steps = self._get_pipeline()
            self.config_space = self.get_hyperparameter_search_space()
            self.configuration_ = self.config_space.get_default_configuration()
        else:
            # if isinstance(config, dict):
            #     config = Configuration(self.config_space, config)
            #
            # if self.config_space != config.configuration_space:
            #     print(self.config_space._children)
            #     print(config.configuration_space._children)
            #     import difflib
            #     diff = difflib.unified_diff(
            #         str(self.config_space).splitlines(),
            #         str(config.configuration_space).splitlines())
            #     diff = '\n'.join(diff)
            #     raise ValueError('Configuration passed does not come from the '
            #                      'same configuration space. Differences are: '
            #                      '%s' % diff)
            self.configuration_ = config

        # self.set_hyperparameters(self.configuration_, init_params=init_params)

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

        self._additional_run_info = {}

    def get_hyperparameter_search_space(self):
        """Return the configuration space for the CASH problem.

        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the AutoSklearnClassifier.

        """
        if not hasattr(self, 'config_space') or self.config_space is None:
            self.config_space = self._get_hyperparameter_search_space(
                include=self.include_, exclude=self.exclude_,
                dataset_properties=self.dataset_properties_)
        return self.config_space

    def _get_hyperparameter_search_space(self, include=None, exclude=None,
                                         dataset_properties=None):
        """Create the hyperparameter configuration space.

        Parameters
        ----------
        include : dict (optional, default=None)

        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the SimpleRegressionClassifier.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()
        if not 'target_type' in dataset_properties:
            dataset_properties['target_type'] = 'classification'
        if dataset_properties['target_type'] != 'classification':
            dataset_properties['target_type'] = 'classification'

        if 'sparse' not in dataset_properties:
            # This dataset is probaby dense
            dataset_properties['sparse'] = False

        # 获取configuration space
        cs = self._get_base_search_space(
            cs=cs, dataset_properties=dataset_properties,
            exclude=exclude, include=include, pipeline=self.steps)

        self.configuration_space_ = cs
        return cs

    def _get_base_search_space(self, cs, dataset_properties, exclude,
                               include, pipeline):
        if include is None:
            if self.include_ is None:
                include = {}
            else:
                include = self.include_

        if exclude is None:
            if self.exclude_ is None:
                exclude = {}
            else:
                exclude = self.exclude_

        # pipeline: {'classifier': {'DS': DS, 'Bruak': Bruak}}, 构成configuration space
        pipeline = in_ex_cludeHandler(include=include, exclude=exclude, pipeline=pipeline)
        for type, val in pipeline:
            sub_config_space = ConfigurationSpace()
            estimator = CategoricalHyperparameter('__choice__', list(val.keys()), default_value=list(val.keys())[0])
            sub_config_space.add_hyperparameter(estimator)
            for k, v in val.items():
                estimator_cs = v.get_hyperparameter_search_space()
                parent_hyperparameter = {'parent': estimator,
                                         'value': k}
                sub_config_space.add_configuration_space(k, estimator_cs, parent_hyperparameter=parent_hyperparameter)
            cs.add_configuration_space(type, sub_config_space)

        return cs

    # 获取所有的算法组合！！dict
    @staticmethod
    def _get_pipeline():
        steps = []

        default_dataset_properties = {'target_type': 'classification'}

        # 添加domain adaptation的部分
        steps.append(['domain_adaptation', get_domain_adaptation()])
        # Add the classification component
        steps.append(['classifier',
                      get_classification()])
        return steps

    def get_additional_run_info(self):
        """Allows retrieving additional run information from the pipeline.

        Can be overridden by subclasses to return additional information to
        the optimization algorithm.
        """
        return self._additional_run_info

    def run(self, xsource, ysource, xtarget, ytarget, loc):
        isGroup = 0

        dict_config = self.configuration_.get_dictionary()
        pipeline = self._get_pipeline()
        # print(dict_config)
        clf = dict_config['classifier:__choice__']
        adptname = dict_config['domain_adaptation:__choice__']

        cparams = dict()
        aparams = dict()
        for k, v in dict_config.items():
            if 'classifier:' + clf in k:
                cparams[k.split(':')[-1]] = v
            if 'domain_adaptation:' + adptname in k:
                aparams[k.split(':')[-1]] = v

        if adptname in ['CDE_SMOTE', 'FSS_bagging', 'GIS', 'HISNN', 'MCWs', 'VCB']:
            isGroup = 1
            clf = pipeline[1][1][clf]().set_params(**cparams)
            aparams['model'] = clf
            adpt = pipeline[0][1][adptname]().set_params(**aparams)
        else:
            clf = pipeline[1][1][clf]().set_params(**cparams)
            adpt = pipeline[0][1][adptname]().set_params(**aparams)

        res = np.zeros(self.repeat)
        for i in range(self.repeat):
            if isGroup:
                if adptname == 'MCWs':
                    train, test, ytrain, ytest = train_test_split(xtarget, ytarget, test_size=0.9, random_state=42)
                    res[i] = adpt.run(xsource, ysource, train, ytrain, test, ytest, loc)
                else:
                    res[i] = adpt.run(xsource, ysource, xtarget, ytarget, loc)
            else:
                xsource, ysource, xtarget, ytarget = adpt.run(xsource, ysource, xtarget, ytarget, loc)
                try:
                    clf.fit(xsource, ysource)
                    predict = clf.predict(xtarget)
                    res[i] = roc_auc_score(ytarget, predict)
                except:
                    res[i] = 0

        return 1 - np.mean(res)
