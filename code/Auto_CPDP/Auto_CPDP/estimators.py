# -*- encoding: utf-8 -*-
from typing import Optional, List, Dict

import numpy as np
from sklearn.utils.multiclass import type_of_target

from Auto_CPDP.automl import AutoMLCPDP, BaseAutoML
from Auto_CPDP.util.backend import create


def _fit_automl(automl, kwargs, load_models):
    return automl.fit(load_models=load_models, **kwargs)


class AutoEstimator():

    def __init__(
            self,
            maxFE = None,
            time_left_for_this_task=3600,
            per_run_time_limit=360,
            initial_configurations_via_metalearning=25,
            seed=1,
            ml_memory_limit=3072,
            include_estimators=None,
            exclude_estimators=None,
            include_adpts=None,
            exclude_adpts=None,
            tmp_folder=None,
            output_folder=None,
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True,
            shared_mode=False,
            n_jobs: Optional[int] = None,
            disable_evaluator_output=False,
            get_smac_object_callback=None,
            smac_scenario_args=None,
            logging_config=None,
            metadata_directory=None,
            repeat=1,
    ):
        """
        Parameters
        ----------
        time_left_for_this_task : int, optional (default=3600)
            Time limit in seconds for the search of appropriate
            models. By increasing this value, *auto-sklearn* has a higher
            chance of finding better models.

        per_run_time_limit : int, optional (default=360)
            Time limit for a single call to the machine learning model.
            Model fitting will be terminated if the machine learning
            algorithm runs over the time limit. Set this value high enough so
            that typical machine learning algorithms can be fit on the
            training data.

        initial_configurations_via_metalearning : int, optional (default=25)
            Initialize the hyperparameter optimization algorithm with this
            many configurations which worked well on previously seen
            datasets. Disable if the hyperparameter optimization algorithm
            should start from scratch.


        seed : int, optional (default=1)
            Used to seed SMAC. Will determine the output file names.

        ml_memory_limit : int, optional (3072)
            Memory limit in MB for the machine learning algorithm.
            `auto-sklearn` will stop fitting the machine learning algorithm if
            it tries to allocate more than `ml_memory_limit` MB.

        include_estimators : list, optional (None)
            If None, all possible estimators are used. Otherwise specifies
            set of estimators to use.

        exclude_estimators : list, optional (None)
            If None, all possible estimators are used. Otherwise specifies
            set of estimators not to use. Incompatible with include_estimators.

        include_preprocessors : list, optional (None)
            If None all possible preprocessors are used. Otherwise specifies set
            of preprocessors to use.

        exclude_preprocessors : list, optional (None)
            If None all possible preprocessors are used. Otherwise specifies set
            of preprocessors not to use. Incompatible with
            include_preprocessors.

        resampling_strategy : string or object, optional ('holdout')
            how to to handle overfitting, might need 'resampling_strategy_arguments'

            * 'holdout': 67:33 (train:test) split
            * 'holdout-iterative-fit':  67:33 (train:test) split, calls iterative
              fit where possible
            * 'cv': crossvalidation, requires 'folds'
            * 'partial-cv': crossvalidation with intensification, requires
              'folds'
            * BaseCrossValidator object: any BaseCrossValidator class found
                                        in scikit-learn model_selection module
            * _RepeatedSplits object: any _RepeatedSplits class found
                                      in scikit-learn model_selection module
            * BaseShuffleSplit object: any BaseShuffleSplit class found
                                      in scikit-learn model_selection module

        resampling_strategy_arguments : dict, optional if 'holdout' (train_size default=0.67)
            Additional arguments for resampling_strategy:

            * ``train_size`` should be between 0.0 and 1.0 and represent the
              proportion of the dataset to include in the train split.
            * ``shuffle`` determines whether the data is shuffled prior to
              splitting it into train and validation.

            Available arguments:

            * 'holdout': {'train_size': float}
            * 'holdout-iterative-fit':  {'train_size': float}
            * 'cv': {'folds': int}
            * 'partial-cv': {'folds': int, 'shuffle': bool}
            * BaseCrossValidator or _RepeatedSplits or BaseShuffleSplit object: all arguments
                required by chosen class as specified in scikit-learn documentation.
                If arguments are not provided, scikit-learn defaults are used.
                If no defaults are available, an exception is raised.
                Refer to the 'n_splits' argument as 'folds'.

        tmp_folder : string, optional (None)
            folder to store configuration output and log files, if ``None``
            automatically use ``/tmp/autosklearn_tmp_$pid_$random_number``

        output_folder : string, optional (None)
            folder to store predictions for optional test set, if ``None``
            automatically use ``/tmp/autosklearn_output_$pid_$random_number``

        delete_tmp_folder_after_terminate: string, optional (True)
            remove tmp_folder, when finished. If tmp_folder is None
            tmp_dir will always be deleted

        delete_output_folder_after_terminate: bool, optional (True)
            remove output_folder, when finished. If output_folder is None
            output_dir will always be deleted

        shared_mode : bool, optional (False)
            Run smac in shared-model-node. This only works if arguments
            ``tmp_folder`` and ``output_folder`` are given and both
            ``delete_tmp_folder_after_terminate`` and
            ``delete_output_folder_after_terminate`` are set to False. Cannot
            be used together with ``n_jobs``.

        n_jobs : int, optional, experimental
            The number of jobs to run in parallel for ``fit()``. Cannot be
            used together with ``shared_mode``. ``-1`` means using all
            processors. By default, Auto-sklearn uses a single core for
            fitting the machine learning model and a single core for fitting
            an ensemble. Ensemble building is not affected by ``n_jobs`` but
            can be controlled by the number of models in the ensemble. In
            contrast to most scikit-learn models, ``n_jobs`` given in the
            constructor is not applied to the ``predict()`` method.

        disable_evaluator_output: bool or list, optional (False)
            If True, disable model and prediction output. Cannot be used
            together with ensemble building. ``predict()`` cannot be used when
            setting this True. Can also be used as a list to pass more
            fine-grained information on what to save. Allowed elements in the
            list are:

            * ``'y_optimization'`` : do not save the predictions for the
              optimization/validation set, which would later on be used to build
              an ensemble.
            * ``'model'`` : do not save any model files

        smac_scenario_args : dict, optional (None)
            Additional arguments inserted into the scenario of SMAC. See the
            `SMAC documentation <https://automl.github.io/SMAC3/stable/options.html?highlight=scenario#scenario>`_
            for a list of available arguments.

        get_smac_object_callback : callable
            Callback function to create an object of class
            `smac.optimizer.smbo.SMBO <https://automl.github.io/SMAC3/stable/apidoc/smac.optimizer.smbo.html>`_.
            The function must accept the arguments ``scenario_dict``,
            ``instances``, ``num_params``, ``runhistory``, ``seed`` and ``ta``.
            This is an advanced feature. Use only if you are familiar with
            `SMAC <https://automl.github.io/SMAC3/stable/index.html>`_.

        logging_config : dict, optional (None)
            dictionary object specifying the logger configuration. If None,
            the default logging.yaml file is used, which can be found in
            the directory ``util/logging.yaml`` relative to the installation.

        metadata_directory : str, optional (None)
            path to the metadata directory. If None, the default directory
            (autosklearn.metalearning.files) is used.

        Attributes
        ----------

        cv_results\_ : dict of numpy (masked) ndarrays
            A dict with keys as column headers and values as columns, that can be
            imported into a pandas ``DataFrame``.

            Not all keys returned by scikit-learn are supported yet.

        """
        # Raise error if the given total time budget is less than 60 seconds.
        if time_left_for_this_task < 30:
            raise ValueError("Time left for this task must be at least "
                             "30 seconds. ")
        self.maxFE = maxFE
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.initial_configurations_via_metalearning = initial_configurations_via_metalearning
        self.seed = seed
        self.ml_memory_limit = ml_memory_limit
        self.include_estimators = include_estimators
        self.exclude_estimators = exclude_estimators
        self.include_adpts = include_adpts
        self.exclude_adpts = exclude_adpts
        self.tmp_folder = tmp_folder
        self.output_folder = output_folder
        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate
        self.shared_mode = shared_mode
        self.n_jobs = n_jobs
        self.disable_evaluator_output = disable_evaluator_output
        self.get_smac_object_callback = get_smac_object_callback
        self.smac_scenario_args = smac_scenario_args
        self.logging_config = logging_config
        self.repeat = repeat

        self._automl = None  # type: Optional[List[BaseAutoML]]
        # n_jobs after conversion to a number (b/c default is None)
        self._n_jobs = None

    def build_automl(
            self,
            seed: int,
            shared_mode: bool,
            initial_configurations_via_metalearning: int,
            tmp_folder: str,
            output_folder: str,
            smac_scenario_args: Optional[Dict] = None,
    ):

        # 创建相关文件夹
        backend = create(temporary_directory=tmp_folder,
                         output_directory=output_folder,
                         delete_tmp_folder_after_terminate=self.delete_tmp_folder_after_terminate,
                         delete_output_folder_after_terminate=self.delete_output_folder_after_terminate,
                         shared_mode=shared_mode)

        if smac_scenario_args is None:
            smac_scenario_args = self.smac_scenario_args

        # 获得一个model对象
        automl = self._get_automl_class()(
            backend=backend,
            maxFE = self.maxFE,
            time_left_for_this_task=self.time_left_for_this_task,
            per_run_time_limit=self.per_run_time_limit,
            initial_configurations_via_metalearning=
            initial_configurations_via_metalearning,
            seed=seed,
            ml_memory_limit=self.ml_memory_limit,
            include_estimators=self.include_estimators,
            exclude_estimators=self.exclude_estimators,
            include_adpts=self.include_adpts,
            exclude_adpts=self.exclude_adpts,
            shared_mode=shared_mode,
            get_smac_object_callback=self.get_smac_object_callback,
            disable_evaluator_output=self.disable_evaluator_output,
            smac_scenario_args=smac_scenario_args,
            logging_config=self.logging_config,
            repeat=self.repeat,
        )

        return automl

    def fit(self, **kwargs):
        self._automl = []

        self._n_jobs = 1
        shared_mode = self.shared_mode
        seed = self.seed
        automl = self.build_automl(
            seed=seed,
            shared_mode=shared_mode,
            initial_configurations_via_metalearning=(
                self.initial_configurations_via_metalearning
            ),
            tmp_folder=self.tmp_folder,
            output_folder=self.output_folder,
        )
        self._automl.append(automl)
        self._automl[0].fit(**kwargs)

        return self

    def score(self, X, y):
        return self._automl[0].score(X, y)


    @property
    def trajectory_(self):
        if len(self._automl) > 1:
            raise NotImplementedError()
        return self._automl[0].trajectory_

    @property
    def incument_(self):
        return 'incument:', self._automl[0].inc, 'finc:', self._automl[0].finc


    @property
    def fANOVA_input_(self):
        if len(self._automl) > 1:
            raise NotImplementedError()
        return self._automl[0].fANOVA_input_

    def sprint_statistics(self):
        """Return the following statistics of the training result:

        - dataset name
        - metric used
        - best validation score
        - number of target algorithm runs
        - number of successful target algorithm runs
        - number of crashed target algorithm runs
        - number of target algorithm runs that exceeded the memory limit
        - number of target algorithm runs that exceeded the time limit

        Returns
        -------
        str
        """
        return self._automl[0].sprint_statistics()

    def _get_automl_class(self):
        raise NotImplementedError()

    def get_configuration_space(self, X, y):
        self._automl = self.build_automl()
        return self._automl[0].fit(X, y, only_return_configuration_space=True)


class Cpdp(AutoEstimator):
    """
    This class implements the classification task.

    """

    def fit(self, X, y, xtarget, ytarget, loc,
            metric=None,
            feat_type=None,
            dataset_name=None):
        """Fit *auto-sklearn* to given training set (X, y).

        Fit both optimizes the machine learning models and builds an ensemble
        out of them. To disable ensembling, set ``ensemble_size==0``.

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target classes.

        X_test : array-like or sparse matrix of shape = [n_samples, n_features]
            Test data input samples. Will be used to save test predictions for
            all models. This allows to evaluate the performance of Auto-sklearn
            over time.

        y_test : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Test data target classes. Will be used to calculate the test error
            of all models. This allows to evaluate the performance of
            Auto-sklearn over time.

        metric : callable, optional (default='autosklearn.metrics.accuracy')
            An instance of :class:`autosklearn.metrics.Scorer` as created by
            :meth:`autosklearn.metrics.make_scorer`. These are the `Built-in
            Metrics`_.

        feat_type : list, optional (default=None)
            List of str of `len(X.shape[1])` describing the attribute type.
            Possible types are `Categorical` and `Numerical`. `Categorical`
            attributes will be automatically One-Hot encoded. The values
            used for a categorical attribute must be integers, obtained for
            example by `sklearn.preprocessing.LabelEncoder
            <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_.

        dataset_name : str, optional (default=None)
            Create nicer output. If None, a string will be determined by the
            md5 hash of the dataset.

        Returns
        -------n

        self

        """
        # Before running anything else, first check that the
        # type of data is compatible with auto-sklearn. Legal target
        # types are: binary, multiclass, multilabel-indicator.
        target_type = type_of_target(y)
        if target_type in ['multiclass-multioutput',
                           'continuous',
                           'continuous-multioutput',
                           'unknown',
                           ]:
            raise ValueError("classification with data of type %s is"
                             " not supported" % target_type)

        # remember target type for using in predict_proba later.
        self.target_type = target_type

        super().fit(
            X=X,
            y=y,
            xtarget=xtarget,
            ytarget=ytarget,
            loc=loc,
            metric=metric,
            feat_type=feat_type,
            dataset_name=dataset_name,
        )

        return self

    def _get_automl_class(self):
        return AutoMLCPDP
