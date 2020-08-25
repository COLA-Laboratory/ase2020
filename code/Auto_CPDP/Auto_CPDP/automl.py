# -*- encoding: utf-8 -*-
import io
import json
import os
import unittest.mock
from typing import Optional, List
import warnings

from ConfigSpace.read_and_write import pcs
import numpy as np
import scipy.stats

import joblib
import sklearn.utils
import scipy.sparse
from sklearn.metrics.classification import type_of_target
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import StatusType

from Auto_CPDP.evaluation import ExecuteTaFuncWithQueue
from Auto_CPDP.metrics import Scorer
from Auto_CPDP.data.abstract_data_manager import AbstractDataManager
from Auto_CPDP.data.competition_data_manager import CompetitionDataManager
from Auto_CPDP.data.xy_data_manager import XYDataManager
from Auto_CPDP.metrics import calculate_score
from Auto_CPDP.util import StopWatch, get_logger, setup_logger, \
    pipeline
from Auto_CPDP.smbo import AutoMLSMBO
from Auto_CPDP.util.hash import hash_array_or_matrix
from Auto_CPDP.metrics import f1, roc_auc
from Auto_CPDP.constants import *


class AutoML():

    def __init__(self,
                 backend,
                 time_left_for_this_task,
                 per_run_time_limit,
                 maxFE=None,
                 initial_configurations_via_metalearning=25,
                 seed=1,
                 ml_memory_limit=3072,
                 metadata_directory=None,
                 keep_models=True,
                 debug_mode=False,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_adpts=None,
                 exclude_adpts=None,
                 shared_mode=False,
                 precision=32,
                 disable_evaluator_output=False,
                 get_smac_object_callback=None,
                 smac_scenario_args=None,
                 logging_config=None,
                 repeat=1,
                 ):
        super(AutoML, self).__init__()
        self.maxFE = maxFE
        self._backend = backend
        self._time_for_task = time_left_for_this_task
        self._per_run_time_limit = per_run_time_limit
        self._initial_configurations_via_metalearning = \
            initial_configurations_via_metalearning
        self._seed = seed
        self._ml_memory_limit = ml_memory_limit
        self._data_memory_limit = None
        self._metadata_directory = metadata_directory
        self._keep_models = keep_models
        self._include_estimators = include_estimators
        self._exclude_estimators = exclude_estimators
        self._include_adpts = include_adpts
        self._exclude_adpts = exclude_adpts
        self._shared_mode = shared_mode
        self.precision = precision
        self._disable_evaluator_output = disable_evaluator_output
        self._get_smac_object_callback = get_smac_object_callback
        self._smac_scenario_args = smac_scenario_args
        self.logging_config = logging_config

        self._datamanager = None
        self._dataset_name = None
        self._stopwatch = StopWatch()
        self._logger = None
        self._task = None
        self._metric = None
        self._label_num = None
        self._parser = None
        self.models_ = None
        self.ensemble_ = None
        self._can_predict = False

        self._debug_mode = debug_mode
        self.repeat = repeat

        if not isinstance(self._time_for_task, int):
            raise ValueError("time_left_for_this_task not of type integer, "
                             "but %s" % str(type(self._time_for_task)))
        if not isinstance(self._per_run_time_limit, int):
            raise ValueError("per_run_time_limit not of type integer, but %s" %
                             str(type(self._per_run_time_limit)))


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        xtarget: np.ndarray,
        ytarget: np.ndarray,
        loc,
        task: int,
        metric: Scorer,
        feat_type: Optional[List[bool]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: Optional[bool] = False,
        load_models: bool = True,
    ):
        if self._shared_mode:
            # If this fails, it's likely that this is the first call to get
            # the data manager
            try:
                D = self._backend.load_datamanager()
                dataset_name = D.name
            except IOError:
                pass

        if dataset_name is None:
            dataset_name = hash_array_or_matrix(X) + hash_array_or_matrix(xtarget)

        self._backend.save_start_time(self._seed)
        self._stopwatch = StopWatch()
        self._dataset_name = dataset_name
        self._stopwatch.start_task(self._dataset_name)

        self._loggenr = self._get_logger(dataset_name)

        if metric is None:
            raise ValueError('No metric given.')
        if not isinstance(metric, Scorer):
            raise ValueError('Metric must be instance of '
                             'autosklearn.metrics.Scorer.')

        if feat_type is not None and len(feat_type) != X.shape[1]:
            raise ValueError('Array feat_type does not have same number of '
                             'variables as X has features. %d vs %d.' %
                             (len(feat_type), X.shape[1]))
        if feat_type is not None and not all([isinstance(f, str)
                                              for f in feat_type]):
            raise ValueError('Array feat_type must only contain strings.')
        if feat_type is not None:
            for ft in feat_type:
                if ft.lower() not in ['categorical', 'numerical']:
                    raise ValueError('Only `Categorical` and `Numerical` are '
                                     'valid feature types, you passed `%s`' % ft)

        self._data_memory_limit = None
        # 将原本的categorical数据转化成二进制向量
        loaded_data_manager = XYDataManager(
            X, y,
            xtarget=xtarget,
            ytarget=ytarget,
            loc=loc,
            task=task,
            feat_type=feat_type,
            dataset_name=dataset_name,
        )

        return self._fit(
            datamanager=loaded_data_manager,
            metric=metric,
            load_models=load_models,
            only_return_configuration_space=only_return_configuration_space,
        )

    def fit_on_datamanager(self, datamanager, metric, load_models=True):
        self._stopwatch = StopWatch()
        self._backend.save_start_time(self._seed)

        name = os.path.basename(datamanager.name)
        self._stopwatch.start_task(name)
        self._start_task(self._stopwatch, name)
        self._dataset_name = name

        self._logger = self._get_logger(name)
        self._fit(
            datamanager=datamanager,
            metric=metric,
            load_models=load_models,
        )

    def _get_logger(self, name):
        logger_name = 'AutoML(%d):%s' % (self._seed, name)
        setup_logger(os.path.join(self._backend.temporary_directory,
                                  '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)

    @staticmethod
    def _start_task(watcher, task_name):
        watcher.start_task(task_name)

    @staticmethod
    def _stop_task(watcher, task_name):
        watcher.stop_task(task_name)

    @staticmethod
    def _print_load_time(basename, time_left_for_this_task,
                         time_for_load_data, logger):

        time_left_after_reading = max(
            0, time_left_for_this_task - time_for_load_data)
        logger.info('Remaining time after reading %s %5.2f sec' %
                    (basename, time_left_after_reading))
        return time_for_load_data


    def _fit(
        self,
        datamanager: AbstractDataManager,
        metric: Scorer,
        load_models: bool,
        only_return_configuration_space: bool = False,
    ):

        # Check arguments prior to doing anything!
        if not isinstance(self._disable_evaluator_output, (bool, list)):
            raise ValueError('disable_evaluator_output must be of type bool '
                             'or list.')
        if isinstance(self._disable_evaluator_output, list):
            allowed_elements = ['model', 'y_optimization']
            for element in self._disable_evaluator_output:
                if element not in allowed_elements:
                    raise ValueError("List member '%s' for argument "
                                     "'disable_evaluator_output' must be one "
                                     "of " + str(allowed_elements))


        self._backend._make_internals_directory()
        if self._keep_models:
            try:
                os.makedirs(self._backend.get_model_dir())
            except (OSError, FileExistsError) as e:
                if not self._shared_mode:
                    raise

        self._metric = metric
        self._task = datamanager.info['task']
        self._label_num = datamanager.info['label_num']

        # == Pickle the data manager to speed up loading
        data_manager_path = self._backend.save_datamanager(datamanager)

        time_for_load_data = self._stopwatch.wall_elapsed(self._dataset_name)

        if self._debug_mode:
            self._print_load_time(
                self._dataset_name,
                self._time_for_task,
                time_for_load_data,
                self._logger)

        # == Perform dummy predictions
        num_run = 1

        # = Create a searchspace
        # Do this before One Hot Encoding to make sure that it creates a
        # search space for a dense classifier even if one hot encoding would
        # make it sparse (tradeoff; if one hot encoding would make it sparse,
        #  densifier and truncatedSVD would probably lead to a MemoryError,
        # like this we can't use some of the preprocessing methods in case
        # the data became sparse)
        self.configuration_space, configspace_path = self._create_search_space(
            self._backend.temporary_directory,
            self._backend,
            datamanager,
            include_estimators=self._include_estimators,
            exclude_estimators=self._exclude_estimators,
            include_adpts=self._include_adpts,
            exclude_adpts=self._exclude_adpts)

        if only_return_configuration_space:
            return self.configuration_space


        # kill the datamanager as it will be re-loaded anyways from sub processes
        try:
            del self._datamanager
        except Exception:
            pass

        # => RUN SMAC
        smac_task_name = 'runSMAC'
        self._stopwatch.start_task(smac_task_name)
        time_left_for_smac = max(0,
            self._time_for_task - self._stopwatch.wall_elapsed(
            self._dataset_name))

        if self._logger:
            self._logger.info(
                'Start SMAC with %5.2fsec time left' % time_left_for_smac)
        if time_left_for_smac <= 0:
            self._logger.warning("Not starting SMAC because there is no time "
                                 "left.")
            _proc_smac = None
        else:
            if self._per_run_time_limit is None or \
                    self._per_run_time_limit > time_left_for_smac:
                print('Time limit for a single run is higher than total time '
                      'limit. Capping the limit for a single run to the total '
                      'time given to SMAC (%f)' % time_left_for_smac)
                per_run_time_limit = time_left_for_smac
            else:
                per_run_time_limit = self._per_run_time_limit

            _proc_smac = AutoMLSMBO(
                maxFE=self.maxFE,
                config_space=self.configuration_space,
                dataset_name=self._dataset_name,
                backend=self._backend,
                total_walltime_limit=time_left_for_smac,
                func_eval_time_limit=per_run_time_limit,
                memory_limit=self._ml_memory_limit,
                data_memory_limit=self._data_memory_limit,
                watcher=self._stopwatch,
                start_num_run=num_run,
                num_metalearning_cfgs=self._initial_configurations_via_metalearning,
                config_file=configspace_path,
                seed=self._seed,
                metadata_directory=self._metadata_directory,
                metric=self._metric,
                shared_mode=self._shared_mode,
                include_estimators=self._include_estimators,
                exclude_estimators=self._exclude_estimators,
                include_adpts=self._include_adpts,
                exclude_adpts=self._exclude_adpts,
                disable_file_output=self._disable_evaluator_output,
                get_smac_object_callback=self._get_smac_object_callback,
                smac_scenario_args=self._smac_scenario_args,
                repeat=self.repeat
            )

            self.runhistory_, self.trajectory_, self.inc, self.finc= \
                _proc_smac.run_smbo()

            trajectory_filename = os.path.join(
                self._backend.get_smac_output_directory_for_run(self._seed),
                'trajectory.json')
            saveable_trajectory = \
                [list(entry[:2]) + [entry[2].get_dictionary()] + list(entry[3:])
                 for entry in self.trajectory_]
            with open(trajectory_filename, 'w') as fh:
                json.dump(saveable_trajectory, fh)




        return self





    def _create_search_space(self, tmp_dir, backend, datamanager,
                             include_estimators=None,
                             exclude_estimators=None,
                             include_adpts=None,
                             exclude_adpts=None):
        task_name = 'CreateConfigSpace'

        self._stopwatch.start_task(task_name)
        configspace_path = os.path.join(tmp_dir, 'space.pcs')
        # 获取configuration space
        configuration_space = pipeline.get_configuration_space(
            datamanager.info,
            include_estimators=include_estimators,
            exclude_estimators=exclude_estimators,
            include_adpts=include_adpts,
            exclude_adpts=exclude_adpts)
        configuration_space = self.configuration_space_created_hook(
            datamanager, configuration_space)
        sp_string = pcs.write(configuration_space)
        backend.write_txt_file(configspace_path, sp_string,
                               'Configuration space')
        self._stopwatch.stop_task(task_name)

        return configuration_space, configspace_path

    def configuration_space_created_hook(self, datamanager, configuration_space):
        return configuration_space


class BaseAutoML(AutoML):
    """Base class for AutoML objects to hold abstract functions for both
    regression and classification."""

    def __init__(self, *args, **kwargs):
        self._n_outputs = 1
        super().__init__(*args, **kwargs)

    def _perform_input_checks(self, X, y):
        X = self._check_X(X)
        if y is not None:
            y = self._check_y(y)
        return X, y

    def _check_X(self, X):
        X = sklearn.utils.check_array(X, accept_sparse="csr",
                                      force_all_finite=False)
        if scipy.sparse.issparse(X):
            X.sort_indices()
        return X

    def _check_y(self, y):
        y = sklearn.utils.check_array(y, ensure_2d=False)

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Will change shape via np.ravel().",
                          sklearn.utils.DataConversionWarning, stacklevel=2)
            y = np.ravel(y)

        return y


class AutoMLCPDP(BaseAutoML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._task_mapping = {'multilabel-indicator': MULTILABEL_CLASSIFICATION,
                              'multiclass': MULTICLASS_CLASSIFICATION,
                              'binary': BINARY_CLASSIFICATION}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        xtarget: np.ndarray,
        ytarget: np.ndarray,
        loc,
        metric: Optional[Scorer] = None,
        feat_type: Optional[List[bool]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
    ):
        X, y = self._perform_input_checks(X, y)
        xtarget, ytarget = self._perform_input_checks(xtarget, ytarget)

        y_task = type_of_target(y)
        task = self._task_mapping.get(y_task)
        if task is None:
            raise ValueError('Cannot work on data of type %s' % y_task)

        if metric is None:
            if task == MULTILABEL_CLASSIFICATION:
                metric = f1
            else:
                metric = roc_auc

        y, self._classes, self._n_classes = self._process_target_classes(y)

        return super().fit(
            X, y,
            xtarget=xtarget,
            ytarget=ytarget,
            loc=loc,
            task=task,
            metric=metric,
            feat_type=feat_type,
            dataset_name=dataset_name,
            only_return_configuration_space=only_return_configuration_space,
            load_models=load_models,
        )



    def _process_target_classes(self, y):
        y = super()._check_y(y)
        self._n_outputs = 1 if len(y.shape) == 1 else y.shape[1]

        y = np.copy(y)

        _classes = []
        _n_classes = []

        if self._n_outputs == 1:
            classes_k, y = np.unique(y, return_inverse=True)
            _classes.append(classes_k)
            _n_classes.append(classes_k.shape[0])
        else:
            for k in range(self._n_outputs):
                classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
                _classes.append(classes_k)
                _n_classes.append(classes_k.shape[0])

        _n_classes = np.array(_n_classes, dtype=np.int)

        return y, _classes, _n_classes

