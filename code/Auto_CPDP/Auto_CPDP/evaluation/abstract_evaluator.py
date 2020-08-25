import time


import numpy as np
from smac.tae.execute_ta_run import StatusType

from Auto_CPDP.CPDP.CPDP import CPDP_pipeline
from Auto_CPDP.constants import (
    REGRESSION_TASKS,
    MULTILABEL_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
)
from Auto_CPDP.metrics import calculate_score, CLASSIFICATION_METRICS
from Auto_CPDP.util.logging_ import get_logger

from ConfigSpace import Configuration




class AbstractEvaluator(object):
    def __init__(self, backend, queue, metric,
                 configuration=None,
                 all_scoring_functions=False,
                 seed=1,
                 output_y_hat_optimization=True,
                 num_run=None,
                 include=None,
                 exclude=None,
                 disable_file_output=False,
                 init_params=None,
                 repeat=1):

        self.starttime = time.time()

        self.configuration = configuration
        self.backend = backend
        self.queue = queue

        self.datamanager = self.backend.load_datamanager()
        self.include = include
        self.exclude = exclude

        self.X_train = self.datamanager.data.get('X_train')
        self.y_train = self.datamanager.data.get('Y_train')
        self.xtarget = self.datamanager.data.get('xtarget')
        self.ytarget = self.datamanager.data.get('ytarget')
        self.loc = self.datamanager.data.get('loc')

        self.metric = metric
        self.task_type = self.datamanager.info['task']
        self.seed = seed

        self.output_y_hat_optimization = output_y_hat_optimization
        self.all_scoring_functions = all_scoring_functions
        self.disable_file_output = disable_file_output
        self._init_params = init_params
        self.repeat = repeat



        # pipeline 声明利用
        dataset_properties = {
            'task': self.task_type,
            'sparse': self.datamanager.info['is_sparse'] == 1,
            'multilabel': self.task_type == MULTILABEL_CLASSIFICATION,
            'multiclass': self.task_type == MULTICLASS_CLASSIFICATION,
        }

        self.model = CPDP_pipeline(config=self.configuration,
                                 dataset_properties=dataset_properties,
                                 random_state=self.seed,
                                 include=self.include,
                                 exclude=self.exclude,
                                 init_params=self._init_params, repeat=self.repeat)

        categorical_mask = []
        for feat in self.datamanager.feat_type:
            if feat.lower() == 'numerical':
                categorical_mask.append(False)
            elif feat.lower() == 'categorical':
                categorical_mask.append(True)
            else:
                raise ValueError(feat)
        if np.sum(categorical_mask) > 0:
            self._init_params = {
                'categorical_encoding:one_hot_encoding:categorical_features':
                    categorical_mask
            }
        else:
            self._init_params = {}
        if init_params is not None:
            self._init_params.update(init_params)

        if num_run is None:
            num_run = 0
        self.num_run = num_run


        logger_name = '%s(%d):%s' % (self.__class__.__name__.split('.')[-1],
                                     self.seed, self.datamanager.name)
        self.logger = get_logger(logger_name)




    def finish_up(self, loss,
                  additional_run_info, final_call):
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        self.duration = time.time() - self.starttime

        if isinstance(loss, dict):
            loss_ = loss
            loss = loss_[self.metric.name]
        else:
            loss_ = {}

        additional_run_info = (
            {} if additional_run_info is None else additional_run_info
        )
        for metric_name, value in loss_.items():
            additional_run_info[metric_name] = value
        additional_run_info['duration'] = self.duration
        additional_run_info['num_run'] = self.num_run


        rval_dict = {'loss': loss,
                     'additional_run_info': additional_run_info,
                     'status': StatusType.SUCCESS}
        if final_call:
            rval_dict['final_queue_element'] = True

        self.queue.put(rval_dict)


    def evaluate(self):
        loss = self.model.run(self.X_train, self.y_train, self.xtarget, self.ytarget, self.loc)
        self.finish_up(loss, self.model.get_additional_run_info(), final_call=True)




def eval_f(
        queue,
        config,
        backend,
        metric,
        seed,
        num_run,
        instance,
        all_scoring_functions,
        output_y_hat_optimization,
        include,
        exclude,
        disable_file_output,
        init_params=None,
        repeat=1):
    evaluator = AbstractEvaluator(backend=backend,
                                  queue=queue,
                                  metric=metric,
                                  configuration=config,
                                  seed=seed,
                                  num_run=num_run,
                                  all_scoring_functions=all_scoring_functions,
                                  output_y_hat_optimization=output_y_hat_optimization,
                                  include=include,
                                  exclude=exclude,
                                  disable_file_output=disable_file_output,
                                  init_params=init_params,
                                  repeat=repeat)

    evaluator.evaluate()
