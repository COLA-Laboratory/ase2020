import json

from smac.facade.smac_facade import SMAC
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.optimizer import pSMAC

from Auto_CPDP.constants import MULTILABEL_CLASSIFICATION, \
    BINARY_CLASSIFICATION, TASK_TYPES_TO_STRING, CLASSIFICATION_TASKS, \
    REGRESSION_TASKS, MULTICLASS_CLASSIFICATION, REGRESSION
from Auto_CPDP.data.abstract_data_manager import AbstractDataManager
from Auto_CPDP.data.competition_data_manager import CompetitionDataManager
from Auto_CPDP.evaluation import ExecuteTaFuncWithQueue, WORST_POSSIBLE_RESULT
from Auto_CPDP.util import get_logger


# dataset helpers
def load_data(dataset_info, backend, max_mem=None):
    try:
        D = backend.load_datamanager()
    except IOError:
        D = None

    # Datamanager probably doesn't exist
    if D is None:
        if max_mem is None:
            D = CompetitionDataManager(dataset_info)
        else:
            D = CompetitionDataManager(dataset_info, max_memory_in_mb=max_mem)
    return D


def get_smac_object(
        scenario_dict,
        seed,
        ta,
        backend,
        runhistory,
):
    scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob(
        smac_run_id=seed if not scenario_dict['shared-model'] else '*',
    )
    scenario = Scenario(scenario_dict)

    initial_configurations = None
    rh2EPM = RunHistory2EPM4Cost(
        num_params=len(scenario.cs.get_hyperparameters()),
        scenario=scenario,
        success_states=[
            StatusType.SUCCESS,
            StatusType.MEMOUT,
            StatusType.TIMEOUT,
            # As long as we don't have a model for crashes yet!
            StatusType.CRASHED,
        ],
        impute_censored_data=False,
        impute_state=None,
    )
    return SMAC(
        scenario=scenario,
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        initial_configurations=initial_configurations,
        runhistory=runhistory,
        run_id=seed,
    )


def _print_debug_info_of_init_configuration(initial_configurations, basename,
                                            time_for_task, logger, watcher):
    logger.debug('Initial Configurations: (%d)' % len(initial_configurations))
    for initial_configuration in initial_configurations:
        logger.debug(initial_configuration)
    logger.debug('Looking for initial configurations took %5.2fsec',
                 watcher.wall_elapsed('InitialConfigurations'))
    logger.info(
        'Time left for %s after finding initial configurations: %5.2fsec',
        basename, time_for_task - watcher.wall_elapsed(basename))


class AutoMLSMBO(object):

    def __init__(self, maxFE, config_space, dataset_name,
                 backend,
                 total_walltime_limit,
                 func_eval_time_limit,
                 memory_limit,
                 metric,
                 watcher, start_num_run=1,
                 data_memory_limit=None,
                 num_metalearning_cfgs=25,
                 config_file=None,
                 seed=1,
                 metadata_directory=None,
                 shared_mode=False,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_adpts=None,
                 exclude_adpts=None,
                 disable_file_output=False,
                 smac_scenario_args=None,
                 get_smac_object_callback=None,
                 repeat=1):
        super(AutoMLSMBO, self).__init__()
        # data related
        self.maxFE = maxFE
        self.dataset_name = dataset_name
        self.datamanager = None
        self.metric = metric
        self.task = None
        self.backend = backend
        self.repeat = repeat

        # the configuration space
        self.config_space = config_space

        # and a bunch of useful limits
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit = int(func_eval_time_limit)
        self.memory_limit = memory_limit
        self.data_memory_limit = data_memory_limit
        self.watcher = watcher
        self.num_metalearning_cfgs = num_metalearning_cfgs
        self.config_file = config_file
        self.seed = seed
        self.metadata_directory = metadata_directory
        self.start_num_run = start_num_run
        self.shared_mode = shared_mode
        self.include_estimators = include_estimators
        self.exclude_estimators = exclude_estimators
        self.include_adpts = include_adpts
        self.exclude_adpts = exclude_adpts
        self.disable_file_output = disable_file_output
        self.smac_scenario_args = smac_scenario_args
        self.get_smac_object_callback = get_smac_object_callback

        logger_name = '%s(%d):%s' % (self.__class__.__name__, self.seed,
                                     ":" + dataset_name if dataset_name is
                                                           not None else "")
        self.logger = get_logger(logger_name)

    def _send_warnings_to_log(self, message, category, filename, lineno,
                              file=None, line=None):
        self.logger.debug('%s:%s: %s:%s', filename, lineno, category.__name__,
                          message)

    # 加载数据
    def reset_data_manager(self, max_mem=None):
        if max_mem is None:
            max_mem = self.data_memory_limit
        if self.datamanager is not None:
            del self.datamanager
        if isinstance(self.dataset_name, AbstractDataManager):
            self.datamanager = self.dataset_name
        else:
            self.datamanager = load_data(self.dataset_name,
                                         self.backend,
                                         max_mem=max_mem)

        self.task = self.datamanager.info['task']

    def run_smbo(self):
        self.watcher.start_task('SMBO')

        # == first things first: load the datamanager
        self.reset_data_manager()

        # == Initialize non-SMBO stuff
        # first create a scenario
        seed = self.seed
        self.config_space.seed(seed)
        num_params = len(self.config_space.get_hyperparameters())
        # allocate a run history
        num_run = self.start_num_run

        instances = [[json.dumps({'task_id': self.dataset_name})]]

        # TODO rebuild target algorithm to be it's own target algorithm
        # evaluator, which takes into account that a run can be killed prior
        # to the model being fully fitted; thus putting intermediate results
        # into a queue and querying them once the time is over
        exclude = dict()
        include = dict()
        if self.include_adpts is not None and \
                self.exclude_adpts is not None:
            raise ValueError('Cannot specify include_adpts and '
                             'exclude_adpts.')
        elif self.include_adpts is not None:
            include['domain_adaptation'] = self.include_adpts
        elif self.exclude_adpts is not None:
            exclude['domain_adaptation'] = self.exclude_adpts

        if self.include_estimators is not None and \
                self.exclude_estimators is not None:
            raise ValueError('Cannot specify include_estimators and '
                             'exclude_estimators.')
        elif self.include_estimators is not None:
            if self.task in CLASSIFICATION_TASKS:
                include['classifier'] = self.include_estimators
            elif self.task in REGRESSION_TASKS:
                include['regressor'] = self.include_estimators
            else:
                raise ValueError(self.task)
        elif self.exclude_estimators is not None:
            if self.task in CLASSIFICATION_TASKS:
                exclude['classifier'] = self.exclude_estimators
            elif self.task in REGRESSION_TASKS:
                exclude['regressor'] = self.exclude_estimators
            else:
                raise ValueError(self.task)

        ta = ExecuteTaFuncWithQueue(backend=self.backend,
                                    autosklearn_seed=seed,
                                    initial_num_run=num_run,
                                    logger=self.logger,
                                    include=include,
                                    exclude=exclude,
                                    metric=self.metric,
                                    memory_limit=self.memory_limit,
                                    disable_file_output=self.disable_file_output,
                                    repeat=self.repeat,
                                    maxFE=self.maxFE)

        if self.maxFE is None:
            startup_time = self.watcher.wall_elapsed(self.dataset_name)
            total_walltime_limit = self.total_walltime_limit - startup_time - 5
            scenario_dict = {
                'abort_on_first_run_crash': False,
                'cs': self.config_space,
                'cutoff_time': self.func_eval_time_limit,
                'deterministic': 'true',
                'instances': instances,
                'memory_limit': self.memory_limit,
                'output-dir':
                    self.backend.get_smac_output_directory(),
                'run_obj': 'quality',
                'shared-model': self.shared_mode,
                'wallclock_limit': total_walltime_limit,
                'cost_for_crash': WORST_POSSIBLE_RESULT,
            }
        else:
            scenario_dict = {
                'abort_on_first_run_crash': False,
                'cs': self.config_space,
                'runcount-limit': self.maxFE,
                'deterministic': 'true',
                'instances': instances,
                'memory_limit': self.memory_limit,
                'output-dir':
                    self.backend.get_smac_output_directory(),
                'run_obj': 'quality',
                'shared-model': self.shared_mode,
                'cost_for_crash': WORST_POSSIBLE_RESULT,
            }
        if self.smac_scenario_args is not None:
            for arg in [
                'abort_on_first_run_crash',
                'cs',
                'deterministic',
                'instances',
                'output-dir',
                'run_obj',
                'shared-model',
                'cost_for_crash',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning('Cannot override scenario argument %s, '
                                        'will ignore this.', arg)
                    del self.smac_scenario_args[arg]
            for arg in [
                'cutoff_time',
                'memory_limit',
                'wallclock_limit',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning(
                        'Overriding scenario argument %s: %s with value %s',
                        arg,
                        scenario_dict[arg],
                        self.smac_scenario_args[arg]
                    )
            scenario_dict.update(self.smac_scenario_args)

        runhistory = RunHistory(aggregate_func=average_cost)
        smac_args = {
            'scenario_dict': scenario_dict,
            'seed': seed,
            'ta': ta,
            'backend': self.backend,
            'runhistory': runhistory,
        }
        if self.get_smac_object_callback is not None:
            smac = self.get_smac_object_callback(**smac_args)
        else:
            smac = get_smac_object(**smac_args)

        inc = smac.optimize()
        self.inc = inc
        # Patch SMAC to read in data from parallel runs after the last
        # function evaluation
        if self.shared_mode:
            pSMAC.read(
                run_history=smac.solver.runhistory,
                output_dirs=smac.solver.scenario.input_psmac_dirs,
                configuration_space=smac.solver.config_space,
                logger=smac.solver.logger,
            )

        self.runhistory = smac.solver.runhistory
        self.trajectory = smac.solver.intensifier.traj_logger.trajectory
        res = self.metric._optimum - self.runhistory.get_cost(self.inc)

        return self.runhistory, self.trajectory, self.inc, res
