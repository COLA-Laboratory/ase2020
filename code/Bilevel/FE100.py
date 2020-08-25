import errno, os, psutil
import functools
import logging
import signal
from iteration_utilities import deepflatten
import traceback
from Algorithms.Framework import cpdp
from numpy import *
from Utils.helper import MfindCommonMetric
import warnings
from Utils.File import *
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from LS import *
from func_timeout import FunctionTimedOut
from sklearn.model_selection import train_test_split

""" kill the zombie child process """

def wait_child(signum, frame):
    logging.info('receive SIGCHLD')
    try:
        while True:
            # -1 表示任意子进程
            # os.WNOHANG 表示如果没有可用的需要 wait 退出状态的子进程，立即返回不阻塞
            cpid, status = os.waitpid(-1, os.WNOHANG)
            if cpid == 0:
                logging.info('no child process was immediately available')
                break
            exitcode = status >> 8
            logging.info('child process %s exit with exitcode %s', cpid, exitcode)
    except OSError as e:
        if e.errno == errno.ECHILD:
            logging.warning('current process has no existing unwaited-for child processes.')
        else:
            raise
    logging.info('handle SIGCHLD end')


signal.signal(signal.SIGCHLD, wait_child)


# Time budge
# time_per_model= 3

# lower-level part
class Llevel(object):
    """
        up    : given upper-level variables, format as {'clf':x, 'adpt':y}
        params: the responding lower-level variables when given upper-level variables
        ldir  : the path where history will be saved
    """

    def __init__(self, up, params, xsource, ysource, xtarget, ytarget, loc,
                 fe, ldir):
        self.xsource = xsource
        self.ysource = ysource
        self.xtarget = xtarget
        self.ytarget = ytarget
        self.train = None
        self.Ltrain = None
        self.loc = loc
        self.gclf = up['clf']
        self.adpt = up['adpt']

        self.fe = fe

        self.paramName = dict()  # the name of parameters
        self.paramType = dict()  # the type of parameters (integer, real, categery)
        self.paramRVal = dict()  # the range of parameters (origin)
        self.paramRange = dict()  # the range of parameters (transfered)
        i = 0
        for k, v in params.items():
            self.paramName[i] = k
            self.paramType[i] = v[0]
            self.paramRVal[i] = v[1]
            if v[0] == 'i' or v[0] == 'r':
                self.paramRange[i] = v[1]
            elif v[0] == 'c':
                self.paramRange[i] = [0, len(v[1]) - 1]
            i += 1

        self.dir = ldir
        self.trails = Trials()

        if self.adpt in ['NNfilter', 'MCWs']:
            self.train, self.xtarget, self.Ltrain, self.ytarget = train_test_split(self.xtarget, self.ytarget,
                                                                                   test_size=0.9)

        return

    # the objective function (run the algorithm)
    def f(self, params):
        self.p = cpdp(clf=self.gclf, adpt=self.adpt, repeat=10)
        self.p.set_params(**params)

        res = self.p.run(self.xsource, self.ysource, self.xtarget, self.ytarget, self.loc, train=self.train,
                         Ltrain=self.Ltrain)

        return {'loss': -res, 'status': STATUS_OK, 'result': res}

    # the process of lower-level optimization (TPE)
    # @func_set_timeout(time_per_model)
    def run(self):
        paramSpace = dict()

        for i in range(len(self.paramName)):
            if self.paramType[i] == 'i':
                low = min(self.paramRange[i][0], self.paramRange[i][1])
                high = max(self.paramRange[i][0], self.paramRange[i][1])
                if len(self.paramRange[i]) == 3:
                    paramSpace[self.paramName[i]] = hp.choice(self.paramName[i],
                                                              range(low, high,
                                                                    self.paramRange[i][2]))
                else:
                    paramSpace[self.paramName[i]] = hp.choice(self.paramName[i],
                                                              range(low, high))
            if self.paramType[i] == 'c':
                paramSpace[self.paramName[i]] = hp.choice(self.paramName[i], self.paramRVal[i])
            if self.paramType[i] == 'r':
                low = min(self.paramRange[i][0], self.paramRange[i][1])
                high = max(self.paramRange[i][0], self.paramRange[i][1])
                paramSpace[self.paramName[i]] = hp.uniform(self.paramName[i], low,
                                                           high)

        if self.gclf == 'SVM':
            tmpparamSpace = {
                'SVCkernel': hp.choice('SVCkernel', [
                    {'kernel': 'linear', 'max_iter': -1},
                    {'kernel': 'poly', 'degree': hp.choice('degree', range(1, 5)),
                     'polycoef0': hp.uniform('polycoef0', 0, 10),
                     'polygamma': hp.uniform('polygamma', 1e-2, 100),
                     'max_iter': 10},
                    {'kernel': 'sigmoid', 'sigcoef0': hp.uniform('sigcoef0', 0, 10),
                     'siggamma': hp.uniform('siggamma', 1e-2, 100),
                     'max_iter': 10},
                    {'kernel': 'rbf', 'rbfgamma': hp.uniform('rbfgamma', 1e-2, 100),
                     'max_iter': 10}
                ]),
                'svmC': hp.uniform('C', 0.001, 10),
            }
            paramSpace = dict(paramSpace, **tmpparamSpace)

        if self.gclf == 'NB':
            tmpparamSpace = {
                'NBparam': hp.choice('NBparam', [
                    {'NBType': 'gaussian'},
                    {'NBType': 'multinomial', 'malpha': hp.uniform('malpha', 0, 10)},
                    {'NBType': 'complement', 'calpha': hp.uniform('calpha', 0, 10),
                     'norm': hp.choice('norm', [True, False])}])

            }
            paramSpace = dict(paramSpace, **tmpparamSpace)

        if self.adpt == 'TD':
            adptparamSpace = {
                'TDparam': hp.choice('TDparam', [
                    {'TD_strategy': 'NN', 'TD_num': hp.choice('TD_num', range(1, len(self.loc)))},
                    {'TD_strategy': 'EM'}
                ])
            }
            paramSpace = dict(paramSpace, **adptparamSpace)

        best = fmin(self.f, space=paramSpace, algo=tpe.suggest, max_evals=self.fe, trials=self.trails,
                    show_progressbar=False)

        # save the running history
        his = dict()
        his['name'] = list(self.trails.trials[0]['misc']['vals'].keys())
        i = 0
        for item in self.trails.trials:
            if item['state'] == 2:
                results = list(deepflatten(item['misc']['vals'].values()))
                results.append(item['result']['result'])
                his[i] = results
                i += 1
        if i > 0:
            inc_value = self.trails.best_trial['result']['result']
            best = self.trails.best_trial['misc']['vals']
        else:
            try:
                inc_value = self.trails.best_trial['result']['result']
                best = self.trails.best_trial['misc']['vals']
            except:
                inc_value = 0
                best = []

        with open(self.dir + '/' + str(time.time()) + 'FE100.txt', 'w') as f:
            for k, v in his.items():
                print(v, file=f)

        return -inc_value, best


# upper-level part
class Ulevel(object):
    """
        params:  the whole variables that contain two-level variables
        method:  the name of method that is used to perform upper-level optimization
    """

    def __init__(self, parameters, xsource, ysource, xtarget, ytarget, loc, UFE=6, LFE=1000,
                 method='vns', fname=None
                 ):
        self.xsource = xsource
        self.ysource = ysource
        self.xtarget = xtarget
        self.ytarget = ytarget
        self.loc = loc

        self.params = parameters
        params = parameters['up']

        self.paramName = dict()  # the name of parameters
        self.paramType = dict()  # the type of parameters (integer, real, categery, constant)
        self.paramRVal = dict()  # the range of parameters (origin)
        self.paramRange = dict()  # the range of parameters (transfered)
        i = 0
        for k, v in params.items():
            self.paramName[i] = k
            self.paramType[i] = v[0]
            self.paramRVal[i] = v[1]
            if v[0] == 'i' or v[0] == 'r':
                self.paramRange[i] = v[1]
            elif v[0] == 'c':
                self.paramRange[i] = [0, len(v[1]) - 1]
            i += 1

        self.FE = UFE
        self.LFE = LFE
        self.method = method
        self.fname = fname
        if self.fname is None:
            self.ldir = os.getcwd() + '/BL-history/lower/' + str(time.time())
        else:
            self.ldir = os.getcwd() + '/BL-history/lower/' + self.fname
        if not os.path.exists(self.ldir):
            os.makedirs(self.ldir)

        udir = os.getcwd() + '/BL-history/upper/'
        if not os.path.exists(udir):
            os.makedirs(udir)
        self.ufname = udir + '/' + self.fname + 'FE100.txt'

        self.startTime = time.time()
        self.best_config = {}
        self.best_val = 0
        self.best_comb = {}

    def f(self, x):
        para = dict()
        for i in range(len(self.paramName)):
            if self.paramType[i] == 'r':
                para[self.paramName[i]] = x[i]
            elif self.paramType[i] == 'i':
                para[self.paramName[i]] = int(round(x[i]))
            elif self.paramType[i] == 'c':
                para[self.paramName[i]] = self.paramRVal[i][int(round(x[i]))]
        if para['adpt'] in ['VCB', 'MCWs'] and para['clf'] in ['KNN', 'NCC', 'RNC', 'MLP', 'PAC']:
            return 0, {}

        params = dict()
        for i in range(len(self.paramName)):
            if list(para.values())[i] in ['SVM', 'TD', 'NB']:
                continue
            params = dict(params, **(self.params['lp'][list(para.values())[i]]))
        # lower level optimization
        ex = Llevel(para, params, xsource=self.xsource, ysource=self.ysource, xtarget=self.xtarget,
                    ytarget=self.ytarget,
                    loc=self.loc, fe=self.LFE, ldir=self.ldir)
        try:
            res, best = ex.run()
        except Exception as e:
            # print('error', str(e), repr(e))
            # traceback.print_exc()
            # print(traceback.format_exc())
            #
            # print(ex.trails.best_trial)
            res = - ex.trails.best_trial['result']['result']
            best = ex.trails.best_trial['misc']['vals']
            for k, v in best.items():
                best[k] = v[0]
                index = list(ex.paramName.values()).index(k)
                if ex.paramType[index] == 'i':
                    best[k] = list(range(ex.paramRange[index][0], ex.paramRange[index][1]))[v[0]]
                if ex.paramType[index] == 'c':
                    best[k] = ex.paramRVal[index][int(v[0])]

        if res < self.best_val:
            self.best_config = best
            self.best_val = res
            self.best_comb = x

        return res, best

    def run(self):
        print('############################### upper level #######################################')
        if self.method == 'paratabu':
            exc = paraTabu(f=self.f, range=self.paramRange, dir=self.ufname, max=self.FE, stime=self.startTime)
            try:
                fres = exc.run()
            except FunctionTimedOut:
                try:
                    res = np.asarray(list(exc.tabuList.values()))
                    fres = res[int(np.argmin(res[:, -1][:, 0]))]
                except:
                    fres = [], [0, {}]

                with open(self.ufname, 'a+') as f:
                    for k, v in exc.tabuList.items():
                        print(v, file=f)
                    print('time:', time.time() - self.startTime, file=f)

                # kill the processes
                for p in exc.processList:
                    os.kill(p, signal.SIGKILL)


            f = fres[1]
            x = fres[0]

            if len(x) != 0:
                para = dict()
                for i in range(len(self.paramName)):
                    if self.paramType[i] == 'r':
                        para[self.paramName[i]] = x[i]
                    elif self.paramType[i] == 'i':
                        para[self.paramName[i]] = int(round(x[i]))
                    elif self.paramType[i] == 'c':
                        para[self.paramName[i]] = self.paramRVal[i][int(round(x[i]))]
                f[1] = dict(para, **f[1])
            print(f[0])
            return f


warnings.filterwarnings('ignore')


# the function to perform a bi-level optimization
def bl(Xsource, Lsource, Xtarget, Ltarget, loc, fname, method, repeat=10):
    #   parameters template: {name: (type, range)}
    up = \
        {
            'adpt': ('c',
                     ['NNfilter',
                      'GIS',
                      'PCAmining', 'TCAplus', 'UM',
                      'CLIFE', 'FeSCH',
                      'FSS_bagging',
                      'MCWs',
                      'TD',
                      'VCB', 'HISNN',
                      'CDE_SMOTE'
                      ]),
            'clf': ('c', ['RF', 'KNN', 'SVM', 'LR', 'DT', 'NB',
                          'Ridge', 'PAC', 'Perceptron', 'MLP', 'RNC', 'NCC', 'EXtree', 'adaBoost', 'bagging', 'EXs'
                          ])
        }

    lp = \
        {
            'NNfilter': {'NNn_neighbors': ('i', [1, 100]),
                         'NNmetric': ('c', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])},
            'GIS': {'mProb': ('r', [0.02, 0.1]), 'chrmsize': ('r', [0.02, 0.1]), 'popsize': ('i', [2, 31, 2]), \
                    'numparts': ('i', [2, 7]), 'numgens': ('i', [5, 21]), 'mCount': ('i', [3, 11])},
            'CDE_SMOTE': {'CDE_k': ('i', [1, 100]),
                          'CDE_metric': ('c', ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis'])},
            'CLIFE': {'Clife_n': ('i', [1, 100]), 'Clife_alpha': ('r', [0.05, 0.2]), 'Clife_beta': ('r', [0.2, 0.4]), \
                      'percentage': ('r', [0.6, 0.9])},
            'FeSCH': {'Fesch_nt': ('i', [1, Xsource.shape[1]]), 'Fesch_strategy': ('c', ['SFD', 'LDF', 'FCR'])},
            'FSS_bagging': {'FSS_topn': ('i', [1, len(loc)]), 'FSS_ratio': ('r', [0.1, 0.9]),
                            'FSS_score_thre': ('r', [0.3, 0.7])},
            'HISNN': {'HISNN_minham': ('i', [1, Xsource.shape[1]])},
            'MCWs': {'MCW_k': ('i', [2, len(loc)]), 'MCW_sigmma': ('r', [0.01, 10]), 'MCW_lamb': ('r', [1e-6, 1e2])},
            # 'TD': {'TD_strategy': ('c', ['NN', 'EM']), 'TD_num': ('i', [1, len(loc)]),
            #        'case1': {'cond': ('TD_strategy', 'NN'), 'res': ['TD_num']}},
            'VCB': {'VCB_M': ('i', [2, 30]), 'VCB_lamb': ('r', [0.5, 1.5])},
            'UM': {'pvalue': ('r', [0.01, 0.1])},
            'TCAplus': {'kernel_type': ('c', ['primal', 'linear', 'rbf', 'sam']),
                        'dim': ('i', [5, max(Xsource.shape[1], Xtarget.shape[1])]), \
                        'lamb': ('r', [1e-6, 1e2]), 'gamma': ('r', [1e-5, 1e2])},
            'PCAmining': {'pcaDim': ('i', [5, max(Xsource.shape[1], Xtarget.shape[1])])},

            'RF': {'RFn_estimators': ('i', [10, 200]), 'RFcriterion': ('c', ['gini', 'entropy']),
                   'RFmax_features': ('r', [0.2, 1.0]),
                   'RFmin_samples_split': ('i', [2, 40]), 'RFmin_samples_leaf': ('i', [1, 20])},
            'KNN': {'KNNneighbors': ('i', [1, 10]), 'KNNp': ('i', [1, 5])},
            # 'NB': {'NBType': ('c', ['gaussian', 'multinomial', 'complement']), 'malpha': ('r', [0, 10]), \
            #        'calpha': ('r', [1, 10]), 'norm': ('c', [True, False]), \
            #        'case1': {'cond': ('NBType', 'multinomial'), 'res': ['malpha']},
            #        'case2': {'cond': ('NBType', 'complement'), 'res': ['calpha']}},
            'DT': {'DTcriterion': ('c', ['gini', 'entropy']), 'DTmax_features': ('r', [0.2, 1.0]),
                   'DTmin_samples_split': ('i', [2, 40]),
                   'DTsplitter': ('c', ['best', 'random']), 'DTmin_samples_leaf': ('i', [1, 20])},
            'LR': {'penalty': ('c', ['l1', 'l2']), 'lrC': ('r', [0.001, 10]), 'maxiter': ('i', [50, 200]), \
                   'fit_intercept': ('c', [True, False])},
            # 'SVM': {'kernel': ('c', ['linear', 'rbf', 'poly', 'sigmoid']), 'degree': ('i', [1, 5]),
            #         'coef0': ('r', [0, 10]),
            #         'gamma': ('r', [1e-2, 100]), 'svmC': ('r', [0.001, 10]),
            #         'case1': {'cond': ('kernel', 'rbf'), 'res': ['gamma']},
            #         'case2': {'cond': ('kernel', 'poly'), 'res': ['gamma', 'coef0', 'degree']},
            #         'case3': {'cond': ('kernel', 'sigmoid'), 'res': ['gamma', 'coef0']}}
            'Ridge': {'Ridge_alpha': ('r', [0.001, 100]), 'Ridge_fit_intercept': ('c', [True, False]),
                      'Ridge_tol': ('r', [1e-5, 0.1])},
            'PAC': {'PAC_c': ('r', [1e-3, 100]), 'PAC_fit_intercept': ('c', [True, False]),
                    'PAC_tol': ('r', [1e-5, 0.1]), 'PAC_loss': ('c', ['hinge', 'squared_hinge'])},
            'Perceptron': {'Per_penalty': ('c', ['l1', 'l2']), 'Per_alpha': ('r', [1e-5, 0.1]),
                           'Per_fit_intercept': ('c', [True, False]), 'Per_tol': ('r', [1e-5, 0.1])},
            'MLP': {'MLP_hidden_layer_sizes': ('i', [50, 200]),
                    'MLP_activation': ('c', ['identity', 'logistic', 'tanh', 'relu']),
                    'MLP_maxiter': ('i', [100, 250]), 'solver': ('c', ['lbfgs', 'sgd', 'adam'])},
            'RNC': {'radius': ('r', [0, 10000]), 'RNC_weights': ('c', ['uniform', 'distance'])},
            'NCC': {'NCC_metric': ('c', ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis']),
                    'NCC_shrink_thre': ('r', [0, 10])},
            'EXtree': {'EX_criterion': ('c', ['gini', 'entropy']), 'EX_splitter': ('c', ['random', 'best']),
                       'EX_max_feature': ('r', [0.2, 1.0]), 'EX_min_samples_split': ('i', [2, 40]),
                       'EX_min_samples_leaf': ('i', [1, 20])},
            'adaBoost': {'ada_n_estimators': ('i', [10, 200]), 'ada_learning_rate': ('r', [0.01, 10])},
            'bagging': {'bag_n_estimators': ('i', [10, 200]), 'bag_max_samples': ('r', [0.7, 1.0]),
                        'bag_max_features': ('r', [0.7, 1.0])},
            'EXs': {'EXs_criterion': ('c', ['gini', 'entropy']), 'EXs_n_estimator': ('i', [10, 200]),
                    'EXs_max_feature': ('r', [0.2, 1.0]), 'EXs_min_samples_split': ('i', [2, 40]),
                    'EX_min_samples_leaf': ('i', [1, 20])}
        }
        
    
    his = []
    create_dir('resBL')
    fnameList('resBL', his)
    curr = 'resBL/'+fname+'-FE100.txt'
    print(curr)
    if curr in his:
        with open(curr, 'r') as f:
            lines = f.readlines()
            if len(lines)/4 >= repeat:
                return
            else:
                repeat = int(repeat - len(lines)/4)
    
    stime = time.time()
    for i in range(repeat):
        params = {'up': up, 'lp': lp}
        ex = Ulevel(params, xsource=Xsource, ysource=Lsource, xtarget=Xtarget, ytarget=Ltarget, loc=loc,
                    UFE=208, LFE=100, method=method, fname=fname)
        res, inc = ex.run()
        path = create_dir('resBL')
        with open(path + fname + '-FE100' + '.txt', 'a+') as f:
            print(inc, file=f)
            print(res, file=f)
            print('---------------------', file=f)
            print('time:', time.time() - stime, file=f)


# main function
if __name__ == '__main__':
    begin_num = 1
    end_num = 20


    flist = []
    group = sorted(['ReLink', 'AEEEM', 'JURECZKO'])

    for i in range(len(group)):
        tmp = []
        fnameList('data/' + group[i], tmp)
        tmp = sorted(tmp)
        flist.append(tmp)

    for c in range(begin_num, end_num + 1):
        if c in range(6):
            tmp = flist[0].copy()
            target = tmp.pop(c - 1)
        if c in range(6, 18):
            tmp = flist[1].copy()
            target = tmp.pop(c - 6)
        if c in range(18, 21):
            tmp = flist[2].copy()
            target = tmp.pop(c - 18)

        Xsource, Lsource, Xtarget, Ltarget, loc = MfindCommonMetric(tmp, target, split=True)
        bl(Xsource, Lsource, Xtarget, Ltarget, loc, target.split('/')[-1].split('.')[0], repeat=30, method='paratabu')
        if len(psutil.Process(os.getpid()).children()) >= 2:
           os.wait()
    print('done')
    os._exit(0)
