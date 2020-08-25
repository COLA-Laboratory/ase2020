from hyperopt import hp, STATUS_OK, fmin, tpe, Trials, atpe
from sklearn.model_selection import train_test_split

from Algorithms.Framework import cpdp
from Utils.File import fnameList, create_dir
from Utils.helper import MfindCommonMetric
import warnings, time

warnings.filterwarnings('ignore')
from func_timeout import FunctionTimedOut, func_set_timeout
import numpy as np


class AFO(object):
    def __init__(self, xsource, ysource, xtarget, ytarget, loc):
        self.xsource = xsource
        self.ysource = ysource
        self.xtarget = xtarget
        self.ytarget = ytarget
        self.loc = loc
        self.trails = Trials()

        # print('init')
        return

    def search_space(self):
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
                'CLIFE': {'Clife_n': ('i', [1, 100]), 'Clife_alpha': ('r', [0.05, 0.2]),
                          'Clife_beta': ('r', [0.2, 0.4]), \
                          'percentage': ('r', [0.6, 0.9])},
                'FeSCH': {'Fesch_nt': ('i', [1, Xsource.shape[1]]), 'Fesch_strategy': ('c', ['SFD', 'LDF', 'FCR'])},
                'FSS_bagging': {'FSS_topn': ('i', [1, len(loc)]), 'FSS_ratio': ('r', [0.1, 0.9]),
                                'FSS_score_thre': ('r', [0.3, 0.7])},
                'HISNN': {'HISNN_minham': ('i', [1, Xsource.shape[1]])},
                'MCWs': {'MCW_k': ('i', [2, len(loc)]), 'MCW_sigmma': ('r', [0.01, 10]),
                         'MCW_lamb': ('r', [1e-6, 1e2])},
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
                        'EXs_min_samples_leaf': ('i', [1, 20])}
            }

        adpt = []
        clf = []
        for k, v in lp.items():
            b = dict()
            uu = dict()
            ll = dict()
            for key, val in v.items():
                if val[0] == 'i':
                    low = min(val[1][0], val[1][1])
                    high = max(val[1][0], val[1][1])
                    if len(val[1]) == 3:
                        b[key] = hp.choice(key, range(low, high, val[1][2]))
                    else:
                        b[key] = hp.choice(key, range(low, high))
                if val[0] == 'c':
                    b[key] = hp.choice(key, val[1])
                if val[0] == 'r':
                    low = min(val[1][0], val[1][1])
                    high = max(val[1][0], val[1][1])
                    b[key] = hp.uniform(key, low, high)

            if k in up['adpt'][1]:
                uu[k] = hp.choice(k, [b])
                adpt.append(uu)
            if k in up['clf'][1]:
                ll[k] = hp.choice(k, [b])
                clf.append(ll)
        clf.append({'SVM': hp.choice('SVM', [
            {
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
        ])})

        clf.append({'NB': hp.choice('NB', [
            {
                'NBparam': hp.choice('NBparam', [
                    {'NBType': 'gaussian'},
                    {'NBType': 'multinomial', 'malpha': hp.uniform('malpha', 0, 10)},
                    {'NBType': 'complement', 'calpha': hp.uniform('calpha', 0, 10),
                     'norm': hp.choice('norm', [True, False])}])

            }
        ])})

        adpt.append({'TD': hp.choice('TD', [
            {
                'TDparam': hp.choice('TDparam', [
                    {'TD_strategy': 'NN', 'TD_num': hp.choice('TD_num', range(1, len(self.loc)))},
                    {'TD_strategy': 'EM'}
                ])
            }
        ])})

        spaces = dict()
        spaces['adpt'] = hp.choice('adpt', adpt)
        spaces['clf'] = hp.choice('clf', clf)
        return spaces

    def objFunc(self, params):
        clfparams = params['clf']
        adptparams = params['adpt']

        clf = list(clfparams.keys())[0]
        adpt = list(adptparams.keys())[0]
        params = {**clfparams[clf], **adptparams[adpt]}
        if clf in ['KNN', 'NCC', 'RNC', 'MLP', 'PAC'] and adpt in ['VCB', 'MCWs']:
            res = 0
        else:
            p = cpdp(clf=clf, adpt=adpt, repeat=10)
            p.set_params(**params)

            if adpt in ['NNfilter', 'MCWs']:
                train, test, ytrain, ytest = train_test_split(self.xtarget, self.ytarget, test_size=0.9,
                                                              random_state=42)
                try:
                    res = p.run(self.xsource, self.ysource, train, ytrain, self.loc, train=test, Ltrain=ytest)
                except:
                    if len(p.res) == 0:
                        res = 0
                    else:
                        res = np.mean(np.asarray(p.res))
            else:
                try:
                    res = p.run(self.xsource, self.ysource, self.xtarget, self.ytarget, self.loc)
                except:
                    if len(p.res) == 0:
                        res = 0
                    else:
                        res = np.mean(np.asarray(p.res))

        return {'loss': -res, 'status': STATUS_OK, 'result': res}

    @func_set_timeout(3600)
    def parameter_optimization(self):
        spaces = self.search_space()
        fmin(self.objFunc, algo=tpe.suggest, space=spaces, max_evals=10000000, trials=self.trails,
             show_progressbar=False)

    def run(self):
        try:
            self.parameter_optimization()
        except:
            count = 0
            for item in self.trails.trials:
                if item['state'] == 2:
                    count += 1

            if count > 0:
                inc_value = self.trails.best_trial['result']['result']
                best = self.trails.best_trial['misc']['vals']
            else:
                try:
                    inc_value = self.trails.best_trial['result']['result']
                    best = self.trails.best_trial['misc']['vals']
                except:
                    inc_value = 0
                    best = []
        config = dict()
        if len(best) > 0:
            for k, v in best.items():
                if len(v) != 0:
                    config[k] = v
        else:
            config = best

        return config, inc_value


# main function
if __name__ == '__main__':
    begin_num = 18
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

        repeat = 10
        donef = []
        create_dir('./resAFO')
        fnameList('./resAFO', donef)
        if './resAFO/' + target.split('/')[-1].split('.')[0] + '.txt' in donef:
            with open('./resAFO/' + target.split('/')[-1].split('.')[0] + '.txt', 'r') as f:
                lines = f.readlines()
                print(len(lines)/3, './resAFO/' + target.split('/')[-1].split('.')[0] + '.txt')
                if len(lines) / 3 < 10:
                    repeat = int(10 - len(lines) / 3)
                else:
                    continue

        print('======' + target.split('/')[-1].split('.')[0] + '======')
        for i in range(repeat):
            print('No. ', i + 1, ' ....')
            stime = time.time()
            m = AFO(xsource=Xsource, ysource=Lsource, xtarget=Xtarget, ytarget=Ltarget, loc=loc)
            config, inc_value = m.run()
            print(config, inc_value)
            # print(time.time() - stime)
            # path = create_dir('resAFO')
            # with open(path + target.split('/')[-1].split('.')[0] + '.txt', 'a+') as f:
            #     print(config, inc_value, file=f)
            #     print('time:', time.time() - stime, file=f)
            #     print('---------------------------', file=f)
    print('done')
