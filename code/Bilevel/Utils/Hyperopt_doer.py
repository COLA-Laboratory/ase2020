from iteration_utilities import deepflatten
from Algorithms.Framework import cpdp
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from func_timeout import func_set_timeout
import numpy as np


class optParamAll(object):
    def __init__(self, sx, sy, tx, ty, loc, classifier, adaptation, fe=1000, train=None, Ltrain=None):
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
        self.train = train
        self.Ltrain = Ltrain
        self.loc = loc
        self.adaptation = adaptation
        self.clf = classifier
        self.fe = fe
        self.trails = Trials()

    def objFunc(self, params):
        self.p = cpdp(clf=self.clf, adpt=self.adaptation)
        self.p.set_params(**params)
        sx = self.sx
        sy = self.sy
        tx = self.tx
        ty = self.ty
        loc = self.loc
        res = self.p.run(sx, sy, tx, ty, loc, train=self.train, Ltrain=self.Ltrain)

        return {'loss': -np.mean(res), 'status': STATUS_OK, 'result': np.mean(res)}

    @func_set_timeout(86400)
    def run(self):
        if self.adaptation == 'UM':
            adptdefault_value = {
                'pvalue': 0.05
            }
            adptparamSpace = {
                'pvalue': hp.uniform('pvalue', 0.01, 0.1)
            }

        if self.adaptation == 'TCAplus':
            adptdefault_value = {
                'kernel_type': 'linear',
                'dim': 5,
                'lamb': 1,
                'gamma': 1
            }
            adptparamSpace = {
                'kernel_type': hp.choice('kernel_type', ['primal', 'linear', 'rbf', 'sam']),
                'dim': hp.choice('dim', range(5, max(self.sx.shape[1], self.tx.shape[1]))),
                'lamb': hp.uniform('lamb', 1e-6, 1e2),
                'gamma': hp.uniform('gamma', 1e-5, 1e2)
            }

        if self.adaptation == 'PCAmining':
            adptdefault_value = {
                'pcaDim': 5
            }
            adptparamSpace = {
                'pcaDim': hp.choice('pcaDim', range(5, max(self.sx.shape[1], self.tx.shape[1])))
            }

        if self.adaptation == 'NNfilter':
            adptdefault_value = {
                'NNn_neighbors': 10,
                'NNmetric': 'euclidean'
            }
            adptparamSpace = {
                'NNn_neighbors': hp.choice('NNn_neighbors', range(1, 100)),
                'NNmetric': hp.choice('NNmetric', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
            }

        if self.adaptation == 'GIS':
            adptdefault_value = {
                'mProb': 0.05,
                'chrmsize': 0.02,
                'popsize': 30,
                'numparts': 5,
                'numgens': 20,
                'mCount': 5
            }
            adptparamSpace = {
                'mProb': hp.uniform('mProb', 0.02, 0.1),
                'chrmsize': hp.uniform('chrmsize', 0.02, 0.1),
                'popsize': hp.choice('popsize', range(2, 31, 2)),
                'numparts': hp.choice('numparts', range(2, 7)),
                'numgens': hp.choice('numgens', range(5, 21)),
                'mCount': hp.choice('mCount', range(3, 11))
            }

        if self.adaptation == 'CDE_SMOTE':
            adptdefault_value = {
                'CDE_k': 3,
                'CDE_metric': 'euclidean'
            }
            adptparamSpace = {
                'CDE_k': hp.choice('CDE_k', range(1, 100)),
                'CDE_metric': hp.choice('CDE_metric',
                                        ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis'])
            }

        if self.adaptation == 'CLIFE':
            adptdefault_value = {
                'Clife_n': 10,
                'Clife_alpha': 0.15,
                'Clife_beta': 0.35,
                'percentage': 0.8
            }
            adptparamSpace = {
                'Clife_n': hp.choice('Clife_n', range(1, 100)),
                'Clife_alpha': hp.uniform('Clife_alpha', 0.05, 0.2),
                'Clife_beta': hp.uniform('Clife_beta', 0.2, 0.4),
                'percentage': hp.uniform('percentage', 0.6, 0.9)
            }

        if self.adaptation == 'FeSCH':
            adptdefault_value = {
                'Fesch_nt': 1,
                'Fesch_strategy': 'SFD'
            }
            adptparamSpace = {
                'Fesch_nt': hp.choice('Fesch_nt', range(1, self.sx.shape[1])),
                'Fesch_strategy': hp.choice('Fesch_strategy', ['SFD', 'LDF', 'FCR'])
            }

        if self.adaptation == 'FSS_bagging':
            adptdefault_value = {
                'FSS_topn': 10,
                'FSS_ratio': 0.1,
                'FSS_score_thre': 0.5
            }
            adptparamSpace = {
                'FSS_topn': hp.choice('FSS_topn', range(1, len(self.loc))),
                'FSS_ratio': hp.uniform('FSS_ratio', 0.1, 0.9),
                'FSS_score_thre': hp.uniform('FSS_score_thre', 0.3, 0.7)
            }

        if self.adaptation == 'HISNN':
            adptdefault_value = {
                'HISNN_minham': 1.0
            }
            adptparamSpace = {
                'HISNN_minham': hp.choice('HISNN_minham', range(1, self.sx.shape[1]))
            }

        if self.adaptation == 'MCWs':
            adptdefault_value = {
                'MCW_k': 4,
                'MCW_sigmma': 1.0,
                'MCW_lamb': 1.0,
            }
            adptparamSpace = {
                'MCW_k': hp.choice('MCW_k', range(2, len(self.loc))),
                'MCW_sigmma': hp.uniform('MCW_sigmma', 0.01, 10),
                'MCW_lamb': hp.uniform('MCW_lamb', 1e-6, 1e2)
            }

        if self.adaptation == 'TD':
            adptdefault_value = {
                'TDparam': {'TD_strategy': 'NN', 'TD_num': 3, }
            }
            adptparamSpace = {
                'TDparam': hp.choice('TDparam', [
                    {'TD_strategy': 'NN', 'TD_num': hp.choice('TD_num', range(1, len(self.loc)))},
                    {'TD_strategy': 'EM'}
                ])
            }

        if self.adaptation == 'VCB':
            adptdefault_value = {
                'VCB_M': 30,
                'VCB_lamb': 1.0,
            }
            adptparamSpace = {
                'VCB_M': hp.choice('VCB_M', range(2, 30)),
                'VCB_lamb': hp.uniform('VCB_lamb', 0.5, 1.5),
            }

        if self.clf == 'RF':
            clfdefault_value = {
                'RFn_estimators': 10,
                'RFcriterion': 'gini',
                'RFmax_features': 1.0,
                'RFmin_samples_split': 2,
                'RFmin_samples_leaf': 1
            }
            clfparamSpace = {
                'RFn_estimators': hp.choice('RFn_estimators', range(10, 200)),
                'RFcriterion': hp.choice('RFcriterion', ['gini', 'entropy']),
                'RFmax_features': hp.uniform('RFmax_features', 0.2, 1.0),
                'RFmin_samples_split': hp.choice('RFmin_samples_split', range(2, 40)),
                'RFmin_samples_leaf': hp.choice('RFmin_samples_leaf', range(1, 20))
            }

        if self.clf == 'SVM':
            clfdefault_value = {
                'SVCkernel': {'kernel': 'linear', 'max_iter': -1},
                'svmC': 1.0
            }
            clfparamSpace = {
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

        if self.clf == 'KNN':
            clfdefault_value = {
                'KNNneighbors': 5,
                'KNNp': 2
            }
            clfparamSpace = {
                'KNNneighbors': hp.choice('KNNneighbors', range(1, 50)),
                'KNNp': hp.choice('KNNp', range(1, 5))
            }

        if self.clf == 'NB':
            clfdefault_value = {
                'NBparam': {'NBType': 'gaussian'}
            }
            clfparamSpace = {
                'NBparam': hp.choice('NBparam', [
                    {'NBType': 'gaussian'},
                    {'NBType': 'multinomial', 'malpha': hp.uniform('malpha', 0, 10)},
                    {'NBType': 'complement', 'calpha': hp.uniform('calpha', 0, 10),
                     'norm': hp.choice('norm', [True, False])}])

            }

        if self.clf == 'DT':
            clfdefault_value = {
                'DTcriterion': 'gini',
                'DTmax_features': 1.0,
                'DTsplitter': 'best',
                'DTmin_samples_split': 2,
                'DTmin_samples_leaf': 1
            }
            clfparamSpace = {
                'DTcriterion': hp.choice('DTcriterion', ['gini', 'entropy']),
                'DTmax_features': hp.uniform('DTmax_features', 0.2, 1.0),
                'DTsplitter': hp.choice('DTsplitter', ['best', 'random']),
                'DTmin_samples_split': hp.choice('DTmin_samples_split', range(2, 40)),
                'DTmin_samples_leaf': hp.choice('DTmin_samples_leaf', range(1, 20))
            }

        if self.clf == 'LR':
            clfdefault_value = {
                'penalty': 'l2',
                'lrC': 1.0,
                'maxiter': 100,
                'fit_intercept': True
            }
            clfparamSpace = {
                'penalty': hp.choice('penalty', ['l1', 'l2']),
                'lrC': hp.uniform('lrC', 0.001, 10),
                'maxiter': hp.choice('maxiter', range(50, 200)),
                'fit_intercept': hp.choice('fit_intercept', [True, False])
            }

        paramSpace = dict(adptparamSpace, **clfparamSpace)
        default_value = dict(adptdefault_value, **clfdefault_value)
        self.def_value = self.objFunc(default_value)['result']
        best = fmin(self.objFunc, space=paramSpace, algo=tpe.suggest, max_evals=self.fe, trials=self.trails)

        his = dict()
        his['name'] = list(self.trails.trials[0]['misc']['vals'].keys())
        i = 0
        for item in self.trails.trials:
            results = list(deepflatten(item['misc']['vals'].values()))
            results.append(item['result']['result'])
            his[i] = results
            i += 1

        inc_value = self.trails.best_trial['result']['result']

        return np.asarray([self.def_value, inc_value]), his, best


class optParamAdpt(object):
    def __init__(self, sx, sy, tx, ty, loc, classifier, adaptation, fe=1000, train=None, Ltrain=None):
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
        self.train = train
        self.Ltrain = Ltrain
        self.loc = loc
        self.adaptation = adaptation
        self.clf = classifier
        self.fe = fe
        self.trails = Trials()

    def objFunc(self, params):
        self.p = cpdp(clf=self.clf, adpt=self.adaptation)
        self.p.set_params(**params)
        sx = self.sx
        sy = self.sy
        tx = self.tx
        ty = self.ty
        loc = self.loc
        res = self.p.run(sx, sy, tx, ty, loc, train=self.train, Ltrain=self.Ltrain)

        return {'loss': -np.mean(res), 'status': STATUS_OK, 'result': np.mean(res)}

    @func_set_timeout(30)
    def run(self):
        if self.adaptation == 'UM':
            adptdefault_value = {
                'pvalue': 0.05
            }
            adptparamSpace = {
                'pvalue': hp.uniform('pvalue', 0.01, 0.1)
            }

        if self.adaptation == 'TCAplus':
            adptdefault_value = {
                'kernel_type': 'linear',
                'dim': 5,
                'lamb': 1,
                'gamma': 1
            }
            adptparamSpace = {
                'kernel_type': hp.choice('kernel_type', ['primal', 'linear', 'rbf', 'sam']),
                'dim': hp.choice('dim', range(5, max(self.sx.shape[1], self.tx.shape[1]))),
                'lamb': hp.uniform('lamb', 1e-6, 1e2),
                'gamma': hp.uniform('gamma', 1e-5, 1e2)
            }

        if self.adaptation == 'PCAmining':
            adptdefault_value = {
                'pcaDim': 5
            }
            adptparamSpace = {
                'pcaDim': hp.choice('pcaDim', range(5, max(self.sx.shape[1], self.tx.shape[1])))
            }

        if self.adaptation == 'NNfilter':
            adptdefault_value = {
                'NNn_neighbors': 10,
                'NNmetric': 'euclidean'
            }
            adptparamSpace = {
                'NNn_neighbors': hp.choice('NNn_neighbors', range(1, 100)),
                'NNmetric': hp.choice('NNmetric', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
            }

        if self.adaptation == 'GIS':
            adptdefault_value = {
                'mProb': 0.05,
                'chrmsize': 0.02,
                'popsize': 30,
                'numparts': 5,
                'numgens': 20,
                'mCount': 5
            }
            adptparamSpace = {
                'mProb': hp.uniform('mProb', 0.02, 0.1),
                'chrmsize': hp.uniform('chrmsize', 0.02, 0.1),
                'popsize': hp.choice('popsize', range(2, 31, 2)),
                'numparts': hp.choice('numparts', range(2, 7)),
                'numgens': hp.choice('numgens', range(5, 21)),
                'mCount': hp.choice('mCount', range(3, 11))
            }

        if self.adaptation == 'CDE_SMOTE':
            adptdefault_value = {
                'CDE_k': 3,
                'CDE_metric': 'euclidean'
            }
            adptparamSpace = {
                'CDE_k': hp.choice('CDE_k', range(1, 100)),
                'CDE_metric': hp.choice('CDE_metric',
                                        ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis'])
            }

        if self.adaptation == 'CLIFE':
            adptdefault_value = {
                'Clife_n': 10,
                'Clife_alpha': 0.15,
                'Clife_beta': 0.35,
                'percentage': 0.8
            }
            adptparamSpace = {
                'Clife_n': hp.choice('Clife_n', range(1, 100)),
                'Clife_alpha': hp.uniform('Clife_alpha', 0.05, 0.2),
                'Clife_beta': hp.uniform('Clife_beta', 0.2, 0.4),
                'percentage': hp.uniform('percentage', 0.6, 0.9)
            }

        if self.adaptation == 'FeSCH':
            adptdefault_value = {
                'Fesch_nt': 1,
                'Fesch_strategy': 'SFD'
            }
            adptparamSpace = {
                'Fesch_nt': hp.choice('Fesch_nt', range(1, self.sx.shape[1])),
                'Fesch_strategy': hp.choice('Fesch_strategy', ['SFD', 'LDF', 'FCR'])
            }

        if self.adaptation == 'FSS_bagging':
            adptdefault_value = {
                'FSS_topn': 10,
                'FSS_ratio': 0.1,
                'FSS_score_thre': 0.5
            }
            adptparamSpace = {
                'FSS_topn': hp.choice('FSS_topn', range(1, len(self.loc))),
                'FSS_ratio': hp.uniform('FSS_ratio', 0.1, 0.9),
                'FSS_score_thre': hp.uniform('FSS_score_thre', 0.3, 0.7)
            }

        if self.adaptation == 'HISNN':
            adptdefault_value = {
                'HISNN_minham': 1.0
            }
            adptparamSpace = {
                'HISNN_minham': hp.choice('HISNN_minham', range(1, self.sx.shape[1]))
            }

        if self.adaptation == 'MCWs':
            adptdefault_value = {
                'MCW_k': 4,
                'MCW_sigmma': 1.0,
                'MCW_lamb': 1.0,
            }
            adptparamSpace = {
                'MCW_k': hp.choice('MCW_k', range(2, len(self.loc))),
                'MCW_sigmma': hp.uniform('MCW_sigmma', 0.01, 10),
                'MCW_lamb': hp.uniform('MCW_lamb', 1e-6, 1e2)
            }

        if self.adaptation == 'TD':
            adptdefault_value = {
                'TDparam': {'TD_strategy': 'NN', 'TD_num': 3, }
            }
            adptparamSpace = {
                'TDparam': hp.choice('TDparam', [
                    {'TD_strategy': 'NN', 'TD_num': hp.choice('TD_num', range(1, len(self.loc)))},
                    {'TD_strategy': 'EM'}
                ])
            }

        if self.adaptation == 'VCB':
            adptdefault_value = {
                'VCB_M': 30,
                'VCB_lamb': 1.0,
            }
            adptparamSpace = {
                'VCB_M': hp.choice('VCB_M', range(2, 30)),
                'VCB_lamb': hp.uniform('VCB_lamb', 0.5, 1.5),
            }

        self.def_value = self.objFunc(adptdefault_value)['result']
        best = fmin(self.objFunc, space=adptparamSpace, algo=tpe.suggest, max_evals=self.fe, trials=self.trails)
        his = dict()
        his['name'] = list(self.trails.trials[0]['misc']['vals'].keys())
        i = 0
        for item in self.trails.trials:
            results = list(deepflatten(item['misc']['vals'].values()))
            results.append(item['result']['result'])
            his[i] = results
            i += 1

        inc_value = self.trails.best_trial['result']['result']

        return np.asarray([self.def_value, inc_value]), his, best


class optParamCLF(object):
    def __init__(self, sx, sy, tx, ty, loc, classifier, adaptation, fe=1000, train=None, Ltrain=None):
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
        self.train = train
        self.Ltrain = Ltrain
        self.loc = loc
        self.adaptation = adaptation
        self.clf = classifier
        self.fe = fe
        self.trails = Trials()

    def objFunc(self, params):
        self.p = cpdp(clf=self.clf, adpt=self.adaptation)
        self.p.set_params(**params)
        sx = self.sx
        sy = self.sy
        tx = self.tx
        ty = self.ty
        loc = self.loc
        res = self.p.run(sx, sy, tx, ty, loc, train=self.train, Ltrain=self.Ltrain)

        return {'loss': -np.mean(res), 'status': STATUS_OK, 'result': np.mean(res)}

    @func_set_timeout(86400)
    def run(self):
        if self.clf == 'RF':
            clfdefault_value = {
                'RFn_estimators': 10,
                'RFcriterion': 'gini',
                'RFmax_features': 1.0,
                'RFmin_samples_split': 2,
                'RFmin_samples_leaf': 1
            }
            clfparamSpace = {
                'RFn_estimators': hp.choice('RFn_estimators', range(10, 200)),
                'RFcriterion': hp.choice('RFcriterion', ['gini', 'entropy']),
                'RFmax_features': hp.uniform('RFmax_features', 0.2, 1.0),
                'RFmin_samples_split': hp.choice('RFmin_samples_split', range(2, 40)),
                'RFmin_samples_leaf': hp.choice('RFmin_samples_leaf', range(1, 20))
            }

        if self.clf == 'SVM':
            clfdefault_value = {
                'SVCkernel': {'kernel': 'linear', 'max_iter': -1},
                'svmC': 1.0
            }
            clfparamSpace = {
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

        if self.clf == 'KNN':
            clfdefault_value = {
                'KNNneighbors': 5,
                'KNNp': 2
            }
            clfparamSpace = {
                'KNNneighbors': hp.choice('KNNneighbors', range(1, 50)),
                'KNNp': hp.choice('KNNp', range(1, 5))
            }

        if self.clf == 'NB':
            clfdefault_value = {
                'NBparam': {'NBType': 'gaussian'}
            }
            clfparamSpace = {
                'NBparam': hp.choice('NBparam', [
                    {'NBType': 'gaussian'},
                    {'NBType': 'multinomial', 'malpha': hp.uniform('malpha', 0, 10)},
                    {'NBType': 'complement', 'calpha': hp.uniform('calpha', 0, 10),
                     'norm': hp.choice('norm', [True, False])}])

            }

        if self.clf == 'DT':
            clfdefault_value = {
                'DTcriterion': 'gini',
                'DTmax_features': 1.0,
                'DTsplitter': 'best',
                'DTmin_samples_split': 2,
                'DTmin_samples_leaf': 1
            }
            clfparamSpace = {
                'DTcriterion': hp.choice('DTcriterion', ['gini', 'entropy']),
                'DTmax_features': hp.uniform('DTmax_features', 0.2, 1.0),
                'DTsplitter': hp.choice('DTsplitter', ['best', 'random']),
                'DTmin_samples_split': hp.choice('DTmin_samples_split', range(2, 40)),
                'DTmin_samples_leaf': hp.choice('DTmin_samples_leaf', range(1, 20))
            }

        if self.clf == 'LR':
            clfdefault_value = {
                'penalty': 'l2',
                'lrC': 1.0,
                'maxiter': 100,
                'fit_intercept': True
            }
            clfparamSpace = {
                'penalty': hp.choice('penalty', ['l1', 'l2']),
                'lrC': hp.uniform('lrC', 0.001, 10),
                'maxiter': hp.choice('maxiter', range(50, 200)),
                'fit_intercept': hp.choice('fit_intercept', [True, False])
            }

        self.def_value = self.objFunc(clfdefault_value)['result']
        best = fmin(self.objFunc, space=clfparamSpace, algo=tpe.suggest, max_evals=self.fe, trials=self.trails)

        his = dict()
        his['name'] = list(self.trails.trials[0]['misc']['vals'].keys())
        i = 0
        for item in self.trails.trials:
            results = list(deepflatten(item['misc']['vals'].values()))
            results.append(item['result']['result'])
            his[i] = results
            i += 1

        inc_value = self.trails.best_trial['result']['result']

        # print(def_value)
        return np.asarray([self.def_value, inc_value]), his, best


class optParamSEQ(object):
    def __init__(self, sx, sy, tx, ty, loc, classifier, adaptation, fe=1000, train=None, Ltrain=None):
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
        self.train = train
        self.Ltrain = Ltrain
        self.loc = loc
        self.adaptation = adaptation
        self.clf = classifier
        self.fe = fe
        self.trails = Trials()
        self.Atrails = Trials()

        self.SEQ = 0

    def objFunc(self, params):
        if self.SEQ == 1:
            params = dict(params, **self.Adptbest)
        self.p = cpdp(clf=self.clf, adpt=self.adaptation)
        self.p.set_params(**params)
        sx = self.sx
        sy = self.sy
        tx = self.tx
        ty = self.ty
        loc = self.loc
        res = self.p.run(sx, sy, tx, ty, loc, train=self.train, Ltrain=self.Ltrain)

        return {'loss': -np.mean(res), 'status': STATUS_OK, 'result': np.mean(res)}

    @func_set_timeout(86400)
    def run(self):
        if self.adaptation == 'UM':
            adptdefault_value = {
                'pvalue': 0.05
            }
            adptparamSpace = {
                'pvalue': hp.uniform('pvalue', 0.01, 0.1)
            }

        if self.adaptation == 'TCAplus':
            adptdefault_value = {
                'kernel_type': 'linear',
                'dim': 5,
                'lamb': 1,
                'gamma': 1
            }
            adptparamSpace = {
                'kernel_type': hp.choice('kernel_type', ['primal', 'linear', 'rbf', 'sam']),
                'dim': hp.choice('dim', range(5, max(self.sx.shape[1], self.tx.shape[1]))),
                'lamb': hp.uniform('lamb', 1e-6, 1e2),
                'gamma': hp.uniform('gamma', 1e-5, 1e2)
            }

        if self.adaptation == 'PCAmining':
            adptdefault_value = {
                'pcaDim': 5
            }
            adptparamSpace = {
                'pcaDim': hp.choice('pcaDim', range(5, max(self.sx.shape[1], self.tx.shape[1])))
            }

        if self.adaptation == 'NNfilter':
            adptdefault_value = {
                'NNn_neighbors': 10,
                'NNmetric': 'euclidean'
            }
            adptparamSpace = {
                'NNn_neighbors': hp.choice('NNn_neighbors', range(1, 100)),
                'NNmetric': hp.choice('NNmetric', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
            }

        if self.adaptation == 'GIS':
            adptdefault_value = {
                'mProb': 0.05,
                'chrmsize': 0.02,
                'popsize': 30,
                'numparts': 5,
                'numgens': 20,
                'mCount': 5
            }
            adptparamSpace = {
                'mProb': hp.uniform('mProb', 0.02, 0.1),
                'chrmsize': hp.uniform('chrmsize', 0.02, 0.1),
                'popsize': hp.choice('popsize', range(2, 31, 2)),
                'numparts': hp.choice('numparts', range(2, 7)),
                'numgens': hp.choice('numgens', range(5, 21)),
                'mCount': hp.choice('mCount', range(3, 11))
            }

        if self.adaptation == 'CDE_SMOTE':
            adptdefault_value = {
                'CDE_k': 3,
                'CDE_metric': 'euclidean'
            }
            adptparamSpace = {
                'CDE_k': hp.choice('CDE_k', range(1, 100)),
                'CDE_metric': hp.choice('CDE_metric',
                                        ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis'])
            }

        if self.adaptation == 'CLIFE':
            adptdefault_value = {
                'Clife_n': 10,
                'Clife_alpha': 0.15,
                'Clife_beta': 0.35,
                'percentage': 0.8
            }
            adptparamSpace = {
                'Clife_n': hp.choice('Clife_n', range(1, 100)),
                'Clife_alpha': hp.uniform('Clife_alpha', 0.05, 0.2),
                'Clife_beta': hp.uniform('Clife_beta', 0.2, 0.4),
                'percentage': hp.uniform('percentage', 0.6, 0.9)
            }

        if self.adaptation == 'FeSCH':
            adptdefault_value = {
                'Fesch_nt': 1,
                'Fesch_strategy': 'SFD'
            }
            adptparamSpace = {
                'Fesch_nt': hp.choice('Fesch_nt', range(1, self.sx.shape[1])),
                'Fesch_strategy': hp.choice('Fesch_strategy', ['SFD', 'LDF', 'FCR'])
            }

        if self.adaptation == 'FSS_bagging':
            adptdefault_value = {
                'FSS_topn': 10,
                'FSS_ratio': 0.1,
                'FSS_score_thre': 0.5
            }
            adptparamSpace = {
                'FSS_topn': hp.choice('FSS_topn', range(1, len(self.loc))),
                'FSS_ratio': hp.uniform('FSS_ratio', 0.1, 0.9),
                'FSS_score_thre': hp.uniform('FSS_score_thre', 0.3, 0.7)
            }

        if self.adaptation == 'HISNN':
            adptdefault_value = {
                'HISNN_minham': 1.0
            }
            adptparamSpace = {
                'HISNN_minham': hp.choice('HISNN_minham', range(1, self.sx.shape[1]))
            }

        if self.adaptation == 'MCWs':
            adptdefault_value = {
                'MCW_k': 4,
                'MCW_sigmma': 1.0,
                'MCW_lamb': 1.0,
            }
            adptparamSpace = {
                'MCW_k': hp.choice('MCW_k', range(2, len(self.loc))),
                'MCW_sigmma': hp.uniform('MCW_sigmma', 0.01, 10),
                'MCW_lamb': hp.uniform('MCW_lamb', 1e-6, 1e2)
            }

        if self.adaptation == 'TD':
            adptdefault_value = {
                'TDparam': {'TD_strategy': 'NN', 'TD_num': 3, }
            }
            adptparamSpace = {
                'TDparam': hp.choice('TDparam', [
                    {'TD_strategy': 'NN', 'TD_num': hp.choice('TD_num', range(1, len(self.loc)))},
                    {'TD_strategy': 'EM'}
                ])
            }

        if self.adaptation == 'VCB':
            adptdefault_value = {
                'VCB_M': 30,
                'VCB_lamb': 1.0,
            }
            adptparamSpace = {
                'VCB_M': hp.choice('VCB_M', range(2, 30)),
                'VCB_lamb': hp.uniform('VCB_lamb', 0.5, 1.5),
            }

        if self.clf == 'RF':
            clfdefault_value = {
                'RFn_estimators': 10,
                'RFcriterion': 'gini',
                'RFmax_features': 1.0,
                'RFmin_samples_split': 2,
                'RFmin_samples_leaf': 1
            }
            clfparamSpace = {
                'RFn_estimators': hp.choice('RFn_estimators', range(10, 200)),
                'RFcriterion': hp.choice('RFcriterion', ['gini', 'entropy']),
                'RFmax_features': hp.uniform('RFmax_features', 0.2, 1.0),
                'RFmin_samples_split': hp.choice('RFmin_samples_split', range(2, 40)),
                'RFmin_samples_leaf': hp.choice('RFmin_samples_leaf', range(1, 20))
            }

        if self.clf == 'SVM':
            clfdefault_value = {
                'SVCkernel': {'kernel': 'linear', 'max_iter': -1},
                'svmC': 1.0
            }
            clfparamSpace = {
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

        if self.clf == 'KNN':
            clfdefault_value = {
                'KNNneighbors': 5,
                'KNNp': 2
            }
            clfparamSpace = {
                'KNNneighbors': hp.choice('KNNneighbors', range(1, 50)),
                'KNNp': hp.choice('KNNp', range(1, 5))
            }

        if self.clf == 'NB':
            clfdefault_value = {
                'NBparam': {'NBType': 'gaussian'}
            }
            clfparamSpace = {
                'NBparam': hp.choice('NBparam', [
                    {'NBType': 'gaussian'},
                    {'NBType': 'multinomial', 'malpha': hp.uniform('malpha', 0, 10)},
                    {'NBType': 'complement', 'calpha': hp.uniform('calpha', 0, 10),
                     'norm': hp.choice('norm', [True, False])}])

            }

        if self.clf == 'DT':
            clfdefault_value = {
                'DTcriterion': 'gini',
                'DTmax_features': 1.0,
                'DTsplitter': 'best',
                'DTmin_samples_split': 2,
                'DTmin_samples_leaf': 1
            }
            clfparamSpace = {
                'DTcriterion': hp.choice('DTcriterion', ['gini', 'entropy']),
                'DTmax_features': hp.uniform('DTmax_features', 0.2, 1.0),
                'DTsplitter': hp.choice('DTsplitter', ['best', 'random']),
                'DTmin_samples_split': hp.choice('DTmin_samples_split', range(2, 40)),
                'DTmin_samples_leaf': hp.choice('DTmin_samples_leaf', range(1, 20))
            }

        if self.clf == 'LR':
            clfdefault_value = {
                'penalty': 'l2',
                'lrC': 1.0,
                'maxiter': 100,
                'fit_intercept': True
            }
            clfparamSpace = {
                'penalty': hp.choice('penalty', ['l1', 'l2']),
                'lrC': hp.uniform('lrC', 0.001, 10),
                'maxiter': hp.choice('maxiter', range(50, 200)),
                'fit_intercept': hp.choice('fit_intercept', [True, False])
            }

        default_value = dict(adptdefault_value, **clfdefault_value)
        self.def_value = self.objFunc(default_value)['result']
        self.Adptbest = fmin(self.objFunc, space=adptparamSpace, algo=tpe.suggest, max_evals=int(self.fe * 0.5),
                             trials=self.Atrails)
        self.Adptbest = space_eval(adptparamSpace, self.Adptbest)

        his = dict()
        try:
            his['name'] = list(self.Atrails.trials[0]['misc']['vals'].keys()) + list(clfdefault_value.keys())
        except:
            his['name'] = [None]
        i = 0
        for item in self.Atrails.trials:
            if item['state'] == 2:
                results = list(deepflatten(item['misc']['vals'].values())) + list(clfdefault_value.values())
                results.append(item['result']['result'])
                his[i] = results
                i += 1

        self.SEQ = 1
        Clfbest = fmin(self.objFunc, space=clfparamSpace, algo=tpe.suggest, max_evals=int(self.fe * 0.5),
                       trials=self.trails)

        try:
            his['name1'] = list(self.Adptbest.keys()) + list(self.trails.trials[0]['misc']['vals'].keys())
        except:
            his['name1'] = [None]
        for item in self.trails.trials:
            if item['state'] == 2:
                results = list(self.Adptbest.values()) + list(deepflatten(item['misc']['vals'].values()))
                results.append(item['result']['result'])
                his[i] = results
                i += 1

        inc_value = self.trails.best_trial['result']['result']

        return np.asarray([self.def_value, inc_value]), his, Clfbest
