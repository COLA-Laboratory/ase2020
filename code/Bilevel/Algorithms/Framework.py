from sklearn.metrics import roc_auc_score
from Algorithms.transfer import TCAplus, NNfilter, PCAmining, GIS, UM, CDE_SMOTE, CLIFE_MORPH, FeSCH, FSS_bagging, \
    HISNN, multiple_components_weights, Training_Data_Selection, Value_Cognitive_Boosting
from collections import defaultdict
from Algorithms.classifier import *
import numpy as np


class cpdp(object):
    def __init__(self, kernel_type='linear', dim=5, lamb=1, gamma=1,  # TCA
                 NNn_neighbors=5, NNmetric='euclidean',  # NNfilter
                 pcaDim=5,  # PCAmining
                 pvalue=0.05,  # Universal
                 mProb=0.05, mCount=5, popsize=30, chrmsize=0.02, numgens=20, numparts=5,  # GIS
                 CDE_k=3, CDE_metric='euclidean',  # CDE_SMOTE
                 Clife_n=10, Clife_alpha=0.15, Clife_beta=0.35, percentage=0.8,  # CLIFE_SMOTE
                 Fesch_nt=1, Fesch_strategy='SFD',  # FeSCH
                 FSS_topn=10, FSS_ratio=0.1, FSS_score_thre=0.5,  # FSS_bagging
                 HISNN_minham=1.0,  # HISNN
                 MCW_k=4, MCW_sigmma=1.0, MCW_lamb=1.0,  # MCWs
                 TD_strategy='NN', TD_num=3,  # Training-Data-Selection
                 VCB_M=30, VCB_lamb=1.0,  # VCB

                 RFn_estimators=10, RFcriterion='gini', RFmax_features=1.0, RFmin_samples_split=2,
                 RFmin_samples_leaf=1,  # RF
                 DTcriterion='gini', DTmax_features=1.0, DTsplitter='best',
                 DTmin_samples_split=2, DTmin_samples_leaf=1,  # Decision-Tree
                 KNNneighbors=5, KNNp=2,  # KNN
                 penalty='l2', lrC=1.0, maxiter=100, fit_intercept=True,  # Logistic regression
                 Ridge_alpha=0.001, Ridge_fit_intercept=False, Ridge_tol=1e-3,  # Ridge
                 PAC_c=1e-3, PAC_fit_intercept=False, PAC_tol=1e-3, PAC_loss='hinge',
                 Per_penalty='l2', Per_alpha=1, Per_fit_intercept=False, Per_tol=1e-3,
                 MLP_hidden_layer_sizes=100, MLP_activation='relu', MLP_maxiter=200, solver='adam',
                 radius=1, RNC_weights='uniform',
                 NCC_metric='euclidean', NCC_shrink_thre=0,
                 EX_criterion='gini', EX_splitter='best', EX_max_feature=1.0, EX_min_samples_split=2,
                 EX_min_samples_leaf=1,
                 ada_n_estimators=100, ada_learning_rate=0.1,
                 bag_n_estimators=100, bag_max_samples=1.0, bag_max_features=1.0,
                 EXs_criterion='gini', EXs_max_feature=1.0, EXs_min_samples_split=2, EXs_min_samples_leaf=1,
                 EXs_n_estimator=100,

                 clf='SVM', adpt='TCA',  # TL and Classifier
                 repeat=10
                 ):
        # the number of repetition of algorithm
        self.repeat = repeat

        # TL and Classifier
        self.clfType = clf
        self.adpt = adpt

        # Universal
        self.pvalue = pvalue
        # TCA
        self.kernelType = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        # PCAmining
        self.pcaDim = pcaDim
        # GIS
        self.mProb = mProb
        self.mCount = mCount
        self.popsize = popsize
        self.chrmsize = chrmsize
        self.numgens = numgens
        self.numparts = numparts
        # NNfilter
        self.NNn_neighbors = NNn_neighbors
        self.NNmetric = NNmetric
        # CDE-SMOTE
        self.CDE_k = CDE_k
        self.CDE_metric = CDE_metric
        # CLIFE-MORPH
        self.Cliff_alpha = Clife_alpha
        self.Clife_beta = Clife_beta
        self.Clife_n = Clife_n
        self.percentage = percentage
        # FeSCH
        self.Fesch_nt = Fesch_nt
        self.Fesch_strategy = Fesch_strategy
        # FSS_bagging
        self.FSS_topn = FSS_topn
        self.FSS_ratio = FSS_ratio
        self.FSS_score_thre = FSS_score_thre
        # HISNN
        self.HISNNminham = HISNN_minham
        # MCWs
        self.MCW_k = MCW_k
        self.MCW_lamb = MCW_lamb
        self.MCW_sigmma = MCW_sigmma
        # TD
        self.TD_num = TD_num
        self.TD_strategy = TD_strategy
        # VCB
        self.VCB_lamb = VCB_lamb
        self.VCB_M = VCB_M

        # SVM
        self.SVCkernel = 'linear'
        self.coef0 = 0
        self.gamma = 1
        self.degree = 3
        self.svmC = 1
        self.svmMaxiter = -1
        # KNN
        self.KNNneighbors = KNNneighbors
        self.KNNp = KNNp
        # RF
        self.RFn_estimators = RFn_estimators
        self.RFcriterion = RFcriterion
        self.RFmax_features = RFmax_features
        self.RFmin_samples_split = RFmin_samples_split
        self.RFmin_samples_leaf = RFmin_samples_leaf
        # NB
        self.NBType = 'gaussian'
        self.alpha = 1.0
        self.norm = True
        # Decision Tree
        self.DTsplitter = DTsplitter
        self.DTcriterion = DTcriterion
        self.DTmax_features = DTmax_features
        self.DTmin_samples_leaf = DTmin_samples_leaf
        self.DTmin_samples_split = DTmin_samples_split
        # LR
        self.penalty = penalty
        self.lrC = lrC
        self.maxiter = maxiter
        self.fit_intercept = fit_intercept
        # Ridge
        self.Ridge_alpha = Ridge_alpha
        self.Ridge_fit = Ridge_fit_intercept
        self.Ridge_tol = Ridge_tol
        # PAC
        self.PAC_c = PAC_c
        self.PAC_fit = PAC_fit_intercept
        self.PAC_tol = PAC_tol
        self.PAC_loss = PAC_loss
        # Perceptron
        self.Per_penalty = Per_penalty
        self.Per_fit = Per_fit_intercept
        self.Per_alpha = Per_alpha
        self.Per_tol = Per_tol
        # MLP
        self.MLP_hidden = MLP_hidden_layer_sizes
        self.MLP_activate = MLP_activation
        self.MLP_maxiter = MLP_maxiter
        # RNC
        self.RNC_radius = radius
        self.RNC_weight = RNC_weights
        # NCC
        self.NCC_metric = NCC_metric
        self.NCC_shrink_thre = NCC_shrink_thre
        # EXtree
        self.EX_criterion = EX_criterion
        self.EX_splitter = EX_splitter
        self.EX_max_feature = EX_max_feature
        self.EX_min_leaf = EX_min_samples_leaf
        self.EX_min_split = EX_min_samples_split
        # adaBoost
        self.ada_n = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate
        # bagging
        self.bag_n = bag_n_estimators
        self.bag_max_feature = bag_max_features
        self.bag_max_samples = bag_max_samples
        # EXs
        self.EXs_criterion = EXs_criterion
        self.EXs_max_feature = EXs_max_feature
        self.EXs_min_leaf = EXs_min_samples_leaf
        self.EXs_min_split = EXs_min_samples_split
        self.EXs_n_estimator = EXs_n_estimator

        self.res = []

    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        if 'svmC' in params:
            self.svmC = params['svmC']
            self.SVCkernel = params['SVCkernel']['kernel']
            self.svmMaxiter = params['SVCkernel']['max_iter']

            if self.SVCkernel == 'rbf':
                self.gamma = params['SVCkernel']['rbfgamma']
            elif self.SVCkernel == 'sigmoid':
                self.gamma = params['SVCkernel']['siggamma']
                self.coef0 = params['SVCkernel']['sigcoef0']
            elif self.SVCkernel == 'poly':
                self.gamma = params['SVCkernel']['polygamma']
                self.degree = params['SVCkernel']['degree']
                self.coef0 = params['SVCkernel']['polycoef0']
            params.pop('SVCkernel')
            params.pop('svmC')

        if 'NBparam' in params:
            self.NBType = params['NBparam']['NBType']
            if self.NBType == 'multinomial':
                self.alpha = params['NBparam']['malpha']
            elif self.NBType == 'complement':
                self.alpha = params['NBparam']['calpha']
                self.norm = params['NBparam']['norm']
            params.pop('NBparam')

        if 'TDparam' in params:
            self.TD_strategy = params['TDparam']['TD_strategy']
            if self.TD_strategy == 'NN':
                self.TD_num = params['TDparam']['TD_num']
            params.pop('TDparam')

        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def run(self, Xsource, Ysource, Xtarget, Ytarget, loc, train=None, Ltrain=None):
        # initial the DAtor and Classifier
        if self.clfType == 'RF':
            self.m = RandomForestClassifier(n_estimators=self.RFn_estimators, criterion=self.RFcriterion,
                                            max_features=self.RFmax_features,
                                            min_samples_split=self.RFmin_samples_split,
                                            min_samples_leaf=self.RFmin_samples_leaf)

        if self.clfType == 'SVM':
            self.m = SVC(kernel=self.SVCkernel, C=self.svmC, degree=self.degree, coef0=self.coef0, gamma=self.gamma,
                         max_iter=10)

        if self.clfType == 'KNN':
            self.m = KNeighborsClassifier(n_neighbors=self.KNNneighbors, p=self.KNNp)

        if self.clfType == 'NB':
            if self.NBType == 'gaussian':
                self.m = GaussianNB()
            elif self.NBType == 'multinomial':
                self.m = MultinomialNB(alpha=self.alpha)
            elif self.NBType == 'complement':
                self.m = ComplementNB(alpha=self.alpha, norm=self.norm)

        if self.clfType == 'DT':
            self.m = DecisionTreeClassifier(criterion=self.DTcriterion, splitter=self.DTsplitter,
                                            max_features=self.DTmax_features,
                                            min_samples_split=self.DTmin_samples_split,
                                            min_samples_leaf=self.DTmin_samples_leaf)

        if self.clfType == 'LR':
            self.m = LogisticRegression(C=self.lrC, penalty=self.penalty, fit_intercept=self.fit_intercept,
                                        max_iter=self.maxiter)

        if self.clfType == 'Ridge':
            self.m = RidgeClassifier(alpha=self.Ridge_alpha, fit_intercept=self.Ridge_fit, tol=self.Ridge_tol)

        if self.clfType == 'PAC':
            self.m = PassiveAggressiveClassifier(C=self.PAC_c, fit_intercept=self.PAC_fit, tol=self.PAC_tol,
                                                 loss=self.PAC_loss)

        if self.clfType == 'Perceptron':
            self.m = Perceptron(penalty=self.Per_penalty, fit_intercept=self.Per_fit, tol=self.Per_tol,
                                alpha=self.Per_alpha)

        if self.clfType == 'MLP':
            self.m = MLPClassifier(hidden_layer_sizes=self.MLP_hidden, activation=self.MLP_activate,
                                   max_iter=self.MLP_maxiter)

        if self.clfType == 'RNC':
            self.m = RadiusNeighborsClassifier(radius=self.RNC_radius, weights=self.RNC_weight)

        if self.clfType == 'NCC':
            self.m = NearestCentroid(metric=self.NCC_metric, shrink_threshold=self.NCC_shrink_thre)

        if self.clfType == 'EXtree':
            self.m = ExtraTreeClassifier(criterion=self.EX_criterion, splitter=self.EX_splitter,
                                         max_features=self.EX_max_feature,
                                         min_samples_leaf=self.EX_min_leaf, min_samples_split=self.EX_min_split)

        if self.clfType == 'adaBoost':
            self.m = AdaBoostClassifier(n_estimators=self.ada_n, learning_rate=self.ada_learning_rate)

        if self.clfType == 'bagging':
            self.m = BaggingClassifier(n_estimators=self.bag_n, max_features=self.bag_max_feature,
                                       max_samples=self.bag_max_samples)

        if self.clfType == 'EXs':
            self.m = ExtraTreesClassifier(n_estimators=self.EXs_n_estimator, criterion=self.EXs_criterion,
                                          max_features=self.EXs_max_feature,
                                          min_samples_split=self.EXs_min_split, min_samples_leaf=self.EXs_min_leaf)

        if self.adpt == 'UM':
            self.DA = UM.Universal(pvalue=self.pvalue)

        if self.adpt == 'TCAplus':
            self.DA = TCAplus.TCA(kernel_type=self.kernelType, dim=self.dim, lamb=self.lamb, gamma=self.gamma)

        if self.adpt == 'NNfilter':
            self.DA = NNfilter.NNfilter(n_neighbors=self.NNn_neighbors, metric=self.NNmetric)

        if self.adpt == 'PCAmining':
            self.DA = PCAmining.PCAmining(dim=self.pcaDim)

        if self.adpt == 'GIS':
            model = GIS.GIS(model=self.m, model_name=self.clfType, mProb=self.mProb, mCount=self.mCount,
                            popsize=self.popsize,
                            chrmsize=self.chrmsize,
                            numgens=self.numgens, numparts=self.numparts)
            for i in range(self.repeat):
                self.res.append(model.run(Xsource=Xsource, Xtarget=Xtarget, Ysource=Ysource, Ytarget=Ytarget))
            return np.mean(np.asarray(self.res))

        if self.adpt == 'CDE_SMOTE':
            model = CDE_SMOTE.CDE_SMOTE(model=self.m, k=self.CDE_k, metric=self.CDE_metric)
            for i in range(self.repeat):
                self.res.append(model.run(Xs=Xsource, Xt=Xtarget, Ys=Ysource, Yt=Ytarget))
            return np.mean(np.asarray(self.res))

        if self.adpt == 'CLIFE':
            self.DA = CLIFE_MORPH.CLIFE_MORPH(n=self.Clife_n, alpha=self.Cliff_alpha, beta=self.Clife_beta,
                                              percentage=self.percentage)

        if self.adpt == 'FeSCH':
            self.DA = FeSCH.FeSCH(strategy=self.Fesch_strategy, nt=self.Fesch_nt)

        if self.adpt == 'FSS_bagging':
            model = FSS_bagging.FSS_bagging(model=self.m, FSS=self.FSS_ratio, topN=self.FSS_topn,
                                            score_thre=self.FSS_score_thre)
            for i in range(self.repeat):
                self.res.append(model.run(Xsource=Xsource, Xtarget=Xtarget, Ysource=Ysource, Ytarget=Ytarget, loc=loc))
            return np.mean(np.asarray(self.res))

        if self.adpt == 'HISNN':
            model = HISNN.HISNN(model=self.m, MinHam=self.HISNNminham)
            for i in range(self.repeat):
                self.res.append(model.run(Xs=Xsource, Xt=Xtarget, Ys=Ysource, Yt=Ytarget))
            return np.mean(np.asarray(self.res))

        if self.adpt == 'MCWs':
            model = multiple_components_weights.MCWs(model=self.m, k=self.MCW_k, lamb=self.MCW_lamb,
                                                     sigmma=self.MCW_sigmma)
            for i in range(self.repeat):
                self.res.append(
                    model.run(Xs=Xsource, Ys=Ysource, test=Xtarget, l_test=Ytarget, train=train, l_train=Ltrain,
                              loc=loc))
            return np.mean(np.asarray(self.res))

        if self.adpt == 'TD':
            self.DA = Training_Data_Selection.TDS(strategy=self.TD_strategy, expected_num=self.TD_num)

        if self.adpt == 'VCB':
            model = Value_Cognitive_Boosting.VCB(model=self.m, M=self.VCB_M, lamb=self.VCB_lamb)
            for i in range(self.repeat):
                self.res.append(model.run(Xs=Xsource, Xt=Xtarget, Ys=Ysource, Yt=Ytarget))
            return np.mean(np.asarray(self.res))

        for i in range(self.repeat):
            if self.adpt in ['UM', 'TD', 'FSS_bagging']:
                Xsource, Ysource, Xtarget, Ytarget = self.DA.run(Xsource, Ysource, Xtarget, Ytarget, loc)
            else:
                if self.adpt == 'NNfilter':
                    Xsource, Ysource, train, ltrain = self.DA.run(Xsource, Ysource, train, Ltrain)
                    Xtarget = np.log(Xtarget + 1)
                else:
                    Xsource, Ysource, Xtarget, Ytarget = self.DA.run(Xsource, Ysource, Xtarget, Ytarget)

            if np.asarray(Xsource).ndim <= 1 or len(np.unique(Ysource)) <= 1:
                self.res.append(0)
            else:
                try:
                    self.m.fit(Xsource, Ysource)
                    predict = self.m.predict(Xtarget)
                    self.res.append(roc_auc_score(Ytarget, predict))
                except:
                    self.res.append(0)

        return np.mean(np.asarray(self.res))
