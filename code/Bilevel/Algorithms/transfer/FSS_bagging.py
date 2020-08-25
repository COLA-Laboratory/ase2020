import numpy as np
import random, copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


class FSS_bagging(object):
    def __init__(self, model, topN=10, FSS=0.1, score_thre=0.5):
        self.topN = topN
        self.FSS = FSS
        self.score_thre = score_thre
        self.model = model

    def _sample(self, Xsource, Xtarget):
        K = min(500, Xsource.shape[0], Xtarget.shape[0])
        Ltrain = np.ones(K)
        Ltest = np.ones(K) * -1

        Train = random.sample(range(Xsource.shape[0]), Xsource.shape[0] - K)
        Test = random.sample(range(Xtarget.shape[0]), Xtarget.shape[0] - K)
        Train = np.delete(Xsource, Train, axis=0)
        Test = np.delete(Xtarget, Test, axis=0)

        data = np.concatenate((Train, Test), axis=0)
        label = np.concatenate((Ltrain, Ltest), axis=0)

        return data, label

    def _calDistance(self, Xsource, Xtarget):
        acc = np.zeros(10)
        for i in range(10):
            x, y = self._sample(Xsource, Xtarget)
            try:
                acc[i] = np.mean(cross_val_score(self.model, x, y, scoring='accuracy', cv=5))
            except:
                acc[i] = 0
        return 2 * abs((np.mean(acc) - 0.5))

    def _remove_unstable_feature(self, x, xtarget):
        fx, fy = self._sample(x, xtarget)
        lr = LogisticRegression()
        lr.fit(fx, fy)
        coef = dict()
        for i in range(x.shape[1]):
            coef[i] = lr.coef_[0][i]
        coef = sorted(coef.items(), key=lambda d: d[1], reverse=True)

        dump = []
        for i in range(int(x.shape[1] * self.FSS)):
            dump.append(coef[i][0])

        x = np.delete(x, dump, axis=1)
        xtarget = np.delete(xtarget, dump, axis=1)
        return x, xtarget

    def run(self, Xsource, Ysource, Xtarget, Ytarget, loc):
        prediction = dict()
        self.topN = min(self.topN, len(loc))
        dist = dict()

        for i in range(len(loc)):
            if i < len(loc) - 1:
                train = Xsource[loc[i]:loc[i + 1]]
                dist[i] = self._calDistance(train, Xtarget)
            else:
                train = Xsource[loc[i]:]
                dist[i] = self._calDistance(train, Xtarget)

        dist = sorted(dist.items(), key=lambda d: d[1])

        for i in range(self.topN):
            xt = copy.deepcopy(Xtarget)
            index = dist[i][0]
            if index < len(loc) - 1:
                tmp = Xsource[loc[index]:loc[index + 1]]
                temp = Ysource[loc[index]:loc[index + 1]]
            else:
                tmp = Xsource[loc[index]:]
                temp = Ysource[loc[index]:]
            tmp, xt = self._remove_unstable_feature(tmp, xt)
            try:
                self.model.fit(tmp, temp)
                prediction[i] = self.model.predict(xt)
            except:
                prediction[i] = np.random.randint(0, 2, xt.shape[0])

        res = np.asarray(list(prediction.values()))
        res = res.mean(axis=0)
        res[res > self.score_thre] = 1
        res[res <= self.score_thre] = 0
        return roc_auc_score(Ytarget, res)
