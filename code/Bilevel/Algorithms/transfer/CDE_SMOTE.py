from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import numpy as np


# the efficiency of over-sampling should be optimized!

class CDE_SMOTE():
    def __init__(self, model, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.model = model
        self.flag = 0

    def _over_sampling(self, x, idx, num):
        x_over = x[idx]
        knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        knn.fit(x_over)
        neighbors = knn.kneighbors(x_over, return_distance=False)
        if x_over.shape[0] > num:
            idx = np.random.choice(x_over.shape[0], num, replace=False)
        else:
            idx = np.random.choice(x_over.shape[0], num, replace=True)
        for i in idx:
            i = int(i)
            rnd = int(neighbors[i][int(np.random.choice(self.k, 1))])
            xnew = x_over[i] + np.random.random() * (x_over[i] - x[rnd])
            x = np.concatenate((x, xnew.reshape(1, -1)), axis=0)
        return x

    def _class_distribution_estimation(self):
        m = np.bincount(self.Ysource)
        x = self._over_sampling(self.Xsource, np.where(self.Ysource == 1)[0], m[0] - m[1])
        y = np.concatenate((self.Ysource, np.ones(m[0] - m[1])), axis=0)
        self.model.fit(x, y)
        prediction = self.model.predict(self.Xtarget).astype(np.int)
        return np.bincount(prediction)

    def _class_distribution_modification(self, n):
        m = np.bincount(self.Ysource)
        num = int(m[0] * n[0] / n[1]) - m[1]
        self.Xsource = self._over_sampling(self.Xsource, np.where(self.Ysource == 1)[0], num)
        self.Ysource = np.concatenate((self.Ysource, np.ones(num)), axis=0)
        self.model.fit(self.Xsource, self.Ysource)

    def run(self, Xs, Ys, Xt, Yt):
        self.Xsource = np.asarray(Xs)
        self.Xtarget = np.asarray(Xt)
        self.Ysource = np.asarray(Ys).astype(np.int)
        self.Ytarget = np.asarray(Yt)

        try:
            n = self._class_distribution_estimation()
            self._class_distribution_modification(n)
            prediction = self.model.predict(self.Xtarget)
            return roc_auc_score(self.Ytarget, prediction)
        except:
            return 0


