import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


class HISNN(object):
    def __init__(self, model, MinHam=1.0):
        self.MinHam = MinHam
        self.model = model

    def _MahalanobisDist(self, data, base):

        covariance = np.cov(base.T)  # calculate the covarince matrix
        inv_covariance = np.linalg.pinv(covariance)
        mean = np.mean(base, axis=0)
        dist = np.zeros((np.asarray(data)).shape[0])
        for i in range(dist.shape[0]):
            dist[i] = distance.mahalanobis(data[i], mean, inv_covariance)
        return dist

    def _TrainInstanceFiltering(self):
        # source outlier remove based on source
        dist = self._MahalanobisDist(self.Xsource, self.Xsource)
        threshold = np.mean(dist) * 3 * np.std(dist)
        outliers = []
        for i in range(len(dist)):
            if dist[i] > threshold:
                outliers.append(i)  # index of the outlier
        self.Xsource = np.delete(self.Xsource, outliers, axis=0)
        self.Ysource = np.delete(self.Ysource, outliers, axis=0)

        # source outlier remove based on target
        dist = self._MahalanobisDist(self.Xsource, self.Xtarget)
        threshold = np.mean(dist) * 3 * np.std(dist)
        outliers = []
        for i in range(len(dist)):
            if dist[i] > threshold:
                outliers.append(i)  # index of the outlier
        self.Xsource = np.delete(self.Xsource, outliers, axis=0)
        self.Ysource = np.delete(self.Ysource, outliers, axis=0)


        # NN filter for source data based on target
        neigh = NearestNeighbors(radius=self.MinHam, metric='hamming')
        neigh.fit(self.Xsource)
        res = neigh.radius_neighbors(self.Xtarget, return_distance=False)

        tmp = np.concatenate((self.Xsource, self.Ysource.reshape(-1, 1)), axis=1)
        x = tmp[res[0]]
        for item in res[1:]:
            x = np.concatenate((x, tmp[item]), axis=0)
            x = np.unique(x, axis=0)

        self.Xsource = x[:, :-1]
        self.Ysource = x[:, -1]

    def predict(self):
        predict = np.zeros(self.Xtarget.shape[0])
        neigh = NearestNeighbors(radius=self.MinHam, metric='hamming')
        neigh.fit(self.Xsource)
        res = neigh.radius_neighbors(self.Xtarget, return_distance=False)
        for i in range(res.shape[0]):
            # case 1
            if len(res[i]) == 1:
                subRes = neigh.radius_neighbors(self.Xsource[res[i][0]])
                # case 1-1
                if len(subRes) == 1:
                    predict[i] = self.model.predict(self.Xtarget[i].reshape(1, -1))
                else:
                    tmp = np.unique(self.Ysource[subRes])
                    # case 1-2
                    if len(tmp) == 1:
                        predict[i] = tmp[0]
                    # case 1-3
                    else:
                        predict[i] = self.model.predict(self.Xtarget[i].reshape(1, -1))
            else:
                tmp = np.unique(self.Ysource[res[i]])
                # case 2
                if len(tmp) == 1:
                    predict[i] = tmp[0]
                # case 3
                else:
                    predict[i] = self.model.predict(self.Xtarget[i].reshape(1, -1))

        self.AUC = roc_auc_score(self.Ytarget, predict)

    def run(self, Xs, Ys, Xt, Yt):
        self.Xsource = np.asarray(Xs)
        self.Ysource = np.asarray(Ys)
        self.Xtarget = np.asarray(Xt)
        self.Ytarget = np.asarray(Yt)

        self._TrainInstanceFiltering()
        if self.Xsource.shape[0] <= 1:
            return 0
        self.model.fit(np.log(self.Xsource + 1), self.Ysource)
        self.predict()
        return self.AUC
