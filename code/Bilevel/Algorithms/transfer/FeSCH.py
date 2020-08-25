import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from minepy import MINE
from info_gain import info_gain
import powerlaw, copy

class FeSCH():
    def __init__(self, nt=1, strategy='SFD'):
        self.nt = nt
        self.strategy = strategy

    def _feature_clustering(self):
        x = np.concatenate((self.Xs, self.Xt), axis=0)
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(x.T)
        distance, idx = knn.kneighbors(x.T)
        dc = np.min(distance)
        dist = cdist(x.T, x.T, 'euclidean')
        ro = np.sum(dist - dc, axis=1) + dc
        self.ro = copy.deepcopy(ro)

        sigma = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            tmp = dist[i]
            if ro[i] == np.max(ro):
                sigma[i] = np.max(tmp)
            else:
                sigma[i] = np.min(tmp[ro > ro[i]])


        ro = ( ro - np.mean(ro) ) / np.std(ro)
        sigma = (sigma - np.mean(sigma)) / np.std(sigma)
        gamma = ro * sigma

        gamma_sort = sorted(gamma, reverse=True)
        res = powerlaw.find_xmin(gamma_sort)
        centers = np.where(gamma>res[0])[0]
        if len(centers) == x.shape[1]:
            return [centers]
        else:
            res = []
            tmp = x.T[centers]
            tmp1 = np.delete(x.T, centers, axis=0)
            idx = np.delete(range(x.shape[1]), centers)
            knn.fit(tmp)
            nn = knn.kneighbors(tmp1, n_neighbors=1, return_distance=False)


            for i in centers:
                res.append([i])
            for i in range(nn.shape[0]):
                res[centers[nn[i][0]]].append(idx[i])
            return res

    def _feature_selection(self):
        cluster = self._feature_clustering()

        res = []
        for item in cluster:
            num = int(np.ceil(len(item)*self.nt / self.Xs.shape[1]))

            if self.strategy == 'LDF':

                tmp = np.argsort(self.ro[item])[-num:]
                tmp = np.asarray(item)[tmp]
                res = np.concatenate((res, tmp), axis=0)

            if self.strategy == 'SFD':
                mine = MINE()
                score = []
                length = min(len(self.Xs.T[0]), len(self.Xt.T[0]))
                for it in item:
                    mine.compute_score(self.Xs.T[it][:length], self.Xt.T[it][:length])
                    tmp = mine.mic()
                    score.append(tmp)
                res = np.concatenate((res, np.asarray(item)[np.argsort(score)[-num:]]), axis=0)

            if self.strategy == 'FCR':
                score = []
                for it in item:
                    tmp = info_gain.info_gain(list(self.Xs.T[it]), list(self.Ys))
                    score.append(tmp)
                res = np.concatenate((res, np.asarray(item)[np.argsort(score)[-num:]]), axis=0)
        return res

    def run(self, Xs, Ys, Xt, Yt):
        self.Xs = np.asarray(Xs)
        self.Ys = np.asarray(Ys)
        self.Xt = np.asarray(Xt)
        self.Yt = np.asarray(Yt)

        idx = self._feature_selection().astype(np.int)
        return self.Xs.T[idx].T, self.Ys, self.Xt.T[idx].T, self.Yt


