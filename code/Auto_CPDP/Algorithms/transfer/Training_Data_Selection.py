import numpy as np
from sklearn.mixture import GaussianMixture


class TDS():
    def __init__(self, strategy='NN', expected_num=3):
        self.strategy = strategy
        self.expected_num = expected_num

    def _get_distributional_characteristics_vector(self, Xs, Ys, Xt, Yt, loc):
        self.X = dict()
        self.Y = dict()
        Vchracteristics = dict()
        for i in range(len(loc)):
            if i < len(loc) - 1:
                x = Xs[loc[i]:loc[i + 1]]
                y = Ys[loc[i]:loc[i + 1]]
            else:
                x = Xs[loc[i]:]
                y = Ys[loc[i]:]
            m = x.mean(axis=0)
            std = x.std(axis=0)
            Vchracteristics[i] = np.concatenate((m, std))
            self.X[i] = x
            self.Y[i] = y

        m = Xt.mean(axis=0)
        std = Xt.mean(axis=0)
        return Vchracteristics, np.concatenate((m, std))

    def _EM_clustering(self, source, target):
        x = np.asarray(list(source.values()))
        x = np.concatenate((x, [target]), axis=0)

        gmm = GaussianMixture(n_components=2, covariance_type='full')
        gmm.fit(x)
        label_t = gmm.predict(target.reshape(1, -1))
        xs = self.X[0]
        ys = self.Y[0]

        for k, v in source.items():
            label = gmm.predict(v.reshape(1, -1))
            if label == label_t:
                xs = np.concatenate((xs, self.X[k]), axis=0)
                ys = np.concatenate((ys, self.Y[k]), axis=0)

        xs = xs[self.X[0].shape[0]:, :]
        ys = ys[self.Y[0].shape[0]:]

        return xs, ys

    def _Nearest_Neighbor_Selection(self, source, target):
        dist = dict()
        for k, v in source.items():
            dist[k] = np.linalg.norm(v - target)

        dist = sorted(dist.items(), key=lambda x: x[1])
        xs = self.X[dist[0][0]]
        ys = self.Y[dist[0][0]]

        for i in range(1, self.expected_num):
            xs = np.concatenate((xs, self.X[dist[i][0]]))
            ys = np.concatenate((ys, self.Y[dist[i][0]]))

        return xs, ys

    def run(self, Xs, Ys, Xt, Yt, loc):
        self.expected_num = min(len(loc), self.expected_num)
        source, target = self._get_distributional_characteristics_vector(Xs, Ys, Xt, Yt, loc)

        if self.strategy == 'EM':
            xs, ys = self._EM_clustering(source, target)
            return xs, ys, Xt, Yt

        elif self.strategy == 'NN':
            xs, ys = self._Nearest_Neighbor_Selection(source, target)
            return xs, ys, Xt, Yt
