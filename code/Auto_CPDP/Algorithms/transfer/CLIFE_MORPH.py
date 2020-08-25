import numpy as np
from sklearn.neighbors import NearestNeighbors
class CLIFE_MORPH():
    def __init__(self, n=10, alpha=0.15, beta=0.35, percentage=0.8):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.percentage = percentage

    def _UniqueRanges(self, x):
        idx = np.argsort(x)
        subrange = []
        c = idx.shape[0] % self.n
        if c != 0:
            b = np.split(idx[:idx.shape[0] - c], self.n)
            b[-1] = np.concatenate((b[-1], idx[idx.shape[0] - c:]))
        else:
            b = np.split(idx, self.n)
        for i in range(len(b)):
            begin = x[b[i][0]]
            end = x[b[i][-1]]
            subrange.append( [begin, end] )

        return subrange

    def _compute_power(self, subrange, idx):
        PR = np.zeros((2, len(subrange)))
        D = self.Xsource.shape[0]
        for i in range(2):
            first = self.Xsource[self.Ysource==i][:, idx]
            rest = self.Xsource[self.Ysource!=i][:, idx]
            p_first = len(first) / D
            p_rest = len(rest) / D
            for j in range(len(subrange)):
                low = subrange[j][0]
                high = subrange[j][1]
                if low == high:
                    like_first = len(first) * p_first
                    like_rest = len(rest) * p_rest
                else:
                    if j == len(subrange) - 1:
                        like_first = len(first[np.where((first >= low) & (first <= high))]) * p_first
                        like_rest = len(rest[np.where((rest >= low) & (rest <= high))]) * p_rest
                    else:
                        like_first = len(first[np.where((first >= low) & (first < high))]) * p_first
                        like_rest = len(rest[np.where((rest >= low) & (rest < high))]) * p_rest

                PR[i][j] = like_first**2 / (like_first + like_rest)
        return PR

    def _cliff_selection(self, M, subrange):
        res = []
        num = int(self.Xsource.shape[0] * (self.percentage))
        num_0 = int(num * self.Ysource[self.Ysource == 0].shape[0] / self.Ysource.shape[0])
        i = 0
        while i < num_0:
            for j in range(self.Xsource.shape[1]):
                # label
                row = 0
                # subrange
                column = np.argmax(M[j][0])
                idx_candidate = np.where(self.Ysource==row)[0]
                candidate = self.Xsource[idx_candidate][:, j]
                idx_tmp = np.where((candidate >= subrange[j][column][0]) &
                                   (candidate <= subrange[j][column][1]))
                idx_add = idx_candidate[idx_tmp]
                res = list( set(res + list(idx_add)))
                if len(res) > num_0:
                    res = res[:int(-len(res) + num_0)]
                    i = num_0
                else:
                    i = len(res)

        while i < num:
            for j in range(self.Xsource.shape[1]):
                # label
                row = 1
                # subrange
                column = np.argmax(M[j][1])
                idx_candidate = np.where(self.Ysource==row)[0]
                candidate = self.Xsource[idx_candidate][:, j]
                idx_tmp = np.where((candidate >= subrange[j][column][0]) &
                                   (candidate <= subrange[j][column][1]))
                idx_add = idx_candidate[idx_tmp]
                res = list( set(res + list(idx_add)))
                if len(res) > num:
                    res = res[:int(-len(res) + num)]
                    i = num
                else:
                    i = len(res)

        self.Xsource = self.Xsource[np.asarray(res).astype(int)]
        self.Ysource = self.Ysource[np.asarray(res).astype(int)]


    def _CLIFE(self):
        PR = np.zeros((self.Xsource.shape[1], 2, self.n))
        sub = []
        for i in range(self.Xsource.shape[1]):
            subrange = self._UniqueRanges(self.Xsource[:, i])
            sub.append(subrange)
            PR[i] = self._compute_power(subrange, i)
        self._cliff_selection(PR, sub)

    def _MORPH(self):
        set1 = self.Xsource[self.Ysource == 1]
        set2 = self.Xsource[self.Ysource != 1]
        knn1 = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn2 = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn1.fit(set2)
        knn2.fit(set1)
        for i in range(self.Xsource.shape[0]):
            if self.Ysource[i] == 1:
                NUN = set2[knn1.kneighbors(self.Xsource[i].reshape(1, -1), return_distance=False)]
            else:
                NUN = set1[knn2.kneighbors(self.Xsource[i].reshape(1, -1), return_distance=False)]

            rnd = np.random.uniform(0.15, 0.35, 1)
            opt_rnd = np.random.randint(1, 3, 1)
            self.Xsource[i] = self.Xsource[i] + np.power(-1, opt_rnd) * (self.Xsource[i] - NUN) * rnd

    def run(self, Xs, Ys, Xt, Yt):
        self.Xsource = np.asarray(Xs)
        self.Xtarget = np.asarray(Xt)
        self.Ysource = np.asarray(Ys)
        self.Ytarget = np.asarray(Yt)

        self._CLIFE()
        self._MORPH()
        return self.Xsource, self.Ysource, self.Xtarget, self.Ytarget

