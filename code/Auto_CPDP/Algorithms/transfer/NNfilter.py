import numpy as np
from sklearn.neighbors import NearestNeighbors


class NNfilter(object):
    def __init__(self, n_neighbors=10, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def run(self, Xsource, Ysource, Xtarget, Ytarget):
        # Log transformation
        Xsource = np.log(Xsource + 1)
        Xtarget = np.log(Xtarget + 1)

        if self.n_neighbors > Xsource.shape[0]:
            return 0, 0, 0, 0

        knn = NearestNeighbors(metric=self.metric)
        knn.fit(Xsource)
        data = []
        ysel = []

        # find neighbors
        for item in Xtarget:
            tmp = knn.kneighbors(item.reshape(1, -1), self.n_neighbors, return_distance=False)
            tmp = tmp[0]
            for i in tmp:
                if list(Xsource[i]) not in data:
                    data.append(list(Xsource[i]))
                    ysel.append(Ysource[i])
        Xsource, idx = np.unique(np.asanyarray(data), axis=0, return_index=True)
        Ysource = np.asanyarray(ysel)
        Ysource = Ysource[idx]

        return Xsource, Ysource, Xtarget, Ytarget
