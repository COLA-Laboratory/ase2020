import numpy as np
from sklearn.decomposition import PCA


class PCAmining():
    def __init__(self, dim=5):
        self.dim = dim

    def run(self, Xs, Ys, Xt, Yt):
        self.Xsource = np.asarray(Xs)
        self.Xtarget = np.asarray(Xt)
        self.Ysource = np.asarray(Ys)
        self.Ytarget = np.asarray(Yt)

        pca = PCA(n_components=self.dim)
        pca.fit(self.Xsource)
        self.Xsource = pca.transform(self.Xsource)
        self.Xtarget = pca.transform(self.Xtarget)

        return self.Xsource, self.Ysource, self.Xtarget, self.Ytarget
