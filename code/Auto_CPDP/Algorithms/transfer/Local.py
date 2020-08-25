from Utils.fastmap.python_fastmap import fastmap
import numpy as np


class Local():
    def __init__(self, minInstance=1):
        self.minInstance = minInstance

    def _distance(self, x, y):
        c = np.max(x)


    def _WHERE(self):
        # reduce dimensionality (fastmap)
        self.Xsource = fastmap(self.Xsource, 2)
        self.Xtarget = fastmap(self.Xtarget, 2)
        # clustering


    def run(self, Xs, Ys, Xt, Yt):
        self.Xsource = np.asarray(Xs)
        self.Xtarget = np.asarray(Xt)
        self.Ysource = np.asarray(Ys)
        self.Ytarget = np.asarray(Yt)

        if self.minInstance == 1:
            self.minInstance = np.sqrt(self.Xsource.shape[0] + self.Xtarget[0])




        return