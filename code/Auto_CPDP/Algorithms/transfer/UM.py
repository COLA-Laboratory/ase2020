import Utils.cliffsDelta as cliff
from scipy.stats import mannwhitneyu
import numpy as np
import copy


# No context factor
class Universal(object):
    def __init__(self, pvalue=0.05):
        self.p = pvalue

    def _compareMetricDistribution(self, x1, x2):
        s, p = mannwhitneyu(x1, x2)
        if p < self.p:
            sig_diff = 1
        else:
            sig_diff = 0
        return sig_diff

    def _quantifyDifference(self, x1, x2):
        d, res = cliff.cliffsDelta(x1, x2)
        return res

    def cluster(self, No_metric, numGroup, group):
        indexOfCluster = 0
        clusterOfGroup = np.zeros(numGroup)

        for i in range(0, numGroup - 1):
            indexNewCluster = indexOfCluster + 1
            for j in range(i + 1, numGroup):
                if self._compareMetricDistribution(group[i][:, No_metric], group[j][:, No_metric]) == 1:
                    if self._quantifyDifference(group[i][:, No_metric], group[j][:, No_metric]) == 'large':
                        indexOfCluster = indexNewCluster
                        clusterOfGroup[j] = indexOfCluster

        return clusterOfGroup

    def rankTransform(self, xsource, xtarget, loc):
        group = []
        for i in range(len(loc)):
            if i < len(loc) - 1:
                train = xsource[loc[i]:loc[i + 1]]
            else:
                train = xsource[loc[i]:]
            group.append(train)
        group.append(xtarget)
        resGroup = copy.deepcopy(group)

        for i in range(xsource.shape[1]):
            clusterIndex = self.cluster(i, len(loc) + 1, group)
            cluster = np.unique(clusterIndex)
            for item in cluster:
                tmp = np.asarray(np.where(clusterIndex == item))[0]
                tmp_data = np.asarray([])
                for ncs in tmp:
                    tmp_data = np.concatenate((tmp_data, group[int(ncs)][:, i]))

                percentiles = np.percentile(sorted(tmp_data), [10, 20, 30, 40, 50, 60, 70, 80, 90])
                for ncs in tmp:
                    ncs = int(ncs)
                    t = resGroup[ncs][:, i]
                    for it in range(len(t)):
                        if t[it] <= percentiles[0]:
                            resGroup[ncs][:, i][it] = 1
                        elif t[it] <= percentiles[1]:
                            resGroup[ncs][:, i][it] = 2
                        elif t[it] <= percentiles[2]:
                            resGroup[ncs][:, i][it] = 3
                        elif t[it] <= percentiles[3]:
                            resGroup[ncs][:, i][it] = 4
                        elif t[it] <= percentiles[4]:
                            resGroup[ncs][:, i][it] = 5
                        elif t[it] <= percentiles[5]:
                            resGroup[ncs][:, i][it] = 6
                        elif t[it] <= percentiles[6]:
                            resGroup[ncs][:, i][it] = 7
                        elif t[it] <= percentiles[7]:
                            resGroup[ncs][:, i][it] = 8
                        elif t[it] <= percentiles[8]:
                            resGroup[ncs][:, i][it] = 9
                        else:
                            resGroup[ncs][:, i][it] = 10

        return resGroup

    def run(self, Xsource, Ysource, Xtarget, Ytarget, loc):
        res = self.rankTransform(Xsource, Xtarget, loc)
        source = np.asarray(res[0])
        for i in range(1, len(loc)):
            source = np.concatenate((source, res[i]), axis=0)
        target = res[-1]

        return source, Ysource, target, Ytarget
