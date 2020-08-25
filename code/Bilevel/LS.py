import numpy as np
import random, copy, os, time
from func_timeout import func_set_timeout
from multiprocessing import Pool, Manager

# time budget
time_budget = 3600


# parallel (fully LS)
class paraTabu():
    def __init__(self, f, range, dir, stime, max=25, popsize=5):
        self.popsize = popsize
        self.max = max
        self.upper = []
        self.lower = []
        for k, v in range.items():
            self.lower.append(v[0])
            self.upper.append(v[1])

        self.objFunc = f
        self.dir = dir
        # tabuList aka a history log
        self.tabuList = Manager().dict()
        self.processList = Manager().list()
        self.stime = stime

    def tabuSearch(self, tabuList, processList):
        # In order to release resources correctly, we record the ID of process.
        processList.append(os.getpid())
        x = np.zeros(len(self.lower))
        archive = dict()
        count = 0

        """ initialization """
        for i in range(len(self.lower)):
            x[i] = int(random.randint(self.lower[i], self.upper[i]))
        archive[str(x)] = []
        res = self.objFunc(x)
        archive[str(x)] = res

        count += 1

        """ main loop """
        while 1:
            tabuList[str(x)] = [x, res]
            """ termination """

            if count >= self.max:
                print('stop!!!')
                break

            """ find neighbors """
            neighbor = dict()
            for i in range(len(self.lower)):
                if x[i] > self.lower[i]:
                    c = 1
                    while 1:
                        tmp = copy.deepcopy(x)
                        tmp[i] -= c
                        if tmp[i] < self.lower[i]:
                            break
                        if str(tmp) not in tabuList.keys() and str(tmp) not in neighbor.keys() and str(x) != str(tmp):
                            neighbor[str(tmp)] = tmp
                            break
                        else:
                            c += 1
                            continue

                if x[i] < self.upper[i]:
                    c = 1
                    while 1:
                        tmp = copy.deepcopy(x)
                        tmp[i] += c
                        if tmp[i] > self.upper[i]:
                            break
                        if str(tmp) not in tabuList.keys() and str(tmp) not in neighbor.keys() and str(x) != str(tmp):
                            neighbor[str(tmp)] = tmp
                            break
                        else:
                            c += 1
                            continue

            """ choose best from neighbors """
            f = []
            tmp_res = []
            keys = []
            if len(neighbor) != 0:
                for k, item in neighbor.items():
                    if str(item) not in tabuList.keys():
                        tmp = self.objFunc(item)
                        tabuList[str(item)] = [item, tmp]
                        archive[str(item)] = tmp
                        tmp_res.append(tmp)

                        f.append(tmp[0])
                        keys.append(k)

                        if count < self.max:
                            count += 1
                        else:
                            break
                    else:
                        continue

                if res[0] > np.min(f):
                    index = int(np.argmin(f))
                    x = neighbor[keys[index]]
                    res = tmp_res[index]

    @func_set_timeout(time_budget)
    def run(self):
        p = Pool(self.popsize)
        for i in range(self.popsize):
            p.apply_async(self.tabuSearch,
                          args=(self.tabuList, self.processList))
        p.close()
        p.join()

        with open(self.dir, 'a+') as f:
            for k, v in self.tabuList.items():
                print(v, file=f)
            print('time:', time.time() - self.stime, file=f)

        res = np.asarray(list(self.tabuList.values()))

        return res[ int(np.argmin(res[:, -1][:, 0])) ]


class paraTabu_multi_start():
    def __init__(self, f, range, dir, stime, max=25, popsize=5):
        self.popsize = popsize
        self.max = max
        self.upper = []
        self.lower = []
        for k, v in range.items():
            self.lower.append(v[0])
            self.upper.append(v[1])

        self.objFunc = f
        self.dir = dir
        # tabuList aka a history log
        self.tabuList = Manager().dict()
        self.processList = Manager().list()
        self.stime = stime

    def tabuSearch(self, tabuList, processList, sn):
        # In order to release resources correctly, we record the ID of process.
        processList.append(os.getpid())
        x = np.zeros(len(self.lower))
        archive = dict()
        count = 0

        """ initialization """
        for i in range(len(self.lower)):
            each = (self.upper[i] - self.lower[i])/self.popsize
            hi = self.lower[i] + (sn+1)*each
            lo = self.lower[i] + sn * each
            x[i] = int(random.randint(int(lo), np.ceil(hi)))
        archive[str(x)] = []
        res = self.objFunc(x)
        archive[str(x)] = res

        count += 1

        """ main loop """
        while 1:
            tabuList[str(x)] = [x, res]
            """ termination """

            if count >= self.max:
                print('stop!!!')
                break

            """ find neighbors """
            neighbor = dict()
            for i in range(len(self.lower)):
                if x[i] > self.lower[i]:
                    c = 1
                    while 1:
                        tmp = copy.deepcopy(x)
                        tmp[i] -= c
                        if tmp[i] < self.lower[i]:
                            break
                        if str(tmp) not in tabuList.keys() and str(tmp) not in neighbor.keys() and str(x) != str(tmp):
                            neighbor[str(tmp)] = tmp
                            break
                        else:
                            c += 1
                            continue

                if x[i] < self.upper[i]:
                    c = 1
                    while 1:
                        tmp = copy.deepcopy(x)
                        tmp[i] += c
                        if tmp[i] > self.upper[i]:
                            break
                        if str(tmp) not in tabuList.keys() and str(tmp) not in neighbor.keys() and str(x) != str(tmp):
                            neighbor[str(tmp)] = tmp
                            break
                        else:
                            c += 1
                            continue

            """ choose best from neighbors """
            f = []
            tmp_res = []
            keys = []
            if len(neighbor) != 0:
                for k, item in neighbor.items():
                    if str(item) not in tabuList.keys():
                        tmp = self.objFunc(item)
                        tabuList[str(item)] = [item, tmp]
                        archive[str(item)] = tmp
                        tmp_res.append(tmp)

                        f.append(tmp[0])
                        keys.append(k)

                        if count < self.max:
                            count += 1
                        else:
                            break
                    else:
                        continue

                if res[0] > np.min(f):
                    index = int(np.argmin(f))
                    x = neighbor[keys[index]]
                    res = tmp_res[index]

    @func_set_timeout(time_budget)
    def run(self):
        p = Pool(self.popsize)
        for i in range(self.popsize):
            p.apply_async(self.tabuSearch,
                          args=(self.tabuList, self.processList, i))
        p.close()
        p.join()

        with open(self.dir, 'a+') as f:
            for k, v in self.tabuList.items():
                print(v, file=f)
            print('time:', time.time() - self.stime, file=f)

        res = np.asarray(list(self.tabuList.values()))

        return res[ int(np.argmin(res[:, -1][:, 0])) ]

