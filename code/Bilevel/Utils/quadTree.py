import numpy as np


class QuadTree():
    # global size of quad tree
    size = 0
    # recursion parameter alpha
    alpha = 0
    # data for each cluster
    ccluster = []
    # cluster sizes (index is cluster number, {@link ArrayList} is list of boxes (x0,y0,x1,y1))
    csize = []

    def __init__(self, parent, payload):
        self.parent = parent
        self.child_nw = None
        self.child_ne = None
        self.child_se = None
        self.child_sw = None

        self.l = []
        # size of the quadrant in x,y dimension
        self.x = None
        self.y = None
        # data within this quadrant (list of (xi, yi))
        self.payload = payload

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def getDensity(self):
        return self.payload.shape[0] / QuadTree.size

    def getSizeDouble(self):
        return self.x, self.y

    def split(self):
        medx = np.median(self.payload[:, 0])
        medy = np.median(self.payload[:, 1])

        nw = []
        sw = []
        ne = []
        se = []

        for i in range(self.payload.shape[0]):
            item = self.payload[i]

            if item[0] <= medx and item[1] >= medy:
                nw.append(item)
            if item[0] <= medx and item[1] <= medy:
                sw.append(item)
            if item[0] >= medx and item[1] >= medy:
                ne.append(item)
            if item[0] >= medx and item[1] <= medy:
                se.append(item)

        try:
            if len(nw) == self.payload.shape[0]:
                raise Exception("payload equal")
            if len(sw) == self.payload.shape[0]:
                raise Exception("payload equal")
            if len(ne) == self.payload.shape[0]:
                raise Exception("payload equal")
            if len(se) == self.payload.shape[0]:
                raise Exception("payload equal")
        except:
            print("payload equal")

        self.child_nw = QuadTree(self, np.asarray(nw))
        self.child_nw.x = self.x[0], medx
        self.child_nw.y = medy, self.y[1]

        self.child_sw = QuadTree(self, np.asarray(sw))
        self.child_sw.x = self.x[0], medx
        self.child_sw.y = self.y[0], medy

        self.child_ne = QuadTree(self, np.asarray(ne))
        self.child_ne.x = medx, self.x[1]
        self.child_ne.y = medy, self.y[1]

        self.child_se = QuadTree(self, np.asarray(se))
        self.child_se.x = medx, self.x[1]
        self.child_se.y = self.y[0], medy
        self.payload = None

        return self.child_nw, self.child_ne, self.child_se, self.child_sw

    @staticmethod
    def recursiveSplit(q):
        """ :param q: QuadTree"""
        if q.payload.shape[0] < QuadTree.alpha:
            return
        try:
            child = q.split()
            QuadTree.recursiveSplit(child[0])
            QuadTree.recursiveSplit(child[1])
            QuadTree.recursiveSplit(child[2])
            QuadTree.recursiveSplit(child[3])
        except:
            return

    def _generateList(self, q):
        if q.child_ne is None:
            self.l.append(q)

        if q.child_ne is not None:
            self._generateList(q.child_ne)
        if q.child_nw is not None:
            self._generateList(q.child_nw)
        if q.child_se is not None:
            self._generateList(q.child_se)
        if q.child_sw is not None:
            self._generateList(q.child_sw)

    def isNeighbour(self, q):
        is_neighbor = False
        our_size = self.x, self.y
        new_size = q.x, q.y

        for i in range(2):
            if our_size[i][0] >= new_size[i][0] and our_size[i][1] <= new_size[i][1]:
                is_neighbor = True
            if (new_size[i][0] <= our_size[i][0] <= new_size[i][1]) \
                    or (new_size[i][0] <= our_size[i][1] <= new_size[i][1]):
                is_neighbor = True
            if our_size[i][1] >= new_size[i][1] and our_size[i][0] <= new_size[i][0]:
                is_neighbor = True

        return is_neighbor

    def gridClustering(self, List):
        if len(List) == 0:
            return

        remove = []
        biggest = List[len(List) - 1]
        stop_rule = biggest.getDensity() * 0.5

        current_clust = [biggest.payload]

        remove.append(len(List) - 1)

        tmpSize = [biggest.getSizeDouble()]

        for i in range(len(List) - 1, -1, -1):
            current = List[i]
            if current.getDensity() > stop_rule and current != biggest \
                    and current.isNeighbour(biggest):
                current_clust.append(current.payload)
                remove.append(i)
                tmpSize.append(current.getSizeDouble())

        for item in remove:
            del List[item]

        QuadTree.ccluster.append(current_clust)
        cnumber = len(QuadTree.ccluster) - 1
        if cnumber not in QuadTree.csize:
            QuadTree.csize.append([cnumber, tmpSize])

        self.gridClustering(List)

    def getList(self, q):
        self._generateList(q)
        self.l = list(sorted(self.l, key=lambda x: x.getDensity()))


a = np.random.random((300, 2))
TREE = QuadTree(None, a)
QuadTree.size = a.shape[0]
QuadTree.alpha = np.sqrt(TREE.size)
TREE.x = np.min(a[:, 0]), np.max(a[:, 0])
TREE.y = np.min(a[:, 1]), np.max(a[:, 1])
QuadTree.recursiveSplit(TREE)
TREE.getList(TREE)
TREE.gridClustering(TREE.l)

print(len(TREE.ccluster))

ctraindata = []
delist = []
for i in range(len(QuadTree.ccluster)):
    ctraindata.append([])
    current = QuadTree.ccluster[i]
    if len(current) > 4:
        for j in range(len(current)):
            if len(ctraindata[i]) == 0:
                ctraindata[i] = current[j]
            else:
                ctraindata[i] = np.concatenate((ctraindata[i], current[j]), axis=0)
    if len(ctraindata[i]) == 0:
        delist.append(i)


for i in range(len(delist)):
    ctraindata.pop(delist[len(delist) - 1 - i])


