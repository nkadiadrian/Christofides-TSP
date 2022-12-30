import math
import numpy as np
import matching
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def euclid(p, q):
    x = p[0] - q[0]
    y = p[1] - q[1]
    return math.sqrt(x * x + y * y)


def makeNode(node):
    return [float(n) for n in node]


class Graph:

    # Complete as described in the specification, taking care of two cases:
    # the -1 case, where we read points in the Euclidean plane, and
    # the n>0 case, where we read a general graph in a different format.
    # self.perm, self.dists, self.n are the key variables to be set up.
    def __init__(self, n, filename, testMode=False):
        self.testMode = testMode
        self.algoName = "Node Numeric Order"
        self.dists = []
        self.n = 0
        self.odd = []
        self.edges = []
        self.xCoords = []
        self.yCoords = []
        if not testMode:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(30, 30))
            self.ax.grid(True, alpha=0.5)
        with open(filename) as f:
            if n < 0:
                for lineI in f:
                    nodeI = makeNode(lineI.split())
                    self.xCoords.append(nodeI[0])
                    self.yCoords.append(nodeI[1])
                    self.n += 1
                    row = []
                    with open(filename) as jIndexF:
                        for lineJ in jIndexF:
                            nodeJ = makeNode(lineJ.split())
                            row.append(euclid(nodeI, nodeJ))
                        self.dists.append(row)
            else:
                self.n = n
                # self.dists = [[0]*n for i in range(n)]
                self.dists = [[np.inf] * n for i in range(n)]  # replaced 0 with np.inf for Christofides extension
                for line in f:
                    edge = makeNode(line.split())
                    self.dists[int(edge[0])][int(edge[1])] = edge[2]
                    self.dists[int(edge[1])][int(edge[0])] = edge[2]
        self.perm = [i for i in range(self.n)]

        # for purpose of scaling graphs
        if self.xCoords:
            self.rad = max(self.xCoords + self.yCoords) / 20

    # Complete as described in the spec, to calculate the cost of the
    # current tour (as represented by self.perm).
    def tourValue(self):
        tourCost = 0
        for i in range(self.n):
            tourCost += self.dists[self.perm[i - 1]][self.perm[i]]
        return tourCost

    # Attempt the swap of cities i and i+1 in self.perm and commit
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self, i):
        oldCost = self.tourValue()
        j = (i + 1) % self.n
        self.swapNodes(i, j)
        if self.tourValue() < oldCost:
            return True
        else:
            self.swapNodes(i, j)
            return False

    def swapNodes(self, i, j):
        self.perm[i], self.perm[j] = self.perm[j], self.perm[i]

    # Consider the effect of reversing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.              
    def tryReverse(self, i, j):
        oldCost = self.tourValue()
        self.reverseNodes(i, j)
        if self.tourValue() < oldCost:
            return True
        else:
            self.reverseNodes(i, j)
            return False

    def reverseNodes(self, i, j):
        self.perm[i:j + 1] = self.perm[i:j + 1][::-1]

    def swapHeuristic(self, k):
        self.algoName = "Swap Heuristic"
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for i in range(self.n):
                if self.trySwap(i):
                    better = True

    def TwoOptHeuristic(self, k):
        self.algoName = "Two Opt Heuristic"
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for j in range(self.n - 1):
                for i in range(j):
                    if self.tryReverse(i, j):
                        better = True

    # Implement the Greedy heuristic which builds a tour starting
    # from node 0, taking the closest (unused) node as 'next'
    # each time.
    def Greedy(self):
        self.algoName = "Greedy"
        newPerm = [0] * self.n
        for i in range(self.n):
            minimum = self.dists[newPerm[i - 1]][self.perm[0]]
            index = 0
            for j in range(len(self.perm)):
                if self.dists[newPerm[i - 1]][self.perm[j]] < minimum:
                    index = j
                    minimum = self.dists[newPerm[i - 1]][self.perm[j]]
            newPerm[i] = self.perm.pop(index)
        self.perm = newPerm

    # Beginning of Christofides implementation
    def makeMST(self, plot=False):
        self.edges = [[0] * self.n for i in range(self.n)]
        inputs = [[np.inf, node, -1] for node in
                  range(self.n)]  # each row in format (minimum edge value, node from, node to at minimum edge value
        inputs[0][0] = 0
        while inputs:
            u = min(inputs)
            inputs.remove(min(inputs))
            if u[2] != -1:
                self.edges[u[2]][u[1]] = 1
                self.edges[u[1]][u[2]] = 1
            for inputElem in inputs:
                if self.dists[u[1]][inputElem[1]] < inputElem[0]:
                    inputElem[0] = self.dists[u[1]][inputElem[1]]
                    inputElem[2] = u[1]
        if plot:
            for i in range(self.n):
                for j in range(self.n):
                    if self.edges[i][j] == 1:
                        self.plotArrow([i, j], "green")
        return self.edges

    def findOddVertices(self):
        self.odd = []
        for i in range(self.n):
            if sum(self.edges[i]) % 2 == 1:
                self.odd.append(i)

    def greedyMinWeightMatching(self, plot=False):
        while self.odd:
            origin = self.odd.pop()
            closestDist = np.inf
            dest = "No possible Match found"
            for node in self.odd:
                if self.dists[origin][node] < closestDist:
                    dest = node
                    closestDist = self.dists[origin][node]
            self.odd.remove(dest)
            self.edges[origin][dest] += 1
            self.edges[dest][origin] += 1
            if plot:
                self.plotArrow([origin, dest], "orange")
                self.plotArrow([dest, origin], "orange")

    def getOddArray(self):
        for i in range(self.n):  # ensures edges between the same node are ignored
            self.dists[i][i] = np.inf
        return [dist[self.odd].tolist() for dist in np.array(self.dists)[self.odd]]

    def optimalMinWeightMatching(self, plot = False):
        minimized = matching.minimize(self.getOddArray())
        newEdges = []
        odd = []
        for newEdge in minimized: # solves abnormality in Hungarian algorithm that allows imperfect matching
            if newEdge[::-1] in minimized:
                newEdges.append(newEdge)
            else:
                odd.append(self.odd[newEdge[0]])
        for nodeFrom, nodeTo in newEdges:
            self.edges[self.odd[nodeFrom]][self.odd[nodeTo]] += 1
            if plot:
                self.plotArrow([nodeFrom, nodeTo], "purple")
        self.odd = odd
        self.greedyMinWeightMatching(plot=plot)

    def getEuclidTour(self, origin):
        tour = [origin]
        nodeFrom = origin
        nodeTo = self.edges[nodeFrom].index(max(self.edges[nodeFrom]))
        while self.edges[nodeFrom][nodeTo] > 0:
            tour.append(nodeTo)
            self.edges[nodeFrom][nodeTo] -= 1
            self.edges[nodeTo][nodeFrom] -= 1
            nodeFrom = nodeTo
            nodeTo = self.edges[nodeFrom].index(max(self.edges[nodeFrom]))
        tour = self.checkComplete(tour)
        return tour

    # ensures any loops missed by previous passes over the edges array are visited
    # these missed loops are created when a node has cardinality above 3
    def checkComplete(self, tour):
        for i in range(len(tour)):
            if sum(self.edges[tour[i]]) > 0:
                tour = tour[:i] + self.getEuclidTour(tour[i]) + tour[i:]
                tour = self.checkComplete(tour)
        return tour

    def getHamiltonTour(self):
        self.perm = list(dict.fromkeys(self.getEuclidTour(self.n - 1)))  # nubs all repeated edges from euler tour

    def Christofides(self, optimal=False, plot=False):
        self.algoName = "Christofides"
        self.makeMST(plot=plot)
        self.findOddVertices()
        if optimal:
            self.optimalMinWeightMatching(plot=plot)
        else:
            self.greedyMinWeightMatching(plot=plot)
        self.getHamiltonTour()

    def plotArrow(self, nodes, color):
        xs = [self.xCoords[nodes[0]], self.xCoords[nodes[1]]]
        ys = [self.yCoords[nodes[0]], self.yCoords[nodes[1]]]
        arrow = FancyArrowPatch(posA=(xs[0], ys[0]), posB=(xs[1], ys[1]), shrinkA=self.rad, shrinkB=self.rad,
                                color=color, arrowstyle='-|>', mutation_scale=20, lw=5)
        self.ax.add_patch(arrow)

    def plotVertices(self):
        area = self.rad * self.rad * 4
        self.ax.scatter(self.xCoords, self.yCoords, s=area, alpha=0.6, edgecolors='k')
        for i in range(self.n):
            self.ax.text(self.xCoords[i], self.yCoords[i], str(i), fontsize=self.rad, ha="center", va="center")

    def plotPath(self):
        for i in range(len(self.perm)):
            self.plotArrow([self.perm[i - 1], self.perm[i]], 'k')
        self.plotVertices()
        self.fig.suptitle(self.algoName, fontsize=70)
        self.fig.show()
