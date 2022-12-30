import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

import graph

cities50Density = 0.0002

increasingNodesSmall = [(i, np.sqrt(i/cities50Density)) for i in range(3, 21)]
increasingNodesLarge = [(i, np.sqrt(i/cities50Density)) for i in range(25, 501, 25)]
decreasingDensitySmall = [(12, i) for i in range(12, 1213, 120)]
decreasingDensityLarge = [(250, i) for i in range(250, 25251, 2500)]

testCases = {"increasingNodesSmall": increasingNodesSmall, "decreasingDensitySmall" : decreasingDensitySmall,
             "increasingNodesLarge": increasingNodesLarge, "decreasingDensityLarge" : decreasingDensityLarge}

def runAllTests():
    for testName, tests in testCases.items():
        test(tests, 5, testName)

def generateEuclidGraph(n, maxDist, name):
    coords = np.random.rand(n, 2) * maxDist
    np.savetxt(name, coords, fmt="%d")


def test(tests, runsPerTest, testName):
    allResults = []
    path = "testResults/" + testName + "/"
    os.makedirs(path + "testGraphs", exist_ok=True)
    for n, maxDist in tests:
        name = path + "testGraphs/testGraph" + str(n) + "_" + str(int(maxDist))
        if n > 16:
            runs = 1
        else:
            runs = runsPerTest
        resultMatrix = np.zeros((runs, 8))
        for i in range(runs):
            fileName = name + "_" + str(i)
            print(fileName) # to see which file could make a test fail also indicating n and maxDist values
            generateEuclidGraph(n, maxDist, fileName)
            g = graph.Graph(-1, fileName, testMode=True)
            algos = [g.tourValue, g.swapHeuristic, g.TwoOptHeuristic, g.Greedy,
                     g.Christofides, g.Christofides, g.TwoOptHeuristic]  # first column is for unsolved value
            for j, algo in enumerate(algos):
                if j in [1,2,6]: # indices of algorithms requiring k argument
                    algo(-1)
                elif j == 5: # second Christofides index as first is for non-optimal matching
                    algo(optimal = True)
                else:
                    algo()
                resultMatrix[i, j] = g.tourValue()
            if n <= 16:
                resultMatrix[i, 7] = solve_tsp_dynamic_programming(np.array(g.dists))[1]
        nAndDensity = [n, n / (maxDist * maxDist)]
        result = np.concatenate((nAndDensity, resultMatrix.mean(axis=0)))
        allResults.append(result)
    allResults = np.array(allResults)
    np.savetxt(path + testName + "Results.csv", allResults, delimiter=",", fmt="%f")
    return allResults


def solve_tsp_dynamic_programming(
        distance_matrix: np.ndarray,
        maxsize: Optional[int] = None
) -> Tuple[List, float]:
    N = frozenset(range(1, distance_matrix.shape[0]))
    memo: Dict[Tuple, int] = {}

    # Step 1: get minimum distance
    @lru_cache(maxsize=maxsize)
    def dist(ni: int, N: frozenset) -> float:
        if not N:
            return distance_matrix[ni, 0]

        # Store the costs in the form (nj, dist(nj, N))
        costs = [
            (nj, distance_matrix[ni, nj] + dist(nj, N.difference({nj})))
            for nj in N
        ]
        nmin, min_cost = min(costs, key=lambda x: x[1])
        memo[(ni, N)] = nmin
        return min_cost

    best_distance = dist(0, N)

    # Step 2: get path with the minimum distance
    ni = 0  # start at the origin
    solution = [0]
    while N:
        ni = memo[(ni, N)]
        solution.append(ni)
        N = N.difference({ni})

    return solution, best_distance

runAllTests()