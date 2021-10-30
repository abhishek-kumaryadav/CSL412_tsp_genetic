import sys
import math
import bisect
from sys import stdin, stdout
from math import gcd, floor, sqrt, log
from collections import defaultdict as dd
from itertools import permutations
from bisect import bisect_left as bl, bisect_right as br
from functools import lru_cache

import numpy as np


sys.setrecursionlimit(100000000)
import logging
import collections

int_r = lambda: int(sys.stdin.readline())
str_r = lambda: sys.stdin.readline().strip()
intList_r = lambda: list(map(int, sys.stdin.readline().strip().split()))
strList_r = lambda: list(sys.stdin.readline().strip())
jn = lambda x, l: x.join(map(str, l))
mul = lambda: map(int, sys.stdin.readline().strip().split())
mulf = lambda: map(float, sys.stdin.readline().strip().split())
ceil = lambda x: int(x) if (x == int(x)) else int(x) + 1
ceildiv = lambda x, d: x // d if (x % d == 0) else x // d + 1
flush = lambda: stdout.flush()
outStr = lambda x: stdout.write(str(x))
mod = 1000000007
logger = logging.getLogger(__name__)
import pprint as pp

INF = 1000000009


class Graph:
    dataset = [
        "./datasets/gr21.txt",
        "./datasets/fri26.txt",
        "./datasets/dantzig42.txt",
    ]

    def __init__(self, input_file_name=None) -> None:
        if input_file_name is None:
            input_file_name = "in.txt"
        else:
            self.input_file_name = input_file_name
        self.matrix = list()
        self.take_inputs()
        self.size = len(self.matrix)
        logger.info("Initialized {}".format(__name__))

    def take_inputs(self):
        logger.debug("Starting matrix initialization")
        if self.input_file_name.isnumeric():
            self.lower_diag_input_util(Graph.dataset[int(self.input_file_name) - 1])
            self.input_file_name = Graph.dataset[int(self.input_file_name) - 1]
        else:
            self.matrix_input_util()
        # matrix_str = pp.pformat(self.matrix)
        self.input_file_name = self.input_file_name.rstrip(".txt")
        logger.debug("Initialized graph class with matrix: {}".format(self.matrix))

    def matrix_input_util(self):
        logger.debug("Entering {}".format("matrix_input_util"))
        adj = collections.defaultdict(dict)
        with open(self.input_file_name, "r") as f:
            lines = f.readlines()
            i = 0
            for line in lines:  # Outer loop, for each line: maintains row in matrix
                graph_input = line.strip().split(" ")
                # graph_input = graph_input[0 : len(graph_input) - 1]
                if graph_input is not None:
                    j = 0
                    # Inner loop, for each weight: maintains column in matrix
                    for weight in graph_input:
                        adj[i][j] = float(weight)
                        adj[j][i] = float(weight)
                        j += 1
                i += 1
        self.matrix = adj
        logger.debug("Exiting {}".format("matrix_input_util"))

    def lower_diag_input_util(self, filename):
        logger.debug("Entering {}".format("lower_diag_input_util"))
        numbers = list()
        with open(filename, "r") as given_file:
            lines = given_file.readlines()
            for i, line in enumerate(lines):
                line = line.split()
                # logger.debug("Printing line #{}: {}".format(i, line))
                for c in line:
                    numbers.append(int(c))
        # logger.debug("Numbers array formed: {}".format(numbers))
        nodes = 0
        sz = len(numbers)

        while nodes * (nodes + 1) / 2 != sz:
            nodes += 1
        adj = collections.defaultdict(dict)
        index = 0
        # logger.debug(
        #     "Numbers array for custom dataset formed: {}".format(pp.pformat(numbers))
        # )
        for i in range(nodes):
            for j in range(i + 1):
                adj[i][j] = float(numbers[index])
                adj[j][i] = float(numbers[index])
                index += 1
        self.matrix = adj
        logger.debug("Exiting {}".format("lower_diag_input_util"))

    def tsp_cost_singular(self, path):
        # logger.debug("Entering {}".format("Graph.tsp_cost_singular"))

        cost = 0
        for i in range(len(path) - 1):
            if self.matrix[path[i]][path[i + 1]] != -1:
                cost += self.matrix[path[i]][path[i + 1]]
            else:
                cost = INF
                break
        if self.matrix[path[0]][path[-1]] != -1:
            cost += self.matrix[path[0]][path[-1]]
        else:
            cost = INF
        # logger.debug("Returning cost: {}".format(cost))
        return cost

    def tsp_cost_multiple(self, paths):
        # logger.debug("Entering {}".format("Graph.tsp_cost_multiple"))
        costs = list()
        for path in paths:
            costs.append(self.tsp_cost_singular(path))
        # logger.debug("Returning costs: {}".format(costs))
        return costs