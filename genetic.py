import sys
import math
import bisect
from sys import stdin, stdout
from math import gcd, floor, sqrt, log
from collections import defaultdict as dd
from itertools import permutations
from bisect import bisect_left as bl, bisect_right as br
from functools import lru_cache
from numpy import argsort
from numpy import dtype, random
from graph import Graph
import json

sys.setrecursionlimit(100000000)

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
import logging
import numpy as np
import copy

INF = 1000000009

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
np.set_printoptions(threshold=np.inf)


class Genetic:
    crossf_dict = ["pmx", "order", "rsb"]
    # mutatef_dict = ["random"]
    par_select_dict = ["random", "routette", "bestCost"]
    pop_init_dict = ["randomValid", "randomInvalid"]

    def __init__(self, args) -> None:
        self.distance = list()
        self.population = list()
        self.graph = Graph(args.input)
        self.mutation_rate = float(args.mutation)
        self.crossover_index = int(args.crossover) - 1
        self.crossover_rate = float(args.crossoverrate)
        self.max_generation = int(args.generation)
        self.population_init_index = int(args.populationinit)
        self.population_decay_rate = float(args.populationdecayrate)
        self.output_file_name = args.output
        self.population_size = int(args.population)
        self.parent_selection = int(args.parentselection) - 1
        self.image_name = "_".join(
            [
                str(i)
                for i in [
                    self.crossover_rate,
                    self.mutation_rate,
                    self.population_decay_rate,
                    self.population_size,
                    self.max_generation,
                    Genetic.crossf_dict[self.crossover_index],
                    Genetic.par_select_dict[self.parent_selection],
                    # self.par_select_dict,
                    Genetic.pop_init_dict[self.population_init_index - 1],
                ]
            ]
        )
        if self.crossover_index == 2:
            self.initialize_distance()
        self.init_population(self.population_size)

    def initialize_distance(self):
        logger.debug("Initializing Floyd Warshal Distance matrix")
        self.distances = [
            [INF for i in range(self.graph.size)] for j in range(self.graph.size)
        ]
        # print(self.graph.matrix[49][49])
        for i in range(self.graph.size):
            self.distances[i][i] = 0
            for j in range(self.graph.size):
                # print(i, j)
                if self.graph.matrix[i][j] != -1:
                    self.distances[i][j] = self.graph.matrix[i][j]
                    self.distances[j][i] = self.graph.matrix[i][j]

        for k in range(self.graph.size):
            for i in range(self.graph.size):
                for j in range(self.graph.size):
                    self.distances[i][j] = min(
                        self.distances[i][j],
                        self.distances[i][k] + self.distances[k][j],
                    )
        logger.debug("Floyd Warshall matrix initialized")

    def init_population(self, population_size):
        population_init_functions = [self.random_init_population]
        logger.debug(
            "Initializing population with {}, Keeping invalid entries: {}".format(
                population_init_functions[
                    (self.population_init_index - 1) // 2
                ].__name__,
                (not (self.population_init_index & 1)),
            )
        )
        population_init_functions[(self.population_init_index - 1) // 2](
            population_size, (self.population_init_index & 1)
        )
        logger.debug("Population initialized as: {}".format(self.population))

    def random_init_population(self, population_size, valid):
        # print(population_size)
        path = tuple([i for i in range(self.graph.size)])
        while len(self.population) < population_size:
            logger.debug("Current population size: {}".format(len(self.population)))
            permutation = np.random.permutation(path)
            cost = self.graph.tsp_cost_singular(permutation)
            if valid and (cost == INF):
                continue
            else:
                np_path = np.array(permutation, dtype=float)
                np_path = np.insert(np_path, 0, cost)
                self.population.append(np_path)
        self.population = np.array(self.population)

    def run_genetic_algorithm(self):
        logger.info("Starting genetic algorithm")
        # print(np.sort(self.population, axis=0))
        # print(self.population)
        current_generation = 1
        # idx = np.argpartition(self.population[:, 0], axis=0, kth=10)
        # print(idx)
        # return
        # print(self.population[idx[:10]])
        # print(np.sort(self.population, axis=0)[:10])
        # return
        minimum_costs = [INF]
        minimum_paths = list()

        while current_generation <= self.max_generation and len(self.population) > 1:
            logger.info("Current generation: {}".format(current_generation))
            offsprings = self.crossover()
            # self.crossover_index = 1
            # offsprings2 = self.crossover()
            logger.debug("Offsprings: {}".format(offsprings))
            self.population = np.concatenate((offsprings, self.population))
            logger.debug(
                "Population after adding offspring: {}".format(len(self.population))
            )
            self.mutation()
            logger.debug("Population after mutation: {}".format(len(self.population)))
            # exit(0)
            new_population_size = int(
                (1 - self.population_decay_rate) * self.population_size
            )
            if new_population_size < len(self.population):
                idx = np.argpartition(
                    self.population[:, 0], axis=0, kth=new_population_size
                )
                self.population = self.population[idx[:new_population_size]]
                self.population_size = len(self.population)
                # self.population = self.population[self.population[:, 0].argsort()][
                #     :new_population_size
                # ]

            logger.debug("Population after decreasing size: {}".format(self.population))
            # minimum_cost = min(self.population.T[0])
            minimum_index = np.argmin(self.population[:, 0], axis=0)
            # print(self.population[minimum_index][0], minimum_cost)
            # return
            if self.population[minimum_index][0] < minimum_costs[-1]:
                minimum_costs.append(self.population[minimum_index][0])
                minimum_paths.append(self.population[minimum_index][1:])
            else:
                minimum_costs.append(self.population[minimum_index][0])
                minimum_paths.append(self.population[minimum_index][1:])
            logger.info(
                "#{} generation cost: {}, path: {}, population size: {}".format(
                    current_generation,
                    minimum_costs[-1],
                    minimum_paths[-1],
                    len(self.population),
                )
            )
            # data = {
            #     "generation": current_generation,
            #     "cost": minimum_cost_yet,
            #     "path": str(minimum_path_yet),
            # }
            # with open(self.output_file_name, "a") as f:
            # file_data = json.load(f)
            # file_data["generation_costs"].append(data)
            # f.seek(0)
            # json.dumps(data, f, ensure_ascii=False)
            # f.write(json.dumps(data))
            current_generation += 1
        return minimum_costs

    def crossover(self):
        logger.debug("Starting crossover")
        crossover_functions = [
            self.pmx_crossover,
            self.order_crossover,
            self.rsb_crossover,
        ]
        offsprings = None
        parents = self.create_parents()
        for p in range(0, len(parents), 2):
            parent1 = self.population[parents[p]]
            parent2 = self.population[parents[p + 1]]
            logger.debug("Crossing {} and {}".format(parent1, parent2))

            new_offsprings = crossover_functions[self.crossover_index](parent1, parent2)
            if offsprings is None:
                offsprings = new_offsprings
            else:
                offsprings = np.concatenate((offsprings, new_offsprings))
        return np.array(offsprings)

    def rsb_crossover(self, parent1, parent2):
        b = 1
        i = random.choice([i for i in range(self.graph.size)])
        dc = sum(self.distances[i]) / (b * (self.graph.size - 1))

        p1 = np.array(list(map(int, copy.deepcopy(parent1[1:]))))
        p2 = np.array(list(map(int, copy.deepcopy(parent2[1:]))))

        visited = [False] * self.graph.size
        visited[i] = True
        offspring = [i]
        while not all(visited):
            current_node = offspring[-1]
            j1 = np.where(p1 == current_node)[0][0]
            j2 = np.where(p2 == current_node)[0][0]
            # print(j1, j2)
            # print(type(p1), type(p2))
            j1 = p1[(j1 + 1) % len(p1)]
            j2 = p2[(j2 + 1) % len(p2)]

            dj1 = self.distances[current_node][j1]
            dj2 = self.distances[current_node][j2]

            dmin = min(dj1, dj2)

            if dmin <= dc:
                if dmin == dj1 and not visited[j1]:
                    offspring.append(j1)
                elif dmin == dj2 and not visited[j2]:
                    offspring.append(j2)
                else:
                    min_dist = INF
                    var_enu = 0
                    for i_enu, i_d in enumerate(self.distances[i]):
                        if not visited[i_enu] and i_d < min_dist:
                            min_dist = i_d
                            var_enu = i_enu
                    offspring.append(var_enu)
            else:
                min_dist = INF
                var_enu = 0
                for i_enu, i_d in enumerate(self.distances[i]):
                    if not visited[i_enu] and i_d < min_dist:
                        min_dist = i_d
                        var_enu = i_enu
                offspring.append(var_enu)
            visited[offspring[-1]] = True
        cost = self.graph.tsp_cost_singular(offspring)
        logger.debug("Offspring from rsb function: {}".format(offspring))
        offspring = np.array(offspring, dtype=float)
        offspring = np.insert(offspring, 0, cost)
        logger.debug("Offspring from rsb function: {}".format(offspring))
        # exit(0)b
        return np.array([offspring])

    def order_crossover(self, parent1, parent2):
        offsprings = list()
        o1 = np.array(list(map(int, copy.deepcopy(parent1[1:]))))
        o2 = np.array(list(map(int, copy.deepcopy(parent2[1:]))))
        idx1 = random.randint(0, len(o1) - 1)
        idx2 = random.randint(0, len(o2) - 1)
        while idx1 >= idx2:
            idx1 = random.randint(0, len(o1) - 1)
            idx2 = random.randint(0, len(o2) - 1)
        set1 = set(o1[idx1 : idx2 + 1])
        set2 = set(o2[idx1 : idx2 + 1])
        list1 = np.concatenate((o1[idx2 + 1 :], o1[0 : idx2 + 1]))
        list2 = np.concatenate((o2[idx2 + 1 :], o2[0 : idx2 + 1]))
        # logger.debug("o1: {}, o2: {}".format(o1, o2))
        # logger.debug("idx1: {}, idx2: {}".format(idx1, idx2))
        # logger.debug("list1: {}, list2: {}".format(list1, list2))
        # logger.debug("set1: {}, set2: {}".format(set1, set2))

        # list2 = o2[idx2 + 1 :] + o2[0:idx1]
        for j in [idx2 + 1, 0]:
            # j = j % self.graph.size
            for i in list2:
                if j == self.graph.size or (j == idx1):
                    break
                if i not in set1:
                    o1[j] = i
                    j += 1
                    set1.add(i)
            # logger.debug("o1: {}, o2: {}".format(o1, o2))
            # logger.debug("set1: {}, set2: {}\n".format(set1, set2))
        # logger.debug("\n")
        # exit(0)
        for j in [idx2 + 1, 0]:
            for i in list1:
                if j == len(o1) or j == idx1:
                    break
                if i not in set2:
                    o2[j] = i
                    j += 1
                    set2.add(i)
            # logger.debug("o1: {}, o2: {}".format(o1, o2))
            # logger.debug("set1: {}, set2: {}\n".format(set1, set2))

        logger.debug("o1: {}, o2: {}".format(o1, o2))
        # exit(0)
        for i in [o1, o2]:
            logger.debug("Finding cost of child: {}".format(i))
            cost = self.graph.tsp_cost_singular(i)
            offspring = np.array(i, dtype=float)
            offspring = np.insert(i, 0, cost)
            offsprings.append(offspring)
        return np.array(offsprings)

    def pmx_crossover(self, parent1, parent2):
        offsprings = list()
        idx = random.randint(0, len(parent1) - 1)
        p1 = np.array(list(map(int, copy.deepcopy(parent1[1:]))))
        p2 = np.array(list(map(int, copy.deepcopy(parent2[1:]))))
        o1 = copy.deepcopy(p1)
        o2 = copy.deepcopy(p2)
        # print(o1, o2, p1, p2)
        for j in range(0, idx + 1):
            logger.debug("idx :{}".format(idx))
            if p1[j] != p2[j]:
                ind1 = np.where(p1 == p2[j])
                ind2 = np.where(p2 == p1[j])
                # print(ind1,p1,p, ind2)
                logger.debug("Position of {} in parent1: {}".format(p2[j], ind1))
                logger.debug("Position of {} in parent2: {}".format(p1[j], ind2))
                o1[j], o1[ind1] = o1[ind1], o1[j]
                o2[j], o2[ind2] = o2[ind2], o2[j]
        for i in [o1, o2]:
            logger.debug("Finding cost of child: {}".format(i))
            cost = self.graph.tsp_cost_singular(i)
            offspring = np.array(i, dtype=float)
            offspring = np.insert(i, 0, cost)
            offsprings.append(offspring)
        return np.array(offsprings)

    def create_parents(self):
        create_parents_functions = [
            self.random_create_parents,
            self.roulette_parents,
            self.best_cost_parents,
        ]
        logger.debug(
            "Initializing parents with {}".format(
                create_parents_functions[self.parent_selection].__name__
            )
        )
        parent_size = int(len(self.population) * self.crossover_rate)
        parents = create_parents_functions[self.parent_selection](parent_size)[
            :parent_size
        ]

        if len(parents) & 1:
            parents = parents[:-1]
        logger.debug("Parents initialized as: {}".format(parents))
        return parents

    def random_create_parents(self, parent_size):
        return np.random.permutation([i for i in range(len(self.population))])

    def roulette_parents(self, parent_size):
        index_fit = dict()
        for i, path in enumerate(self.population):
            index_fit[i] = 100.0 / path[0]
        total_cost = sum(index_fit.values())
        roulette_wheel = [0.0]
        for key in index_fit:
            next_val = roulette_wheel[-1] + index_fit[key] / total_cost
            roulette_wheel.append(next_val)
        sz = len(roulette_wheel)
        logger.debug("Roulette Wheel created: {}".format(roulette_wheel))
        parents = list()
        while len(parents) != parent_size:
            wheel_spin = np.random.random()
            for i in range(1, sz):
                if (
                    wheel_spin >= roulette_wheel[i - 1]
                    and wheel_spin <= roulette_wheel[i]
                ):
                    parent = i - 1
                    if parent not in parents:
                        parents.append(parent)
                    break
        return np.array(parents)

    def best_cost_parents(self, parent_size):
        return self.population[:, 0].argsort()

    def mutation(self):
        mutation_functions = [self.random_mutation]
        num_mutation = int(self.mutation_rate * self.population_size)
        logger.debug("Starting mutation of {} population".format(num_mutation))
        for _ in range(num_mutation):
            pop_index = random.randint(0, self.population_size - 1)
            mutated_child = mutation_functions[0](pop_index)
            # exit(0)
            # self.population=np.concatenate(([mutated_child], self.population))
            self.population = np.append(self.population, [mutated_child], axis=0)
            # logger.debug(self.population)
            # exit(0)
            # index1 = random.randint(0, self.graph.size - 1)
            # index2 = random.randint(0, self.graph.size - 1)
            # while index1==index2:
            #     index2 = random.randint(0, self.graph.size - 1)
            # self.population[pop_index][index1],self.population[pop_index][index2]=self.population[pop_index][index2],self.population[pop_index][index1]

    def random_mutation(self, pop_index):
        index1 = random.randint(0, self.graph.size - 1) + 1
        index2 = random.randint(0, self.graph.size - 1) + 1
        mutated_child = copy.deepcopy(self.population[pop_index])
        while index1 == index2:
            index2 = random.randint(0, self.graph.size - 1) + 1
        temp = mutated_child[index2]
        mutated_child[index2] = mutated_child[index1]
        mutated_child[index1] = temp
        return mutated_child
