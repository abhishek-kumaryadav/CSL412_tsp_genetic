import sys
import math
import bisect
from sys import stdin, stdout
from math import gcd, floor, sqrt, log
from collections import defaultdict as dd
from itertools import permutations
from bisect import bisect_left as bl, bisect_right as br
from functools import lru_cache
import argparse
import matplotlib.pyplot as plt

from graph import Graph
from genetic import Genetic

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
from datetime import datetime


def main():
    logger = logging.getLogger(__name__)
    print(__name__)
    parser = argparse.ArgumentParser(description="Description for my parser")
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument(
        "-i",
        "--input",
        help="Input: Specify input file name, or specify index for following filenames: 1 gr21(default), 2 fri26, 3 dantzig42",
        required=False,
        default="1",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output: Specify output file name",
        required=False,
        default="{}".format(timestr),
    )
    parser.add_argument(
        "-p",
        "--population",
        help="Population: Specify initial population size(default=1000)",
        required=False,
        default="1000",
    )
    parser.add_argument(
        "-m",
        "--mutation",
        help="Mutation: Specify mutation rate(default=0.75)",
        required=False,
        default="0.75",
    )
    parser.add_argument(
        "-c",
        "--crossover",
        help="Crossover: Specify crossover function(integer index) to use from given list: 1 PMX, 2 Order(default), 3 rsb",
        required=False,
        default=2,
    )
    parser.add_argument(
        "-cr",
        "--crossoverrate",
        help="Crossover Rate: Specify percentage of population to crossover per generation(default=1)",
        required=False,
        default=1,
    )
    parser.add_argument(
        "-g",
        "--generation",
        help="Generation: Specify total number of generations(default=300)",
        required=False,
        default=300,
    )
    parser.add_argument(
        "-pi",
        "--populationinit",
        help="Population initialization function: 1 Random valid, 2 Random invalid(default)",
        required=False,
        default=2,
    )
    parser.add_argument(
        "-pdr",
        "--populationdecayrate",
        help="Population decay rate :Rate at which population reduces(default=0)",
        required=False,
        default=0,
    )
    parser.add_argument(
        "-ps",
        "--parentselection",
        help="Parent selection function : 1 Random(default), 2 Roulette, 3 Best cost",
        required=False,
        default=1,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose : 1 Smaller log(default), 2 Detailed log",
        required=False,
        default=1,
    )
    levels = [logging.INFO, logging.DEBUG]
    argument = parser.parse_args()
    FORMAT = "[%(asctime)s | [%(module)s] | %(funcName)s() ] %(message)s"
    logging.basicConfig(
        format=FORMAT,
        filename="out_{}.log".format(argument.output),
        encoding="utf-8",
        level=levels[int(argument.verbose) - 1],
    )

    logger.info(
        "Starting execution: {}".format(datetime.now().strftime("%Y/%m/%d-%I:%M:%S %p"))
    )
    logger.info("Parsed arguments: {}".format(argument))
    genetic = Genetic(argument)
    costs = genetic.run_genetic_algorithm()[1:]
    generations = [i for i in range(len(costs))]

    plt.scatter(generations, costs)
    plt.title(
        "Populationsize: {} | Generations: {} | Final Cost: {}".format(
            argument.population, argument.generation, costs[-1]
        )
    )
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.savefig(
        "{}_{}_{}.png".format(
            genetic.graph.input_file_name, genetic.image_name, argument.output
        )
    )


if __name__ == "__main__":
    main()