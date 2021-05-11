import itertools as it
import numpy as np
from typing import List


def _perform_discrete_partitions(attributes: List) -> List:
    """
    This function performs discrete partitions (2^(v-1) - 1) of a given list of attributes. This means, all but
    empty (first) and full (last) set.

    :param attributes: List of discrete attributes.
    :return: List of discrete partitions.
    """
    unique_attributes = set(attributes)
    gini_combinations = list(it.chain.from_iterable(it.combinations(
        unique_attributes, r) for r in range(len(unique_attributes) + 1)))
    gini_combinations = gini_combinations[1:len(gini_combinations) - 1]
    gini_combinations = list(set(gini_combinations))
    gini_combinations = [gini_combinations[n:n+2] for n in range(0, len(gini_combinations), 2)]

    return gini_combinations


def _perform_continuous_partitions(attributes: List) -> List:
    """
    This function calculates the middle point between each element of a list of numbers.

    :param attributes: List of numbers to use.
    :return: List of split points. Example: Input -> [1, 3, 5]; Output -> [2, 4]
    """
    unique_attributes = set(attributes)
    sorted_attributes = sorted(unique_attributes)
    np_sorted_attributes = np.array(sorted_attributes)
    split_points = (np_sorted_attributes[:-1]+np_sorted_attributes[1:]) / 2

    return split_points


