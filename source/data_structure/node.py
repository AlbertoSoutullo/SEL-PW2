from __future__ import annotations
from typing import Tuple, Dict


class Node:
    """
    This class handles the data of each node in a tree.
    """

    def __init__(self, left: Node = None, right: Node = None, best_attribute: int = None,
                 right_values: Tuple = None, leaf_results: Dict = None):
        """
        :param left: Reference to the left node
        :param right: Reference to the right node
        :param best_attribute: Attribute used to split the node
        :param right_values: Attribute to compare the data, in order to send new data to right. This tuple can have
        one or multiple tuples like ('column', 'value to compare').
        :param leaf_results: None if this node is a leaf, results of classes otherwise in a Dict format, like
        {'class', 'number_of_classes'}.
        """
        self._left_node = left
        self._right_node = right
        self._best_attribute = best_attribute
        self._right_values = right_values
        self._leaf_results = leaf_results

    @property
    def left_node(self):
        return self._left_node

    @property
    def right_node(self):
        return self._right_node

    @property
    def best_attribute(self):
        return self._best_attribute

    @property
    def right_values(self):
        return self._right_values

    @property
    def leaf_results(self):
        return self._leaf_results
