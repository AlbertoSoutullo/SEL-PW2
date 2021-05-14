

class Node:

    def __init__(self, left=None, right=None, best_attribute=None, right_values=None, leaf_results=None):
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
