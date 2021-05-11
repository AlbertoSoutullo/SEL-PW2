import pandas as pd


class Node:

    def __init__(self, left=None, right=None, best_attribute=None, results=None):
        self._left = left
        self._right = right
        self._best_attribute = best_attribute
        self._results = results
