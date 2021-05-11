import pandas as pd


class Node:

    def __init__(self, dataframe: pd.DataFrame):
        self._father = None
        self._right = None
        self._left = None
        self._data = dataframe

    @property
    def father(self):
        return self._father

    @property
    def right_son(self):
        return self._right

    @property
    def left_son(self):
        return self._left

    @property
    def node_data(self):
        return self._data

    @father.setter
    def father(self, father):
        self._father = father

    @right_son.setter
    def right_son(self, node):
        self._right = node

    @left_son.setter
    def left_son(self, node):
        self._left = node

    @node_data.setter
    def node_data(self, dataframe: pd.DataFrame):
        self._data = dataframe
