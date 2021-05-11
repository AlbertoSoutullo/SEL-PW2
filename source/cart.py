from typing import List
import pandas as pd
import numpy as np

from source.node import Node
from utils import divide_set, gini_impurity, _binary_splits_no_dupes, get_categorical_splits


class Cart:

    def __init__(self):
        self._root = None
        self._splits_done = 0
        self._feature_selecteds = {}

    def fit(self, dataset: pd.DataFrame):

        for column in dataset.columns[:-1]:
            self._feature_selecteds[column] = 0

        self._root = self._expand_binary_tree(dataset)

    def classify(self, dataset):

        if self._root._results is not None:
            return self._root._results




    def _expand_binary_tree(self, dataset):
        if len(dataset) == 0:
            return Node()

        best_attribute = None
        best_sets = None
        best_impurity = np.inf
        keep_expanding = False

        current_impurity = gini_impurity(dataset)

        if current_impurity != 0:
            # For each column
            for column in dataset.columns[:-1]:
                unique_values_in_column = list(set(dataset[column]))
                unique_values_in_column.sort()

                if not isinstance(unique_values_in_column[0], str):
                    unique_values_in_column = (np.array(unique_values_in_column[:-1]) +
                                               np.array(unique_values_in_column[1:])) / 2
                else:
                    unique_values_in_column = get_categorical_splits(unique_values_in_column)

                # For each binary split
                for u_value in unique_values_in_column:
                    set1, set2 = divide_set(dataset, column, u_value)
                    gini1 = gini_impurity(set1)
                    gini2 = gini_impurity(set2)
                    impurity_set = (len(set1)/len(dataset)) * gini1 + (len(set2)/len(dataset)) * gini2

                    if impurity_set < best_impurity and len(set1) > 0 and len(set2) > 0:
                        best_attribute = column
                        best_sets = (set1, set2)
                        best_impurity = impurity_set
                        keep_expanding = True

        if keep_expanding:
            self._feature_selecteds[best_attribute] += 1
            self._splits_done += 1
            left_node = self._expand_binary_tree(best_sets[0])
            right_node = self._expand_binary_tree(best_sets[1])
            return Node(left_node, right_node, best_attribute, dataset.value_counts())
        else:
            return Node(None, None, None, dataset.value_counts())

    def calculate_feature_importance(self):
        feature_importance = {}
        for key, value in self._feature_selecteds.items():
            feature_importance[key] = value / self._splits_done

        return feature_importance


test = pd.DataFrame({"height": ["alto", "alto", "bajo", "alto", "test"],
                     "weight": [70, 80, 90, 60, 0],
                     "age": [20, 30, 40, 10, 0],
                     "class": ["A", "A", "B", "A", "C"]})
print(test)
cart = Cart()
cart.fit(test)
f_i = cart.calculate_feature_importance()
print(f_i)