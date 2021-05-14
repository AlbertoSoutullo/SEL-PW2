import random
import pandas as pd
import numpy as np

from source.data_structure.node import Node
from source.utils.utils import get_categorical_splits, divide_set, gini_impurity, count_classes_in_dataset


class Cart:

    def __init__(self, attributes_to_use=-1, seed=0):
        self._root = None
        self._splits_done = 0
        self._features_chose = {}
        self._classifications = []
        self._attributes_to_use = attributes_to_use
        random.seed(seed)

    def fit(self, dataset: pd.DataFrame):

        for column in dataset.columns[:-1]:
            self._features_chose[str(column)] = 0

        self._root = self._expand_binary_tree(dataset)

    def classify(self, dataset):
        self._classifications.clear()

        for i, row in dataset.iterrows():
            possible_classes = self._recursive_classification(row, self._root)
            final_classification = max(possible_classes, key=possible_classes.get)
            self._classifications.append(final_classification)

        return self._classifications

    def _recursive_classification(self, row, node):
        if node.leaf_results is not None:
            return node.leaf_results
        else:
            v = row[node.best_attribute]
            if isinstance(v, str):
                if isinstance(node.right_values, tuple):
                    list_values = node.right_values
                else:
                    list_values = [node.right_values]
                if v in list_values:
                    next_node = node.right_node
                else:
                    next_node = node.left_node
            else:
                if v >= node.right_values:
                    next_node = node.left_node
                else:
                    next_node = node.right_node

        return self._recursive_classification(row, next_node)

    def _expand_binary_tree(self, dataset):
        if len(dataset) == 0:
            return Node()

        best_attribute = None
        right_values = None
        leaf_results = None
        best_sets = None
        best_impurity = np.inf
        keep_expanding = False

        if self._attributes_to_use == -1:
            subsample_attributes = dataset.columns[:-1]
        else:
            subsample_attributes = random.sample(list(dataset.columns[:-1]), self._attributes_to_use)

        current_impurity = gini_impurity(dataset)

        if current_impurity != 0:
            # For each column
            for column in subsample_attributes:
                unique_values_in_column = list(set(dataset[column]))
                unique_values_in_column.sort()

                if not isinstance(unique_values_in_column[0], str):
                    possible_splits = (np.array(unique_values_in_column[:-1]) +
                                       np.array(unique_values_in_column[1:])) / 2
                else:
                    possible_splits = get_categorical_splits(unique_values_in_column)

                # For each binary split
                for split in possible_splits:
                    set1, set2 = divide_set(dataset, column, split)
                    gini1 = gini_impurity(set1)
                    gini2 = gini_impurity(set2)
                    impurity_set = (len(set1)/len(dataset)) * gini1 + (len(set2)/len(dataset)) * gini2

                    if impurity_set < best_impurity and len(set1) > 0 and len(set2) > 0:
                        best_attribute = column
                        right_values = split[1] if isinstance(split, tuple) else split
                        best_sets = (set1, set2)
                        best_impurity = impurity_set
                        keep_expanding = True

        if keep_expanding:
            self._features_chose[str(best_attribute)] += 1
            self._splits_done += 1
            left_node = self._expand_binary_tree(best_sets[0])
            right_node = self._expand_binary_tree(best_sets[1])
            return Node(left_node, right_node, best_attribute, right_values, leaf_results)
        else:
            leaf_results = count_classes_in_dataset(dataset)
            return Node(None, None, None, right_values, leaf_results)

    def calculate_feature_importance(self):
        feature_importance = {}
        for key, value in self._features_chose.items():
            feature_importance[key] = value / self._splits_done

        return feature_importance

    @property
    def features_chose(self):
        return self._features_chose
