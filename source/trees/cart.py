import random
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from source.data_structure.node import Node
from source.utils.utils import get_categorical_splits, divide_set, gini_impurity, count_classes_in_dataset


class Cart:
    """
    Base learner for the forest methods.
    In features attribute chose we will get track of the features used to split nodes.
    In classifications attribute we will save the classifications of this tree when performing the prediction.
    """
    def __init__(self, attributes_to_use: int = -1, seed: int = 0):
        """
        :param attributes_to_use: Number of attributes to use. If default (-1), all will be selected, otherwise,
        a subsample with this number of attributes will be selected to split this node. This is used in random forest.
        Decision forest uses all attributes.
        :param seed: Seed for reproduce results
        """
        self._root = None
        self._features_chose = {}
        self._classifications = []
        self._attributes_to_use = attributes_to_use
        random.seed(seed)

    def fit(self, dataset: pd.DataFrame):
        """
        Fits the tree.
        :param dataset: Pandas Dataframe used.
        :return: None
        """
        for column in dataset.columns[:-1]:
            self._features_chose[str(column)] = 0

        self._root = self._expand_binary_tree(dataset)

    def classify(self, dataset: pd.DataFrame) -> List:
        """
        Classifies a dataframe
        :param dataset: Dataframe to classify
        :return: List of classifications
        """
        self._classifications.clear()

        for i, row in dataset.iterrows():
            possible_classes = self._recursive_classification(row, self._root)
            final_classification = max(possible_classes, key=possible_classes.get)
            self._classifications.append(final_classification)

        return self._classifications

    def _recursive_classification(self, row: pd.Series, node: Node) -> Dict:
        """
        Recursively classifies an instance exploring the nodes of the tree.
        :param row: Instance to classify
        :param node: Node to keep searching
        :return: Dict with the classes and the counts of them.
        """
        if node.leaf_results is not None:
            return node.leaf_results
        else:
            next_node = self._get_next_node_to_go(row, node)

        return self._recursive_classification(row, next_node)

    def _get_next_node_to_go(self, row, current_node: Node):
        """
        Get next node depending on the data it was trained with.
        :param row: Instance to classify
        :param current_node: Node to keep searching
        :return: The next node to search
        """
        v = row[current_node.best_attribute]
        if isinstance(v, str):
            if isinstance(current_node.right_values, tuple):
                list_values = current_node.right_values
            else:
                list_values = [current_node.right_values]
            if v in list_values:
                next_node = current_node.right_node
            else:
                next_node = current_node.left_node
        else:
            if v >= current_node.right_values:
                next_node = current_node.left_node
            else:
                next_node = current_node.right_node

        return next_node

    def _expand_binary_tree(self, dataset: pd.DataFrame) -> Node:
        """
        Recursive function that expands the tree to fit it.
        :param dataset: Dataset to use.
        :return: The root of the tree
        """
        if len(dataset) == 0:
            return Node()

        best_attribute = None
        right_values = None
        leaf_results = None
        best_sets = None
        best_impurity = np.inf
        keep_expanding = False

        subsample_attributes = self._subsample_attributes(dataset)
        current_impurity = gini_impurity(dataset)

        # If not all classes are the same
        if current_impurity != 0:
            best_attribute, right_values, best_sets, best_impurity, keep_expanding = \
                self._calculate_impurities_of_attributes(subsample_attributes, dataset,
                                                         best_attribute, right_values, best_sets,
                                                         best_impurity, keep_expanding)

        if keep_expanding:
            self._features_chose[str(best_attribute)] += 1
            left_node = self._expand_binary_tree(best_sets[0])
            right_node = self._expand_binary_tree(best_sets[1])
            return Node(left_node, right_node, best_attribute, right_values, leaf_results)
        else:
            leaf_results = count_classes_in_dataset(dataset)
            return Node(None, None, None, right_values, leaf_results)

    def _subsample_attributes(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        It subsample the dataset columns if it is required.
        :param dataset: Dataset to be subsample
        :return: Subsample of the dataset
        """
        if self._attributes_to_use == -1:
            subsample_attributes = dataset.columns[:-1]
        else:
            subsample_attributes = random.sample(list(dataset.columns[:-1]), self._attributes_to_use)

        return subsample_attributes

    def _calculate_impurities_of_attributes(self, subsample_attributes, dataset: pd.DataFrame,
                                            best_attribute, right_values, best_sets, best_impurity, keep_expanding):
        """
        It calculates the impurities of the binary combination of the attributes.
        :param subsample_attributes: Columns to explore.
        :param dataset: Data to explore.
        :param best_attribute: Parameters to get track of the situation.
        :param right_values: Parameters to get track of the situation.
        :param best_sets: Parameters to get track of the situation.
        :param best_impurity: Parameters to get track of the situation.
        :param keep_expanding: Parameters to get track of the situation.
        :return:  Parameters to get track of the situation.
        """
        for column in subsample_attributes:
            unique_values_in_column = list(set(dataset[column]))
            unique_values_in_column.sort()

            # Get the middle points, or the combinations of categorical parameters for explore
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
                impurity_set = (len(set1) / len(dataset)) * gini1 + (len(set2) / len(dataset)) * gini2

                if impurity_set < best_impurity and len(set1) > 0 and len(set2) > 0:
                    best_attribute = column
                    right_values = split[1] if isinstance(split, tuple) else split
                    best_sets = (set1, set2)
                    best_impurity = impurity_set
                    keep_expanding = True

        return best_attribute, right_values, best_sets, best_impurity, keep_expanding

    @property
    def features_chose(self):
        return self._features_chose
