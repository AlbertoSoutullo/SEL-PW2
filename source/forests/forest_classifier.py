import random
import pandas as pd
from typing import Dict, List, Tuple


class ForestClassifier:
    """
    Based class of a Forest Classifier. It has the required parameters to work for any Forest Classifier.
    It has a tree parameter. List. All trees will be saved there.
    In features, the features importance will be saved.
    Classification is a List that will hold the final classification of each instance when we want to predict.
    """
    def __init__(self, number_of_trees: int, num_random_features: int, seed: int = 0):
        """
        :param number_of_trees: Number of CART tress that will be created.
        :param num_random_features: Number of random features that will be used.
        :param seed: Seed for reproduce results.
        """
        self._NT = number_of_trees
        self._F = num_random_features
        self._trees = []
        self._features = {}
        self._classifications = []
        random.seed(seed)

    def predict(self, dataset: pd.DataFrame):
        """
        It performs a classification of the given data.
        :param dataset: Pandas dataset to classify.
        :return: None
        """
        for tree in self._trees:
            classifications = tree.classify(dataset)
            self._classifications.append(classifications)

        final_classifications = self._perform_voting(dataset)

        return final_classifications

    def extract_features(self) -> List[Tuple]:
        """
        Extract an ordered (by importance) set of features.
        :return: List with the features like Tuples ('column index', 'importance')
        """
        return list(sorted(self._features.items(), key=lambda x: x[1], reverse=True))

    def _update_features(self, new_features: Dict):
        """
        Add the given features from a tree to the whole register of features used in the forest.
        :param new_features: New set of features that has to be added to features.
        :return: None
        """
        for key, value in new_features.items():
            if key in self._features.keys():
                self._features[key] += value
            else:
                self._features[key] = value

    def _perform_voting(self, dataset: pd.DataFrame) -> List:
        """
        Classifies the given dataset perform the majority voting.
        :param dataset: Data to classify.
        :return: List of final classifications
        """
        final_classifications = []
        for i in range(len(dataset)):
            current_instance_predicted_classes = {}
            for j in range(self._NT):
                tree_prediction = self._classifications[j][i]
                if tree_prediction in current_instance_predicted_classes.keys():
                    current_instance_predicted_classes[tree_prediction] += 1
                else:
                    current_instance_predicted_classes[tree_prediction] = 1
            class_decision = max(current_instance_predicted_classes, key=current_instance_predicted_classes.get)

            final_classifications.append(class_decision)

        return final_classifications

    def fit(self, _dataset):
        pass
