import random
from typing import Dict
import pandas as pd
from source.cart import Cart


class DecisionForestClassifier:

    def __init__(self, number_of_trees: int, num_random_features: int, runif: bool = False, seed: int = 0):
        self._NT = number_of_trees
        self._F = num_random_features
        self._trees = []
        self._features = {}
        self._classifications = []
        self._runif = runif
        random.seed(seed)

    def fit(self, dataset):

        for i in range(self._NT):
            classes = dataset.iloc[:, -1:]

            if self._runif:
                number_or_random_features = random.randint(1, len(dataset.columns))
            else:
                number_or_random_features = self._F

            random_sample = dataset.iloc[:, :-1].sample(n=number_or_random_features,
                                                        random_state=random.randint(0, 100000), axis=1)
            random_sample_complete = pd.concat([random_sample, classes], axis=1)
            cart_tree = Cart(seed=random.randint(0, 100000))
            cart_tree.fit(random_sample_complete)
            self._update_features(cart_tree._feature_selecteds)
            self._trees.append(cart_tree)

    def predict(self, dataset):
        for tree in self._trees:
            classifications = tree.classify(dataset)
            self._classifications.append(classifications)

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

    def extract_features(self):
        return list(sorted(self._features.items(), key=lambda x: x[1], reverse=True))

    def _update_features(self, new_features: Dict):
        for key, value in new_features.items():
            if key in self._features.keys():
                self._features[key] += value
            else:
                self._features[key] = value