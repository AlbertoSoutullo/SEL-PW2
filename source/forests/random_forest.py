import random
from source.forests.forest_classifier import ForestClassifier
from source.trees.cart import Cart


class RandomForestClassifier(ForestClassifier):

    def __init__(self, number_of_trees: int, num_random_features: int, random_partition: float = 0.75, seed: int = 0):
        super().__init__(number_of_trees, num_random_features, seed)
        self._random_partition = random_partition

    def fit(self, dataset):
        for i in range(self._NT):
            random_sample = dataset.sample(frac=self._random_partition, random_state=random.randint(0, 100000))
            cart_tree = Cart(self._F, random.randint(0, 100000))
            cart_tree.fit(random_sample)
            self._update_features(cart_tree.features_chose)
            self._trees.append(cart_tree)
