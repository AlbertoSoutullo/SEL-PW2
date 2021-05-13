import pandas as pd

from source.decision_forest import DecisionForestClassifier
from source.random_forest import RandomForestClassifier


class ForestInterpreter:

    def __init__(self, forest, dataset, NT, F, name_to_export_csv, seed=0):
        self._forest_name = forest
        self._forest_instance = None
        self._dataset = dataset
        self._NT = NT
        self._F = F
        self._seed = seed
        self._name_to_export_csv = name_to_export_csv
        self._results = pd.DataFrame()

    def interpret(self, test, ground_truth):
        for number_of_trees in self._NT:
            for number_of_features in self._F:
                self._forest_instance = self._forest_selection(number_of_trees, number_of_features)

                print(f"Fitting with {self._forest_name} with NT={number_of_trees} and F={number_of_features}...")
                self._forest_instance.fit(self._dataset)

                self._prediction(number_of_trees, number_of_features, test, ground_truth)

        self._results.to_csv(f"logs/{self._name_to_export_csv}")

    def _forest_selection(self, number_of_trees, number_of_features):
        if self._forest_name == "RandomForestClassifier":
            forest_instance = RandomForestClassifier(number_of_trees, number_of_features, seed=self._seed)
        elif self._forest_name == "DecisionForestClassifier":
            if number_of_features == "runif":
                forest_instance = DecisionForestClassifier(number_of_trees, -1, True, self._seed)
            else:
                forest_instance = DecisionForestClassifier(number_of_trees, number_of_features, seed=self._seed)
        else:
            print("Not a valid classifier selected")
            forest_instance = None

        return forest_instance

    def _prediction(self, number_of_trees, number_of_features, test, ground_truth):
        info = f"Classifying with {self._forest_name} with NT={number_of_trees} and F={number_of_features}..."
        print(info)

        predictions = self._forest_instance.predict(test)
        feature_importance = self._forest_instance.extract_features()

        accuracy = len([ground_truth[i] for i in range(0, len(ground_truth))
                        if ground_truth[i] == predictions[i]]) / len(ground_truth)

        self._log_results(number_of_trees, number_of_features, info, feature_importance, accuracy)

    def _log_results(self, number_of_trees, number_of_features, info, feature_importance, accuracy):
        result = pd.DataFrame({"Method": self._forest_name, "NT": number_of_trees,
                               "F": number_of_features, "Accuracy": accuracy}, index=[0])

        self._results = pd.concat([self._results, result])

        with open(f"logs/{self._name_to_export_csv}_NT-{number_of_trees}_F-{number_of_features}", "w") as file:
            file.write(info+"\n")
            feature_importance_info = f"Feature importance -> ('feature': number of appearances)"
            print(feature_importance_info)
            file.write(feature_importance_info+"\n")
            print(feature_importance)
            file.writelines(str(feature_importance)+"\n")
            accuracy = f"Accuracy: {accuracy:.3f}\n\n"
            print(accuracy)
            file.write(accuracy)
