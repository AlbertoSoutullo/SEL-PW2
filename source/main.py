import json
import pandas as pd
from sys import argv
from source.forest_interpreter import ForestInterpreter
from source.utils import calculate_fs

interpreter_configuration = argv[1]

with open(interpreter_configuration) as json_file:
    config = json.load(json_file)

dataset = pd.read_csv(config["dataset"], header=None)
classifier = config["classifier"]
if "F" in config.keys():
    fs = config["F"]
else:
    fs = calculate_fs(classifier, dataset)

if fs is None:
    print(f"Classifier {classifier} not supported.")
    exit()

train = dataset.sample(frac=config["train_test_split"], random_state=config["seed"])
test = dataset.drop(train.index)
ground_truth = test.iloc[:, -1:].iloc[:, 0].values.tolist()
test = test.iloc[:, :-1]

forest_interpreter = ForestInterpreter(classifier,
                                       train,
                                       config["NT"],
                                       fs,
                                       config["csv_out_name"],
                                       config["seed"])

forest_interpreter.interpret(test, ground_truth)
