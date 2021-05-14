# SEL PW2

## Code execution

This project executes a home-made Random Forest or Decision Forest classifier. The datasets available 
to test the code are in `Data` folder. A more in depth study of the algorithms is found in the Report.pdf,
inside `Documentation` folder.

The project is executed from the file `main.py`. This file requires a json configuration, where we establish
the parameters of the forest interpreter.

An example of the structure of the json configuration is the following:

```json
{
  "classifier" : "RandomForestClassifier",
  "dataset": "../Data/cmc.data",
  "train_test_split": 0.75,
  "NT": [1, 10, 25, 50, 75, 100],
  "F": [1, 2, 3, 4],
  "csv_out_name": "RandomForest_cmc.csv",
  "seed": 0
}
```

The explanation of the parameters is the following:

```
{
  "classifier" : str, -> possible values: "RandomForestClassifier" or "DecisionForestClassifier"
  "dataset": str, -> data path to the dataset
  "train_test_split": float, -> fraction [0, 1] of the dataset used in train, the rest will go on test
  "NT": [int, ...], -> List of ints. Number of trees that will evaluated. 
  "F": [int, ...], -> List of ints. Number of features that will be evaluated.
  "csv_out_name": str, -> name of the .csv output file
  "seed": int -> seed to reproduce results
}
```

So, in order to execute the project, you only need to call the main script with the configuration file, as:

```shell
python main.py config_files/random_forest_cmc.json
```

## Project Structure

* **Data**:  Folder where the datasets used for this project are located.
* **Documentation**: Folder where the report of this project is located.
* **source**: Source folder for the code of the project.
    * **config_files**: JSON configuration used for this project.
    * **data_structure**: Python package. The files for the data structure are located here.
    * **forests**: Python package where the files of forest related classes are located.
    * **logs**: Default folder where the results of the executions are saved.
    * **trees**: Python package where the tree (CART) structure is saved.
    * **utils**: Python package where some utils function are saved, as well as the script to produce the 3D scatter plots.
* **README.md**: This file.