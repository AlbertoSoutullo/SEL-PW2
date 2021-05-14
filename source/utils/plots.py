import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

test = pd.read_csv("../logs/RandomForest_kr-vs-kp.csv")
test = test.iloc[:, 1:]
test["Accuracy"] = test["Accuracy"].round(decimals=3)

test = test.loc[test["F"] != "runif"]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=test["NT"], ys=pd.to_numeric(test["F"]), zs=test["Accuracy"],
           linewidths=1, alpha=.7,
           edgecolor='k',
           s = 200,
           c=np.array(test["Accuracy"]),
           cmap='viridis')

ax.set_xlabel("NT", fontsize=15)
ax.set_ylabel("F", fontsize=15)
ax.set_zlabel("Accuracy", fontsize=15)
ax.set_title("Random Forest - KR-VS-KP")
plt.show()
