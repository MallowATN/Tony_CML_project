import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

#Read in file
X_train = np.genfromtxt("src/data/train_model_data/train_features.csv")
y_train = np.genfromtxt("src/data/train_model_data/train_labels.csv")
X_test = np.genfromtxt("src/data/test_model_data/test_features.csv")
y_test = np.genfromtxt("src/data/test_model_data/test_labels.csv")

# Fit in the model
n_estimator = 150
pipe = Pipeline([('std_scaler', StandardScaler()),
                 ('GBR',GradientBoostingRegressor(n_estimators=n_estimator))])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
r2 = pipe.score(X_test,y_test)
# print(r2)

#save metric in text file
with open("src/data/metrics.txt","w") as outfile:
    outfile.write("r2_score: " + str(r2)+"\n")

