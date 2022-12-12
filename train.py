import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

#Read in file
X_train = np.genfromtxt("src/data/train_model_data/train_features.csv")
y_train = np.genfromtxt("src/data/train_model_data/train_labels.csv")
X_test = np.genfromtxt("src/data/test_model_data/test_features.csv")
y_test = np.genfromtxt("src/data/test_model_data/test_labels.csv")

# Create pipeline and fit the model
n_estimator = 150
pipe = Pipeline([('std_scaler', StandardScaler()),
                 ('GBR',GradientBoostingRegressor(n_estimators=n_estimator))])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

#Report training/test set r squared scores
train_score = pipe.score(X_train, y_train)*100
test_score = pipe.score(X_test,y_test)*100

#save metric in text file
with open("src/data/metrics.txt","w") as outfile:
    outfile.write("training variance explained: %2.1f%%\n" % train_score)
    outfile.write("test variance explained: %2.1f%%\n" % test_score)


### Plot Feature Importance 
# print(pipe.steps[1][1].feature_importances_)
df = pd.read_csv('src/data/Processed_Account_Sales.csv')
importances = pipe.steps[1][1].feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels,importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False)

# image formatting
axis_fs = 18
title_fs = 22
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance', fontsize=axis_fs)
ax.set_ylabel('Feature', fontsize=axis_fs)
ax.set_title('GBR\nfeature importance', fontsize=title_fs)

plt.tight_layout()
plt.savefig('feat_importance.png')
plt.close()
### Plot Residuals

y_pred = pipe.predict(X_test) + np.random.normal(0,0.25,len(y_test))
y_jitter = y_test + np.random.normal(0,0.25, len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter, y_pred)), columns = ['true','pred'])

ax = sns.scatterplot(x='true',y='pred',data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True Total Revenue', fontsize=axis_fs)
ax.set_ylabel('Predicted Total Revenue', fontsize=axis_fs)
ax.set_title('Residuals',fontsize=title_fs)

#square aspect ratio or line if needed
ax.plot([1,8000000], [1,8000000], 'black', linewidth=1)
plt.ylim((0,950000))
plt.xlim((0,950000))

plt.tight_layout()
plt.savefig('residuals.png',dpi=120)
plt.close()