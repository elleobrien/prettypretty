from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score
import json
import os
import numpy as np
from sklearn import preprocessing
import pandas as pd
# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# Fit a model
depth = 5
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train,y_train)

# Get overall accuracy
acc = clf.score(X_test, y_test)

# Get precision and recall
y_score = clf.predict(X_test)
#roc_auc = roc_auc_score(y_test, y_score)
#print(roc_auc)

# Outs for a confusion matrix
d = {'actual':y_test, 'predicted':y_score}
df = pd.DataFrame(d)
df.to_csv("classes.csv", index=False)

# Look at dependence on number of estimators
min_estimators = 15
max_estimators = 30

n_estimators = []
score = []

for i in range(min_estimators, max_estimators +1):
        clf.set_params(n_estimators=i)
        clf.fit(X_train,y_train)

        # Recoord the score on the test data
        test_score = clf.score(X_test,y_test)
        n_estimators.append(i)
        score.append(test_score)

out = {'n_estimators':n_estimators,'test score':score}
df = pd.DataFrame(out)
df.to_csv("estimators.csv",index=False)


