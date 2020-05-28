from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
clf = MLPClassifier(hidden_layer_sizes=(100,))
clf.fit(X_train,y_train)

# Get overall accuracy
acc = clf.score(X_test, y_test)
acc = round(acc, 3)

# Get AUC
y_prob = clf.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovo")
roc_auc = round(roc_auc,3)

# Write metrics
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "AUC": roc_auc}, outfile)

# Outs for a confusion matrix
y_pred = clf.predict(X_test)
d = {'actual':y_test, 'predicted':y_pred}
df = pd.DataFrame(d)
df.to_csv("classes.csv", index=False)

# Look at dependence on regularizer
alphas = np.logspace(-5, 3, 5)

regularizer= []
score = []

for i in alphas:
        clf.set_params(alpha=i)
        clf.fit(X_train,y_train)

        # Recoord the score on the test data
        test_score = clf.score(X_test,y_test)
        regularizer.append(i)
        score.append(test_score)

out = {'regularization' :regularizer,'test score':score}
df = pd.DataFrame(out)
df.to_csv("estimators.csv",index=False)

