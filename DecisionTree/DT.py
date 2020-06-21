#importing libraries
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

#loading dataset
df=pd.read_csv("C:/Users/havish/Desktop/iris.csv")

#splitting the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25)

#creating a class node for the tree which we can use for n-depth tree
class Node:
def __init__(self, predicted_class):
self.predicted_class = predicted_class
self.feature_index = 0
self.threshold = 0
self.left = None
self.right = None

#creating DecisionTree class
class DecisionTreeClassifier:
def __init__(self, max_depth=None):
self.max_depth = max_depth

#Fitting the model
def fit(self, X, y):
self.n_classes_ = len(set(y))
self.n_features_ = X.shape[1]
self.tree_ = self._grow_tree(X, y)

#Method for predicting 
def predict(self, X):
return [self._predict(inputs) for inputs in X]

#implementing CART algorithm for finding the gini impurity by best split
def _best_split(self, X, y):
m = y.size
if m &lt;= 1:
return None, None
num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
best_idx, best_thr = None, None
for idx in range(self.n_features_):
thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
num_left = [0] * self.n_classes_
num_right = num_parent.copy()
for i in range(1, m):
c = classes[i - 1]
num_left[c] += 1
num_right[c] -= 1
gini_left = 1.0 - sum(
(num_left[x] / i) ** 2 for x in range(self.n_classes_)
)
gini_right = 1.0 - sum(
(num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
)
gini = (i * gini_left + (m - i) * gini_right) / m
if thresholds[i] == thresholds[i - 1]:
continue
if gini &lt; best_gini:

best_gini = gini
best_idx = idx
best_thr = (thresholds[i] + thresholds[i - 1]) / 2
return best_idx, best_thr

#recurssively finding the bestsplit and building the decision tree
def _grow_tree(self, X, y, depth=0):
num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
predicted_class = np.argmax(num_samples_per_class)
node = Node(predicted_class=predicted_class)
if depth &lt; self.max_depth:
idx, thr = self._best_split(X, y)
if idx is not None:
indices_left = X[:, idx] &lt; thr
X_left, y_left = X[indices_left], y[indices_left]
X_right, y_right = X[~indices_left], y[~indices_left]
node.feature_index = idx
node.threshold = thr
node.left = self._grow_tree(X_left, y_left, depth + 1)
node.right = self._grow_tree(X_right, y_right, depth + 1)
return node

#method for predicting
def _predict(self, inputs):
node = self.tree_
while node.left:
if inputs[node.feature_index] &lt; node.threshold:

node = node.left
else:
node = node.right
return node.predicted_class

#fit the data to the classifier and predicting
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, Y_train)
clf.predict(X_test, Y_test)
