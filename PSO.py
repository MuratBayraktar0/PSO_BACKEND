from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
import pandas as pd
import pyswarms as ps
from sklearn import preprocessing
import app
import json
import csv
from sklearn.preprocessing import StandardScaler
import os
from sklearn.datasets import make_classification

'''X, y = make_classification(n_samples=100, n_features=15, n_classes=3,
                           n_informative=4, n_redundant=1, n_repeated=2,
                           random_state=1)'''

#THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
nameQ = app.request.args.get('name')
iterationQ = app.request.args.get('iteration')
classiferQ = app.request.args.get('classifer')
particleQ = app.request.args.get('particle')

context = app.request.get_json('body')

# the result is a Python dictionary:
data_file = open('upload/'+nameQ, 'w', newline='')
csv_writer = csv.writer(data_file)
 
count = 0
for data in context:
    if count == 0:
        header = data.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(data.values())  

data_file.close()


print(nameQ,iterationQ,classiferQ,particleQ)

dataset = pd.read_csv("upload/"+nameQ)

# print(dataset.shape)
X = dataset.iloc[:, 1:(dataset.shape[1] - 1)].values
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
y = dataset.iloc[:, -1].values


df = pd.DataFrame(X)
df['labels'] = pd.Series(y)
# print(df)
# sns.pairplot(df, hue='labels')
# -------------------------------------------------------------------------------

if int(classiferQ) == 1:
    classifier = KNeighborsClassifier(n_neighbors=3)
else:
    from sklearn import linear_model

    classifier = linear_model.LogisticRegression()

"""from sklearn import linear_model
classifier = linear_model.LogisticRegression()"""

classifier = KNeighborsClassifier(n_neighbors=3)


# Define objective function
def f_per_particle(m, alpha):
    total_features = X.shape[1]
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:, m == 1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
         + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j


# -----------------------------------------------------------------------
# x is population
def fitness(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


# ===============================================================================
options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}

# Call instance of PSO
dimensions = X.shape[1]  # dimensions should be the number of features
# optimizer.reset()
optimizer = ps.discrete.BinaryPSO(
    n_particles=int(particleQ), dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(fitness, iters=int(iterationQ))
oldFeature = X.shape[1]
newFeature = sum(pos[pos == 1])
print(newFeature, "from", oldFeature)
# ===================================================================================
# Create two instances of LogisticRegression
#classfier = linear_model.LogisticRegression()

# Get the selected features from the final positions
X_selected_features = X[:, pos == 1]  # subset

# Perform classification and store performance in P
c1 = classifier.fit(X_selected_features, y)

# Compute performance
subset_performance = (c1.predict(X_selected_features) == y).mean()
print('Performance:', pos)
print('Subset performance: %.3f' % (subset_performance))

def csvGetKey(keys,poss):
    i = 0
    newHeaders = []
    for pos in poss:
        if pos == 1:
            newHeaders.append(keys[i])
        i += 1
    return newHeaders

def cvsToJson(fetures,classes):
    data = []
    newHeaders = csvGetKey(list(context[0]),pos)
    j = 0
    for feture in fetures:
        k = 0
        row = {}
        for fetur in feture:
            if isinstance(fetur, int):
                row[str(newHeaders[k])] = int(fetur)
            if isinstance(fetur, np.int32):
                row[str(newHeaders[k])] = np.int32(fetur)
            if isinstance(fetur, np.int64):
                row[str(newHeaders[k])] = np.int64(fetur)
            if isinstance(fetur, np.float_):
                row[str(newHeaders[k])] = np.float_(fetur)
            elif isinstance(fetur, str):
                row[str(newHeaders[k])] = str(fetur)
            k += 1
        if isinstance(fetur, int):
            row['class'] = int(classes[j])
        if isinstance(fetur, np.int32):
            row['class'] = np.int32(classes[j])
        if isinstance(fetur, np.int64):
            row['class'] = np.int64(classes[j])
        if isinstance(fetur, np.float_):
            row['class'] = np.float_(fetur)
        elif isinstance(fetur, str):
            row['class'] = str(classes[j])
        data.append(row)
        j += 1
    return data

newData = cvsToJson(X_selected_features, y)
#newData = np.concatenate((X_selected_features, y))
#np.savetxt('../../Desktop/tokyo/newDataset/new'+nameQ,X_selected_features, delimiter=',')