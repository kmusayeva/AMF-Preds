import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel("/user/kmusayev/home/PyProject/AMF-Preds/soil_microbiome/data/GlobalAMFungi/all.xlsx")

df = df.query('paper_id=="Rezacova_2021_1HU"')

Y = df.iloc[:, 17:df.shape[1]]

tmp = pd.get_dummies(df['Biome'])

df = pd.concat([tmp, df], axis=1)

Y_bin = Y.apply(lambda y: [u if u==0 else 1 for u in y])

species = Y.columns.tolist()

#X_plot = df[Y[species]==1][env_vars]

env_vars = df['Biome'].unique().tolist()+["pH", "MAP", "MAT"]

X = df[env_vars]
#X = df[env_vars].values

#X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

param_grid_knn = {'n_neighbors': range(1,11)}


species = Y.columns[(Y_bin.sum()/len(Y)).round(2)>=0.01].tolist()

from sklearn.multioutput import MultiOutputClassifier


knn = KNeighborsClassifier()

# Create MultiOutputClassifier wrapper
multi_knn = MultiOutputClassifier(knn)

# Define parameter grid for grid search
param_grid = {
    'estimator__n_neighbors': [3, 5, 7],  # Number of neighbors to consider
    'estimator__weights': ['uniform', 'distance'],  # Weighting scheme
    'estimator__metric': ['euclidean', 'manhattan']  # Distance metric
}

# Perform grid search
grid_search = GridSearchCV(multi_knn, param_grid, cv=5)
grid_search.fit(X, Y_bin[species])

# Best parameters found during grid search
print("Best parameters:", grid_search.best_params_)

# Predict on the test set using the best model
print(grid_search.predict(X))



#knnm = []
#
#for s in species:
#
#    y = Y_bin[s]
#    
#    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
#
#    grid_search = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')
#
#    # Fit the grid search to the data
#    grid_search.fit(X, y)
#
#    # Print the best parameter found
#    #print("Best parameter:", grid_search.best_params_)
#
#    # Print the best score found
#    print("Species %s's best score is %.2f." % (s,grid_search.best_score_))
#    knnm.append(grid_search.best_score_)
#
#
#
#
#R = {'best_score':[],'best_param':[]}
#for s in species:
#
#    y = Y_bin[s]
#    
#
#    param_grid_dt = {'max_depth': range(1, 20), 'min_samples_split': range(2, 11)}
#
#    # Create a Decision Tree classifier
#    decision_tree = DecisionTreeClassifier()
#
#    # Perform grid search using 5-fold cross-validation and accuracy as the scoring metric
#    grid_search = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='accuracy')
#
#    # Fit the grid search to the data
#    grid_search.fit(X, y)
#
#    # Print the best parameters found
#    #print("Best parameters:", grid_search.best_params_)
#    print("Species %s's best score is %.2f." % (s,grid_search.best_score_))
#    R['best_score'].append(grid_search.best_score_)
#    R['best_param'].append(grid_search.best_params_)
#    # Print the best score found
#    #print("Best score:", grid_search.best_score_)
#

#A = pd.DataFrame({"freq":(Y_bin[species].sum()/len(Y)).round(2)})
#A.reset_index(inplace=True)
#B = pd.DataFrame(R)
#pd.concat([A,B], axis=1) 
#






