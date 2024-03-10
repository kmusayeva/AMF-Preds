import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


from .. import global_vars
from ..utils import *


class SpeciesClassification:


    def __init__(self, X, Y, cv=5, freq=0.01):
        self.__X = X
        self.__Y_bin = Y.apply(lambda y: [u if u==0 else 1 for u in y])
        self.__n = X.shape[0]
        self.__p = X.shape[1]
        self.__m = Y.shape[1]
        self.__preds = None
        self.__Yhat = None
        self.__cv = cv 
        self.__freq = freq
        self.__R = {'best_score':[],'best_param':[]}
        self.__params = None



    def get_attributes(self):
        return self.__X, self.__Y_bin, self.__params



    def transform(self):
        preprocessing.StandardScaler().fit(self.__X).transform(self.__X.astype(float))



    def select_subset_Y(self):
        cols = self.__Y_bin.columns[(self.__Y_bin.sum()/len(self.__Y_bin)).round(2)>=self.__freq].tolist()
        return self.__Y_bin[cols]



    def _do_classify(self, model, y):

        print(self.__params)

        grid_search = GridSearchCV(model, param_grid=self.__params, cv=self.__cv, scoring='accuracy')

        grid_search.fit(self.__X, y)

        print("Species best score is %.2f." % grid_search.best_score_)

        self.__R['best_score'].append(grid_search.best_score_)

        self.__R['best_param'].append(grid_search.best_params_)




    def do_knn(self, k=10):

        self.__params = {'n_neighbors':range(1,k)}

        species = self.select_subset_Y()

        self.transform()

        for s in species:

            y = self.__Y_bin[s]

            knn_model = KNeighborsClassifier()

            self._do_classify(knn_model, y)





    def do_decision_tree(self, max_depth=20, min_sample=11):
        
        species = self.select_subset_Y()

        self.__params = {'max_depth': range(1, max_depth), 'min_samples_split': range(2, min_sample)}

        for s in species:

            y = self.__Y_bin[s]

            self._do_classify(DecisionTreeClassifier(), y)




    def do_logistic_regression(self):

        species = self.select_subset_Y()

        self.__params = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']}

        for s in species:

            y = self.__Y_bin[s]

            self._do_classify(LogisticRegression(), y)
        


    def get_perf(self):
        return self.__R



if __name__ == "__main__":
    
    df = pd.read_excel("/user/kmusayev/home/PyProject/AMF-Preds/soil_microbiome/data/GlobalAMFungi/all.xlsx")

    print(df)

    df = df.query('paper_id=="Rezacova_2021_1HU"')

    Y = df.iloc[:, 17:df.shape[1]]

    tmp = pd.get_dummies(df['Biome'])

    df = pd.concat([tmp, df], axis=1)

    species = Y.columns.tolist()

    #X_plot = df[Y[species]==1][env_vars]

    env_vars = df['Biome'].unique().tolist()+["pH", "MAP", "MAT"]

    X = df[env_vars]
    #X = df[env_vars].values

    #X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

    model = SpeciesClassification(X, Y, freq=0.5)

    model.do_logistic_regression()

    perf = model.get_perf()    
    
    print(pd.DataFrame(perf))
