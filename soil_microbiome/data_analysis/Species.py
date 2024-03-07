import numpy as np
import pandas as pd
import pymc3 as pm
import os
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from .. import global_vars
from ..utils import *

class Species:
    def __init__(self, level_name, file_name, env_vars, x_dim):
        self.__level = level_name
        self.__file_name = file_name
        self.__data = None
        self.__env_vars = env_vars
        self.__x_dim = x_dim 
        self.__y_dim = None 
        self.__X = None
        self.__Y = None


    def set_data(self):
        try:
            self.__data = read_file(self.__file_name)
        except ValueError as err:
            print("errro")

        self.__X = self.__data[self.__env_vars]
        start_y = self.__x_dim-1
        end_y = self.__data.shape[1] 
        self.__Y = self.__data.iloc[:, start_y:end_y]
        self.__y_dim = self.__data.shape[1] - self.__x_dim + 1
    

    def get_data(self, index=None):
        if index:
            return self.__X.iloc[index,:], self.__Y.iloc[index,:]
        return self.__X, self.__Y


    def get_norm_env_vars(self):
        mean_values = self.__X.mean()
        std_values = self.__X.std()
        return (self.__X - mean_values) / std_values


    def get_min_max_env_vars(self):
        scaler = MinMaxScaler()
        scaler.fit(self.__X)
        return pd.DataFrame(scaler.transform(self.__X))


    def get_attributes(self):
        return self.__level, self.__x_dim, self.__y_dim


    def print_data(self):
        print(f'X is {self.__X}')
        print(f'Y is {self.__Y}')



    def get_description(self):
        print(f'The taxonomic level: {self.__level}')
        print(f'Input dimension: {self.__x_dim}')
        print(f'Output dimension: {self.__y_dim}')
                


if __name__ == "__main__":
    env_vars = ['pH', 'MO']
    species = Species('Ordre', 'Ordre/sol_Ordre.xlsx', env_vars, x_dim=19)
    species.set_data()
    species.get_description()
    taxon, input_dim, output_dim = species.get_attributes()
    print(f'Taxon: ', taxon)


#        #cols = ['pH', 'MO', 'Sables_fins', 'Sables_grossiers', 'Limons_fins', 'Limons_grossiers', 'CEC', 'K2O']
#
#        #X = df.iloc[:,2:18]
#        X = df[cols]
#        Y = df.iloc[:,18:df.shape[1]]
#
#        n = X.shape[0]
#        C = Y.shape[1]
#        #Y = Y.iloc[:,0:-1]
#
#        Y = Y.apply(lambda y: (y * (n - 1) + 1 / C) / n)
#
#        # Define your data
#        #X = np.random.randn(40, 2)
#        #Y = np.random.dirichlet((10, 5, 3), 40)
#
#        #X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
#
#        X_with_intercept = pd.concat([pd.DataFrame({'Int':np.ones(n)}),X], axis=1)
#
#
#        # Define the PyMC3 model
#        with pm.Model() as model:
#            # Priors for coefficients
#            beta = pm.Normal('beta', mu=0, sigma=0.5, shape=(X_with_intercept.shape[1], Y.shape[1]))
#            
#            # Likelihood function (Dirichlet distribution)
#            theta = pm.Deterministic('theta', pm.math.dot(X_with_intercept, beta))
#
#            mu = pm.Deterministic("mu", pm.math.exp(theta))
#
#            Y_obs = pm.Dirichlet('Y_obs', mu, observed=Y)
#            
#
#
#        # Sample from the posterior distribution
#        with model:
#            trace = pm.sample(2000, tune=1000, cores=2, target_accept=0.9)
#
#
#        fitted_values = trace['mu'][-1000:].mean(axis=0)
#
#        Yhat = fitted_values/fitted_values.sum(axis=1).reshape(-1,1)
#
#        print(np.sqrt(((Y-Yhat)**2).mean(axis=0)))
#
#        r_squared = r2_score(Y, Yhat, multioutput='raw_values')
#
#        # Compute adjusted R-squared for each dimension
#        n = len(Y)
#        p = X_with_intercept.shape[1]  # Assuming X is your feature matrix
#        adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
#
#        # If you want to get the mean adjusted R-squared across dimensions
#        mean_adjusted_r_squared = adjusted_r_squared.mean()
#
#        print("R-squared for each dimension:", r_squared)
#
#        print("Adjusted R-squared for each dimension:", adjusted_r_squared)
#
#        print("Mean Adjusted R-squared:", mean_adjusted_r_squared)
