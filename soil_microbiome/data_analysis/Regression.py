import numpy as np
import pandas as pd
import pymc3 as pm
import os
from sklearn.metrics import r2_score

from .. import global_vars
from ..utils import *



class Regression:
    
    def __init__(self, X, Y, params=None):
        self.__X = X
        self.__Y = Y
        self.__n = X.shape[0]
        self.__p = X.shape[1]
        self.__m = Y.shape[1]
        self.__params = params
        self.__preds = None
        self.__Yhat = None
        self.__adjusted_rscore = None
        self.__rmse = None


    def get_attributes(self):
        return self.__X, self.__Y, self.__params


    #for now do only fitted values
    def do_Dirichlet(self):
          mu = self.__params['mu'] if 'mu' in self.__params else 0.0
          sigma = self.__params['sigma'] if 'sigma' in self.__params else 1.0
          cores = self.__params['cores'] if 'cores' in self.__params else 2
          #Create design matrix
          self.__X = pd.concat([pd.DataFrame({'Intercept': np.ones(self.__n)}), self.__X], axis=1)
          #We need to transform from [0,1] to (0,1)
          Y_mod = self.__Y.apply(lambda y: (y * (self.__n-1)+1/self.__m)/self.__n)

          #Define the PyMC3 model
          with pm.Model() as model:
              
              # Priors for coefficients
              beta = pm.Normal('beta', mu=mu, sigma=sigma, shape=(self.__X.shape[1], self.__m))

              # Likelihood function (Dirichlet distribution)
              theta = pm.Deterministic('theta', pm.math.dot(self.__X, beta))

              mu = pm.Deterministic("mu", pm.math.exp(theta))

              Y_obs = pm.Dirichlet('Y_obs', mu, observed=Y_mod)

          #Sample from the posterior distribution
          with model:
            #try:
                  trace = pm.sample(3000, tune=1000, cores=cores, target_accept=0.9)
            #except:
            #    print("Error in sampling!")
            #    os.sys.exit(1)

          raw_mu = trace['mu'][-2000:].mean(axis=0)

          self.__Yhat = raw_mu/raw_mu.sum(axis=1).reshape(-1,1)

            
    
    def eval_perf(self): 
          r_squared = r2_score(self.__Y, self.__Yhat, multioutput='raw_values')

          #Compute adjusted R-squared for each dimension
          self.__adjusted_rscore = 1 - ((1 - r_squared) * (self.__n - 1) / (self.__n - self.__p - 1))

          #If you want to get the mean adjusted R-squared across dimensions
          #mean_adjusted_r_squared = adjusted_r_squared.mean()

          self.__rmse = np.sqrt(((self.__Y-self.__Yhat)**2).mean(axis=0))

          # print("R-squared for each dimension:", r_squared)

          # print("Adjusted R-squared for each dimension:", adjusted_r_squared)

          # print("Mean Adjusted R-squared:", mean_adjusted_r_squared)


    def get_perf(self):
        return self.__rmse, self.__adjusted_rscore


