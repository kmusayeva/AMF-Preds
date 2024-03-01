#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pymc3 as pm
import os
from sklearn.metrics import r2_score

from .. import global_vars
from ..utils import *


def perform_linear_regression(X, Y):
   pass


def perform_dirichlet_regression(X, Y, params):
      n = len(X)
      C = Y.shape[1]

      #For Dirichlet regression, we need to transform from [0,1] to (0,1)
      Y = Y.apply(lambda y: (y * (n - 1) + 1 / C) / n)
      
      X_with_intercept = pd.concat([pd.DataFrame({'Int':np.ones(n)}),X], axis=1)

      #Define the PyMC3 model
      with pm.Model() as model:
          # Priors for coefficients
          beta = pm.Normal('beta', mu=params['mu'], sigma=params['sigma'], shape=(X_with_intercept.shape[1], Y.shape[1]))
          
          # Likelihood function (Dirichlet distribution)
          theta = pm.Deterministic('theta', pm.math.dot(X_with_intercept, beta))

          mu = pm.Deterministic("mu", pm.math.exp(theta))

          Y_obs = pm.Dirichlet('Y_obs', mu, observed=Y)

      #Sample from the posterior distribution
      with model:
          trace = pm.sample(2000, tune=1000, cores=2, target_accept=0.9)

      fitted_values = trace['mu'][-1000:].mean(axis=0)

      Yhat = fitted_values/fitted_values.sum(axis=1).reshape(-1,1)

      r_squared = r2_score(Y, Yhat, multioutput='raw_values')

      #Compute adjusted R-squared for each dimension
      p = X_with_intercept.shape[1]  
      adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

      #If you want to get the mean adjusted R-squared across dimensions
      #mean_adjusted_r_squared = adjusted_r_squared.mean()

      rmse = np.sqrt(((Y-Yhat)**2).mean(axis=0))

      # print("R-squared for each dimension:", r_squared)

      # print("Adjusted R-squared for each dimension:", adjusted_r_squared)

      # print("Mean Adjusted R-squared:", mean_adjusted_r_squared)

      return rmse, adjusted_r_squared


