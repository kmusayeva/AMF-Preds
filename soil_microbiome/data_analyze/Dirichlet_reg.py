#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.metrics import r2_score

from soil_microbiome import global_vars

class Species:
    def __init__(self, level_name, file_path):
        self.level_name = level_name
        self.file_path = file_path
        self.data = None
        self.X = None
        self.Y = None

    def read_data(self):
        self.data = pd.read_excel(self.file_path)

    def slice_data(self, x_indices, y_indices):
        self.X = self.data.iloc[:, x_indices]
        self.Y = self.data.iloc[:, y_indices]

    def get_X_Y(self):
        return self.X, self.Y

    def perform_regression(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model







df = pd.read_excel(f"{global_vars['data_dir']}Ordre/soil_Ordre.xlsx")
#cols = ['pH', 'MO', 'Sables_fins', 'Sables_grossiers', 'Limons_fins', 'Limons_grossiers', 'CEC', 'K2O']
cols = ['pH', 'MO']


#X = df.iloc[:,2:18]
X = df[cols]
Y = df.iloc[:,18:df.shape[1]]

n = X.shape[0]
C = Y.shape[1]
#Y = Y.iloc[:,0:-1]

Y = Y.apply(lambda y: (y * (n - 1) + 1 / C) / n)

# Define your data
#X = np.random.randn(40, 2)
#Y = np.random.dirichlet((10, 5, 3), 40)

#X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))

X_with_intercept = pd.concat([pd.DataFrame({'Int':np.ones(n)}),X], axis=1)


# Define the PyMC3 model
with pm.Model() as model:
    # Priors for coefficients
    beta = pm.Normal('beta', mu=0, sigma=0.5, shape=(X_with_intercept.shape[1], Y.shape[1]))
    
    # Likelihood function (Dirichlet distribution)
    theta = pm.Deterministic('theta', pm.math.dot(X_with_intercept, beta))

    mu = pm.Deterministic("mu", pm.math.exp(theta))

    Y_obs = pm.Dirichlet('Y_obs', mu, observed=Y)
    


# Sample from the posterior distribution
with model:
    trace = pm.sample(2000, tune=1000, cores=2, target_accept=0.9)


fitted_values = trace['mu'][-1000:].mean(axis=0)

Yhat = fitted_values/fitted_values.sum(axis=1).reshape(-1,1)

print(np.sqrt(((Y-Yhat)**2).mean(axis=0)))

r_squared = r2_score(Y, Yhat, multioutput='raw_values')

# Compute adjusted R-squared for each dimension
n = len(Y)
p = X_with_intercept.shape[1]  # Assuming X is your feature matrix
adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

# If you want to get the mean adjusted R-squared across dimensions
mean_adjusted_r_squared = adjusted_r_squared.mean()

print("R-squared for each dimension:", r_squared)

print("Adjusted R-squared for each dimension:", adjusted_r_squared)

print("Mean Adjusted R-squared:", mean_adjusted_r_squared)


# Do predictions (similar to fitted values)
# Generate new data
#X_new = np.random.randn(10, 2)
#X_new_with_intercept = np.hstack((np.ones((X_new.shape[0], 1)), X_new))
#
## Compute predicted values
#with model:
#    theta_pred = pm.math.dot(X_new_with_intercept, beta)
#    mu_pred = pm.math.exp(theta_pred)
#    Y_pred = pm.Dirichlet('Y_pred', mu_pred)
#
## Sample from the predictive distribution
#with model:
#    pm.set_data({'beta': trace['beta']})
#    trace_pred = pm.sample_posterior_predictive(trace, samples=500)
#
## Get predicted values
#predicted_values = trace_pred['Y_pred'].mean(axis=0)
#
