import pymc3 as pm
from sklearn.metrics import r2_score
from soil_microbiome.utils.utils import *

class DirichletRegression:
    
    def __init__(self, species, params=None):
        self.species = species
        self.params = params
        self.adjusted_rscore, self.rmse = None, None
        self.n, self.p = self.species.X.shape
        self.m = self.species.Y.shape[1]
        self.Yhat = None

    def do(self):
          mu = self.params['mu'] if 'mu' in self.params else 0.0
          sigma = self.params['sigma'] if 'sigma' in self.params else 1.0
          cores = self.params['cores'] if 'cores' in self.params else 2
          #Create design matrix
          X = pd.concat([pd.DataFrame({'Intercept': np.ones(self.n)}), self.species.X], axis=1)
          #We need to transform from [0,1] to (0,1)
          Y_mod = self.species.Y.apply(lambda y: (y * (self.n-1)+1/self.m)/self.n)

          #Define the PyMC3 model
          with pm.Model() as model:
              # Priors for coefficients
              beta = pm.Normal('beta', mu=mu, sigma=sigma, shape=(self.p, self.n))
              # Likelihood function (Dirichlet distribution)
              theta = pm.Deterministic('theta', pm.math.dot(X, beta))
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
          self.Yhat = raw_mu/raw_mu.sum(axis=1).reshape(-1, 1)

            
    
    def evaluate(self):
          r_squared = r2_score(self.species.Y, self.Yhat, multioutput='raw_values')
          #Compute adjusted R-squared for each dimension
          self.adjusted_rscore = 1 - ((1 - r_squared) * (self.n - 1) / (self.n - self.p - 1))
          #If you want to get the mean adjusted R-squared across dimensions
          #mean_adjusted_r_squared = adjusted_r_squared.mean()
          self.rmse = np.sqrt(((self.species.Y-self.Yhat)**2).mean(axis=0))

          # print("R-squared for each dimension:", r_squared)
          # print("Adjusted R-squared for each dimension:", adjusted_r_squared)
          # print("Mean Adjusted R-squared:", mean_adjusted_r_squared)
          print(f"Adjusted r-squared: {self.adjusted_rscore}, root-mean-squared: {self.rmse}")
