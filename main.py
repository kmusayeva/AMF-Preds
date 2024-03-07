#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pymc3 as pm
import sys
from sklearn.metrics import r2_score

from soil_microbiome.data_analysis  import * 
from soil_microbiome.data_process  import * 



if '-t' in sys.argv:
    index_t = sys.argv.index('-t')
    if index_t + 1 < len(sys.argv):
        taxon = sys.argv[index_t + 1]
        print("Taxon value:", taxon)
    else:
        print("Error: No value provided for -t option")
        sys.exit(1)
else:
    print("Error: -t option not found")
    sys.exit(1)


#handle_old_data(taxon)
join_old_soil_species(taxon)

def analyze():

    env_vars = ['pH', 'MO', 'Sables_fins', 'Sables_grossiers', 'Limons_fins', 'Limons_grossiers', 'CEC', 'K2O']
    #env_vars = ['pH', 'CaO']
    #env_vars = ['pH', 'MO']


    species = Species(taxon, f'{taxon}/sol_{taxon}.xlsx', env_vars, x_dim=19)
    species.set_data()
    species.get_description()
    taxon, input_dim, output_dim = species.get_attributes()
    print(f'Taxon: ', taxon)

    X, Y = species.get_data()
    X_norm = species.get_norm_env_vars()
    print(X_norm)

    params = {'mu':0, 'sigma':1000}

    model = Regression(X_norm, Y, params)

    X, Y, t = model.get_attributes()

    model.do_Dirichlet()
    model.eval_perf()
    rmse, adj_rscore = model.get_perf()

    for a, b in zip(Y.columns.tolist(),adj_rscore):
            print(f"{a} : {b}\n")
