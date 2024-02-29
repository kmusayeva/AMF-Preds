import os
import pandas as pd
import numpy as np
from soil_microbiome import global_vars




def handle_old_data(tax_level):

    species_20210216 = pd.read_excel(global_vars['old_species'])

    species_20210216.replace(np.nan, 'Unidentified', inplace=True)
    
    species_20210216.Abundance.round(4)
    
    tmp = species_20210216[['ID', tax_level, 'Abundance']]

    df = tmp.pivot_table(index='ID', columns=tax_level, values='Abundance', aggfunc='sum', fill_value=0)

    output_dir = global_vars['old_data_dir']+tax_level

    df.to_excel(f'{output_dir}.xlsx', index=True)


