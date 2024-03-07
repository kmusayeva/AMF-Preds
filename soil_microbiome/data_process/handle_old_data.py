import os
import pandas as pd
import numpy as np

from ..utils import *
from .. import global_vars



if __name__ == "__main__":
    join_old_soil_species('Order')


def handle_old_data(tax_level):

    species_20210216 = pd.read_excel(global_vars['old_species'])

    species_20210216.replace(np.nan, 'Unidentified', inplace=True)
    
    species_20210216.Abundance.round(4)
    
    tmp = species_20210216[['ID', tax_level, 'Abundance']]

    df = tmp.pivot_table(index='ID', columns=tax_level, values='Abundance', aggfunc='sum', fill_value=0)

    output_dir = os.path.join(global_vars['old_data_dir'], tax_level)

    df.to_excel(f'{output_dir}.xlsx', index=True)



def join_old_soil_species(tax_level):

    sol = pd.read_excel(global_vars['old_soil'])

    dir_path = global_vars['old_data_dir']

    df = pd.read_excel(os.path.join(dir_path, f"{tax_level}.xlsx"))

    columns={
        'CEC /sec (méq/100g)':'CEC',
        'Matière Organique /sec (%)':'MO',
        'Argile - inférieur à 2 µm  /sec (%)':'Argile',
        'Limons fins - entre 2 et 20 µm /sec (%)':'Limons_fins',
        'Limons grossiers - entre 20 et 50 µm /sec (%)':'Limons_grossiers',
        'Sables fins - entre 50 et 200 µm (ou 0,05 et 0,2 mm) /sec (%)':'Sables_fins',
        'Sables grossiers - entre 200 et 2000 µm (ou 0,2 et 2 mm) /sec (%)':'Sables_grossiers',
        'pH eau /sec':'pH',
        'CaO échangeable /sec (mg/kg )':'CaO',
        'K2O échangeable /sec (mg/kg )':'K2O',
        'MgO échangeable /sec (mg/kg )':'MgO',
        'P2O5 Olsen /sec (mg/kg )':'P2O5',
        'Carbone organique /sec (%)':'CO'
                    }

    sol.rename(columns = columns, inplace=True)

    cols = np.concatenate((np.array(["ID", "Longitude", "Latitude"]),np.array(list(columns.values()))))

    sol = sol[cols]
    
    especes_sol = pd.merge(sol, df, on="ID").drop(columns=["ID"])

    #df1 = especes_sol.dropna(subset=["Longitude", "Latitude"]).groupby(["Longitude", "Latitude"]).mean().reset_index()

#    df2 = especes_sol[pd.isna(especes_sol["Longitude"])].groupby(especes_sol.columns[2:18].tolist()).mean().reset_index()
#
#    df2_cols = pd.DataFrame({"Longitude":[np.nan] * df2.shape[0], "Latitude":[np.nan] * df2.shape[0]})
#
##    df2_modified = pd.concat([df2_cols, df2], axis=1)
#
#    final_result = pd.concat([df1, df2], axis=0, ignore_index = True)
  
    especes_sol.to_excel(dir_path+f'/soil_{tax_level}.xlsx', index=False)


