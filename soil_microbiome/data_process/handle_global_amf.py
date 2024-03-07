import pandas as pd
import numpy as np

from ..utils import *
from .. import global_vars


directory = os.path.join(global_vars['global_amf_dir'])

cols_numeric = ['longitude', 'latitude', 'MAT', 'MAP', 'pH']

dfs = []


for filename in os.listdir(directory):

    if filename.endswith(".txt"):
        
        df = pd.read_csv(os.path.join(directory, filename), delimiter='\t')  # Adjust delimiter if necessary

        # Get the file name without the extension
        species_name = os.path.splitext(filename)[0]

        # Rename the 'abundances' column to the species name
        df.rename(columns={'abundances': species_name}, inplace=True)

        dfs.append(df)



combined_df = pd.concat(dfs, ignore_index=True)

combined_df.replace("NA_", np.nan, inplace=True) 

combined_df[cols_numeric] = combined_df[cols_numeric].apply(pd.to_numeric)


print(combined_df)

combined_df_grouped = combined_df.groupby(combined_df.columns[0:16].tolist(), as_index=False)[combined_df.columns[16:len(combined_df.columns)]].agg('sum')

# Reset index to make 'ID' a column again
combined_df_grouped.reset_index(inplace=True)

combined_df_grouped.to_excel(os.path.join(directory, "all.xlsx"), index=False)


