import pandas as pd
import numpy as np

from ..utils import * 
from .. import global_vars

if __name__ == "__main__":
    print(global_vars['soil_file'])


def create_norm_reads(tax_level):

    input_dir = global_vars['reads_dir']

    data_dir = os.path.join(global_vars['data_dir'],tax_level)

    create_directory_if_not_exists(data_dir)

    xlsx_files = [file for file in os.listdir(input_dir) if file.endswith('.xlsx')]

    # Iterate over each file
    for file in xlsx_files:

        dat = pd.read_excel(os.path.join(input_dir, file))

        dat.replace(np.nan, 'Unidentified', inplace=True)
        
        regex_pattern = f'^({tax_level}|MYC)'

        tmp = dat.filter(regex=regex_pattern)

        #tmp = dat.filter(regex='^(Famille|MYC)')
        
        reads = tmp.groupby(tax_level).sum(numeric_only=True).reset_index()
        
        numeric_columns = reads.select_dtypes(include=[np.number])

        reads[numeric_columns.columns] = numeric_columns.div(numeric_columns.sum()).round(4)

        #reads.replace(np.nan, 0.0, inplace=True)

        reads[tax_level].fillna('Unidentified', inplace=True)

        df = reads.melt(id_vars=[tax_level], var_name='ID', value_name='Value')

        df = df.dropna(subset=['Value'])

        df = df.pivot(index='ID', columns=tax_level, values='Value').reset_index()

        #df.set_index('ID', inplace=True)

        directory = os.path.dirname(os.path.join(data_dir, file))
        
        file_name, file_ext = os.path.splitext(os.path.basename(file))
        
        new_file_name = file_name + "_norm" + file_ext
        
        new_file = os.path.join(directory, new_file_name)
        
        df.to_excel(new_file, index=False)


def join_reads(tax_level):

    dir_path = data_dir+tax_level

    xlsx_files = [file for file in os.listdir(dir_path) if file.endswith('.xlsx')]

    # List to store DataFrames
    dfs = []

    # Iterate over files in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.xlsx'):
            # Read Excel file into a DataFrame
            df = pd.read_excel(os.path.join(dir_path, filename))
            dfs.append(df)

    # Concatenate DataFrames along the row axis
    result = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    result = result.groupby('ID').mean().reset_index(False)

    result.fillna(0, inplace=True)

    result.to_excel(dir_path+"/all.xlsx", index=False) 



def convert_to_binary(tax_level):
    
    dir_path = data_dir+tax_level
    
    df = pd.read_excel(dir_path+"/all.xlsx")
    
    df.iloc[:,1:df.shape[1]] = df.iloc[:,1:df.shape[1]].apply(lambda x:[y if y==0.0 else 1 for y in x])

    df.to_excel(dir_path+"/all_binary.xlsx", index=False)



def join_soil_species(tax_level):

    sol = pd.read_excel(soil_file)

    dir_path = data_dir+tax_level

    df = pd.read_excel(dir_path+"/all.xlsx")

    especes_sol = pd.merge(sol, df, on="ID").drop(columns=["ID"])

    df1 = especes_sol.dropna(subset=["Longitude", "Latitude"]).groupby(["Longitude", "Latitude"]).mean().reset_index()

    df2 = especes_sol[pd.isna(especes_sol["Longitude"])].groupby(especes_sol.columns[2:18].tolist()).mean().reset_index()

    df2_cols = pd.DataFrame({"Longitude":[np.nan] * df2.shape[0], "Latitude":[np.nan] * df2.shape[0]})

    df2_modified = pd.concat([df2_cols, df2], axis=1)

    final_result = pd.concat([df1, df2], axis=0, ignore_index = True)

    # rename columns
    final_result.rename(columns={
        "Argile - inférieur à 2 µm (ou 0.002 mm) /sec (%)": "Argile",
        "Azote Total /sec": "Azote",
        "Sables fins - entre 50 et 200 µm (ou 0.05 et 0.2 mm) /sec (%)": "Sables_fins",
        "Sables grossiers - entre 200 et 2000 µm (ou 0.2 et 2 mm) /sec (%)": "Sables_grossiers",
        "Limons fins - entre 2 et 20 µm (ou 0.002 et 0.02 mm) /sec (%)": "Limons_fins",
        "Limons grossiers - entre 20 et 50 µm (ou 0.02 et 0.05 mm) /sec (%)": "Limons_grossiers",
        "Carbone organique /sec (%)": "CO",
        "CEC /sec (méq/100g)": "CEC",
        "K2O échangeable /sec (mg/kg )": "K2O",
        "CaO échangeable /sec (mg/kg )": "CaO",
        "Matière Organique /sec (%)": "MO",
        "MgO échangeable /sec (mg/kg )": "MgO",
        "P2O5 Olsen /sec (mg/kg )": "P2O5",
        "pH eau /sec": "pH"
    }, inplace=True)

    # write to excel
    final_result.to_excel(dir_path + f'/soil_{tax_level}.xlsx', index=False)



