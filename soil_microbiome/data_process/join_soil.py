import pandas as pd
from config import *

def convert_to_binary(tax_level):

    sol = pd.read_excel("~/RProject/AMF-Mycophyto/data/sol/sol_aurea2.xlsx")

    dir_path = output_dir_name+tax_level

    df= pd.read_excel(dir_path+"/all.xlsx")
    
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









