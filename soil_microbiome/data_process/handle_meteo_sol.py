from .. import global_vars
import pandas as pd
from scipy.spatial.distance import cdist
from soil_microbiome.utils.utils import haversine
import os


# Create the DataFrame
def weather_soil(tax_level):
    species = pd.read_excel(os.path.join(global_vars['data_dir'], f"{tax_level}/sol_{tax_level}.xlsx"))
    meteo = pd.read_excel(os.path.join(global_vars['data_dir'], "weather_annual.xlsx"))

    pras = meteo[["lon", "lat"]]
    species_loc = species[["lon", "lat"]]
    species_loc = species_loc.dropna()
    coords1 = pras.to_numpy()
    coords2 = species_loc.to_numpy()

    # Compute the distance matrix using scipy.spatial.distance.cdist
    distance_matrix = cdist(coords2, coords1, lambda u, v: haversine(u, v))

    # Find the index of the minimum distance in df2 for each point in df1
    closest_indices = distance_matrix.argmin(axis=1)
    res = meteo.loc[closest_indices, ["PRA_Code", "TEMPERATURE_AVG", "PRECIPITATION"]].reset_index(drop=True)
    res.rename(columns={"TEMPERATURE_AVG": "MAT", "PRECIPITATION": "MAP"}, inplace=True)
    res2 = pd.concat([res, species], axis=1)
    res2.to_excel(os.path.join(global_vars['data_dir'], f"{tax_level}/sol_meteo_{tax_level}.xlsx"))
