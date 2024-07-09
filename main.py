from m_lp.multi_label_propagation import *
from soil_microbiome.data_process import weather_soil
from soil_microbiome.data_analysis import *

tax_level = 'Species'

# df = pd.read_excel("./soil_microbiome/data/GlobalAMFungi/Species/sol_Species.xlsx")

soil_usage = ['anthropogenic', 'cropland', 'desert', 'forest', 'grassland', 'shrubland', 'tundra', 'woodland']
soil_vars = ['pH', 'MAP', 'MAT']
env_vars = [*soil_usage, *soil_vars]
species = SpeciesEurope(tax_level=tax_level, x_dim=19, env_vars=env_vars, is_global_amf=True)
species.get_top_species(30)
species.print_info()
model = MLClassification(species)
model.evaluate(2)
