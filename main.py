import os
import pandas as pd
import numpy as np
import argparse
import sys
from soil_microbiome.data_process.handle_old_data import *


parser = argparse.ArgumentParser(description='Process taxonomic level.')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)


parser.add_argument('-tax_level', help='Taxonomic level')

args = parser.parse_args()

tax_level = args.tax_level

handle_old_data(tax_level)


#create_norm_reads(tax_level)

#join_reads(tax_level)

#convert_to_binary(tax_level)

#join_soil_species(tax_level)



