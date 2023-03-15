import pandas as pd
from src.dsm.ft import grid

men = pd.read_csv('data/MEN_dataset_natural_form_full', delim_whitespace=True, header=None, names=['w1','w2','sim'])

grid(corpus='data/coca_clean.txt',
     men=men,
     out_path='output/FTmodels',
     dimensionalities=[300, 100],
     window_sizes = [2, 5, 10],
     min_values=[0],
     max_values=[0])
