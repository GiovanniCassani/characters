import pandas as pd
from src.preprocessing.corpus import make


names = pd.read_csv("data/avgRatings_annotated.csv", usecols = ["name", "type"])
madeup_names = names.loc[names['type'] == 'madeup']['name'].str.lower()

corpus_dir = 'data/coca/'
out_dir = 'data/coca_clean.txt'
make(corpus_dir, madeup_names, out_dir)
