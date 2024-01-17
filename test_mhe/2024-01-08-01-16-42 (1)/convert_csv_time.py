import pandas as pd 
import sys 
import os

files = []

for (dirpath, dirnames, filenames) in os.walk(".", topdown=False):
    files.extend(filenames)
    break

for i in files: 

    if i == 'convert_csv_time.py': 
        pass
    else: 
        df = pd.read_csv(i)
        # get t0.
        t0 = df.Time[0]
        # update time.
        df['Time'] = df['Time'] - t0
        df.to_csv(i)
