"""Compare results of TBIP model to ground truth of ideal points from votes.

### References
[1] Vafa, Keyon, Suresh Naidu, and David M. Blei. Text-Based Ideal Points, (2020). 
    https://github.com/keyonvafa/tbip/tree/master
"""

import os
import sys 

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 

data_dir = os.path.join(project_dir, "data")
save_dir = os.path.join(project_dir, "data/results")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# read data from IP model
senator_map = np.loadtxt(os.path.join(data_dir, 'votes/input_2019/senator_map.txt'), dtype=str, delimiter=',')
ideal_point_loc = np.load(os.path.join(data_dir, 'votes/output_2019/ideal_point_loc.npy'))
polarity_loc = np.load(os.path.join(data_dir, 'votes/output_2019/polarity_loc.npy'))
popularity_loc = np.load(os.path.join(data_dir, 'votes/output_2019/popularity_loc.npy'))

# read data from TBIP model
authors = pd.read_csv(os.path.join(data_dir, 'authors.csv'))

# convert IP results to DataFrame and clean
df_ip = pd.DataFrame({'name': senator_map, 'ideal_point': ideal_point_loc})
df_ip[['name', 'party']] = df_ip['name'].str.split(' \(', n=1, expand=True)
df_ip.loc[:, 'party'] = df_ip['party'].str.rstrip(')')

# clean TBIP results
authors.drop(columns='Unnamed: 0', inplace=True)
authors['name'] = authors['name'].str.replace('\n', '')
authors[['name', 'party']] = authors['name'].str.split(' \(', n=1, expand=True)
authors.loc[:, 'party'] = authors['party'].str.rstrip(')')

# uniform party abbreviations
df_ip.loc[df_ip['party']=='Ind', 'party'] = 'Independent'
df_ip.loc[df_ip['party']=='Ld', 'party'] = 'Libdem'
df_ip.loc[df_ip['party']=='Cuk', 'party'] = 'Change Uk'
df_ip.loc[df_ip['party']=='Pc', 'party'] = 'Plaidcymru'
df_ip.loc[df_ip['party']=='Green', 'party'] = 'Gpew'

# join results
authors.set_index(['name', 'party'], inplace=True)
df_ip.set_index(['name', 'party'], inplace=True)
comparison = authors.join(df_ip, on=['name', 'party'], how='inner', lsuffix='_tbip', rsuffix='_ip')

rmse = mean_squared_error(comparison['ideal_point_ip'], comparison['ideal_point_tbip'], squared=False)
r2 = r2_score(comparison['ideal_point_ip'], comparison['ideal_point_tbip'])
results = np.array([rmse, r2])

np.save(os.path.join(save_dir, "results.npy"), results)
