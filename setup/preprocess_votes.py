"""Preprocess UK House of Commons votes using methods described by Vafa et al.

### References
[1] Vafa, Keyon, Suresh Naidu, and David M. Blei. Text-Based Ideal Points, (2020). 
    https://github.com/keyonvafa/tbip/tree/master
"""

import os
import sys 

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 

import numpy as np
import pandas as pd

data_dir = os.path.join(project_dir, "data/votes")
save_dir = os.path.join(project_dir, "data/votes/input_2019")

#%%

# read data
df = pd.read_csv(os.path.join(data_dir, 'original_data.csv'))
df.drop(columns='Unnamed: 0', inplace=True)

#%%

# retain only votes until 2019
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%dT%H:%M:%S')
df = df[df['Date'].dt.year<2020]

# drop votes without party abbreviation
df = df[~df['PartyAbbreviation'].isna()]

# create feature with "first_name last_name (party)"
df.loc[:, 'NameParty'] = df['Name'] + ' (' + df['PartyAbbreviation'] + ')'

#%%

# create necessary arrays
# NOTE: we are considering members of parliament, not senators, but we maintain the notation of Vafa et al.

senator = df['NameParty'].to_numpy() # non-unique array of members of parliament names
senator_to_senator_id = dict([(y.title(), x) for x, y in enumerate(sorted(set(senator)))]) # mp name: id
senator_indices = np.array([senator_to_senator_id[s.title()] for s in senator]) # non-unique array of members of parliament ids
senator_map = np.array(list(senator_to_senator_id.keys())) # unique array of members of parliament names

bill = df['DivisionId'].to_numpy() # non-unique array of original bill ids
bill_to_bill_id = dict([(y, x) for x, y in enumerate(sorted(set(df['DivisionId'])))]) # original bill id: sequential bill id
bill_indices = np.array([bill_to_bill_id[b] for b in bill]) # non-unique array of sequential bill ids

votes = df['Vote'].to_numpy() # array of all votes for all mps on all bills as 1 (low), 0 (high)

#%%

# save data
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
  
np.save(os.path.join(save_dir, 'votes.npy'), votes)
np.save(os.path.join(save_dir, 'senator_indices.npy'), senator_indices)
np.save(os.path.join(save_dir, 'bill_indices.npy'), bill_indices)
np.savetxt(os.path.join(save_dir, 'senator_map.txt'), senator_map, fmt="%s")