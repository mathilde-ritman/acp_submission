#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import xarray as xr
import numpy as np
import glob
import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from my_library.track_analyses import helpers
import pathlib
import logging

datadir = pathlib.Path(f'/work/bb1153/b382635/plots/tracked_results_2025/dataset_paper/results_data/acp_submission/')


# In[2]:


# load data filtering results created during tracking 

# i) systems that hit boundaries
df1 = pd.read_csv('/work/bb1153/b382635/data/final_tracks/updraft_ice_only/amazon/data_filtering_stats/system_hits_boundary.csv', index_col='system_id')
invalid = df1.index[df1.hits_boundary==True]

# ii) number of cores
df2 = pd.read_csv('/work/bb1153/b382635/data/final_tracks/updraft_ice_only/amazon/data_filtering_stats/system_n_cores.csv', index_col='system_id')

# collect
df = pd.concat((df1,df2), axis=1)
hits_bndry = df.index[df.hits_boundary]


# In[3]:


# load statistics to calculate (iii) system size relative to the domain and (iv)  whether the fist core arose below the freezing level


# In[5]:


# select whether to iterate
iterate = 1
if iterate:
    batch = int(sys.argv[1])
    size = int(sys.argv[2])
    n_clouds = size
else:
    batch = size = None
    n_clouds = 100
data_params = dict(batch=batch, size=size, n_clouds=n_clouds)

fdir = f'/work/bb1153/b382635/data/track_statistics/updraft_ice_only/amazon/system-wise/fcsfirst/'  
ds = helpers.load_stats(fdir, ['cloud_area', 'core_bh'], sidx_ignore=hits_bndry, **data_params)


# In[6]:


# (iii) size relative to domain
n_cells = 300 * 400
cell_area = 11000**2 # m2
domain_area = cell_area * n_cells # m2
rel_size = (100 * (ds.cloud_area.max('time') / domain_area)).to_dataframe(name='relative_size')


# In[17]:


# (iv) first core arises at what height?
n_cores_above_freezing = (ds.core_bh.min('time')>4000).sum('core').to_dataframe('n_cores_above_freezing')
n_cores_above_freezing = n_cores_above_freezing.astype(int)


# In[21]:


# results
previous = pd.read_csv(datadir / 'system_validity.csv', index_col='system_id')
new = pd.concat((df, rel_size, n_cores_above_freezing), axis=1)
final = pd.concat((previous, new), axis=0).groupby(level=0).first() # keep existing result and append new
final.index.name = 'system_id'

# save
final.to_csv(datadir / 'system_validity.csv')


# In[ ]:


logging.info(f'saved {new.index.size} new data points')


# In[24]:


# # view
pd.read_csv(datadir / 'system_validity.csv', index_col='system_id')


# In[ ]:




