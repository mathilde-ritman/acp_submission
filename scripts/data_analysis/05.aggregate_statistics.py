#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import pandas as pd
import os
import re
import sys
import glob
import pathlib
import datetime
import numpy as np
import scipy
import easygems.healpix as egh
import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})
import logging
from my_library.track_analyses import helpers
import pathlib

import time
import logging
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logging.info(f"[TIMER] {name}: {dt:.2f} s")


outdir = pathlib.Path(f'/work/bb1153/b382635/plots/tracked_results_2025/dataset_paper/results_data/acp_submission/')


# In[2]:


# specify valid data


# In[3]:


df = pd.read_csv(outdir / 'system_validity.csv', index_col='system_id')


# In[4]:


# filter those that hit the boundary or start above freezing
invalid = df.index[np.logical_or(df.hits_boundary, df.dcc_first_tracked_above_freezing==True)]
invalid


# In[14]:


# are there any valid clouds without stats calculated... ?
fdir = f'/work/bb1153/b382635/data/track_statistics/updraft_ice_only/amazon/system-wise/fcsfirst/'
valid = df.index[~df.index.isin(invalid)]
stats_exist_for = []
for f in glob.glob(fdir+'*'):
    stats_exist_for.append(int(re.search(r'cloud_(\d+)\.nc$', pathlib.Path(f).name).group(1)))
missing_stats = [x for x in invalid if x not in stats_exist_for]
len(missing_stats)


# In[5]:


# select whether to iterate

outdir = outdir / 'dcc_statisticsV2'
os.makedirs(outdir, exist_ok=True)

iterate = 1
if iterate:
    batch, size = int(sys.argv[1]), int(sys.argv[2])
    outpath = outdir / f'b{batch}s{size}.nc'
    n_clouds = size
else:
    batch, size = 6, 50
    n_clouds = size
    outpath = outdir / f'b{batch}s{n_clouds}.nc'
    
data_params = dict(sidx_ignore=invalid, batch=batch, size=size, n_clouds=n_clouds)


# In[6]:


# calculate whether complex or isolated multicore DCC


# In[7]:


fdir = f'/work/bb1153/b382635/data/track_statistics/updraft_ice_only/amazon/system-wise/fcsfirst/'
data = helpers.load_stats(fdir, ['core_max_w'], **data_params)


# In[8]:


with timer('isolated'):
    # define spatial footprints
    core_footprint = data.core_max_w>0
    all_footprints = core_footprint.any('core')
    core_exists = core_footprint.max(('lat','lon','time'))
    # determine if they overlap in space
    overlap = (core_footprint.any('time') & (all_footprints & ~core_footprint).any('time')).any(('lat','lon'))
    # do all cores overlap? Y -> is isolated
    is_isolated_dcc = overlap.where(core_exists).all('core') 
    n_cores = core_exists.sum('core')
    is_isolated_dcc = is_isolated_dcc.where(n_cores>1, 1) # set single-core clouds as isolated
    logging.info(f"assessed core overlaps")


# In[9]:


# load results


# In[10]:


def func(ds):
    def aggregate_time(d):
        vmax = ['cth','core_area','core_depth','anvil_depth','core_max_w','core_th']
        vmin = ['abh','core_bh']
        for v in vmax:
            d[v] = d[v].max('time')
        for v in vmin:
            d[v] = d[v].min('time')
        return d
    ds['anvil_depth'] = ds.anvil_depth.max(('lat','lon'))
    ds['abh'] = ds.abh.min(('lat','lon'))
    ds['core_area'] = ds.core_area.max('level_full')
    ds['core_depth'] = ds.core_depth.max(('lat','lon'))
    ds['core_max_w'] = ds.core_max_w.max(('lat','lon'))
    ds = aggregate_time(ds)
    return ds

# load
agg_vars = ['cloud_area','cth','cloud_depth','core_area',
            'core_depth','anvil_depth','core_max_w','abh','core_th','core_bh',
           ]
with timer('load stats'):
    ds = helpers.load_stats(fdir, agg_vars, apply=func, sidx_ignore=invalid, batch=batch, size=size, n_clouds=n_clouds)


# In[11]:


# convert units
ds['cloud_area'] = ds['cloud_area'] / (1000**2)
ds['core_area'] = ds['core_area'] / (1000**2)
ds['cloud_depth'] = ds['cloud_depth'] / (1000)
ds['anvil_depth'] = ds['anvil_depth'] / (1000)
ds['core_depth'] = ds['core_depth'] / (1000)
ds['cth'] = ds['cth'] / (1000)
ds['abh'] = ds['abh'] / (1000)
ds['core_th'] = ds['core_th'] / (1000)
ds['core_bh'] = ds['core_bh'] / (1000)
logging.info(f"got bulk statistics")


# In[12]:


# collect metadata


# In[13]:


# - n cores
path = '/work/bb1153/b382635/data/final_tracks/updraft_ice_only/amazon/data_filtering_stats/system_n_cores.csv'
ncores = pd.read_csv(path, index_col='system_id').rename_axis('system')

# - ABHs
path = '/work/bb1153/b382635/data/final_tracks/updraft_ice_only/amazon/data_filtering_stats/system_anvil_base_height.csv'
abh = pd.read_csv(path, index_col='system_id').rename_axis('system')

# - lifetime
path = '/work/bb1153/b382635/data/final_tracks/updraft_ice_only/amazon/data_filtering_stats/system_lifetime.csv'
lifetime = pd.read_csv(path, index_col='system_id').rename_axis('system')

# to dataset
abh = xr.Dataset.from_dataframe(abh).rename({'0':'ABH'}).round()
lifetime['0'] = pd.to_timedelta(lifetime['0'])
lifetime = xr.Dataset.from_dataframe(lifetime).rename({'0':'lifetime'})
ncores = xr.Dataset.from_dataframe(ncores)
metadata = xr.Dataset({'ABH':abh.ABH, 'lifetime':lifetime.lifetime, 'ncores':ncores.n_cores})

# collect metadata
mds = metadata.sel(system=ds.system)
ds['n_cores'] = mds.ncores
ds['lifetime'] = mds.lifetime

# safe time storage
ds['lifetime'] = (ds["lifetime"] / np.timedelta64(1, "s")).astype("float32") # to seconds
ds.lifetime.attrs = dict(units='seconds')
logging.info(f"collected metatdata")


# In[14]:


# assign whether or not the DCC is isolated


# In[15]:


ds['is_isolated'] = is_isolated_dcc
ds['is_isolated'] = ds.is_isolated.where(ds.n_cores<4, 0) # force systems with >3 cores to be 'complex'
logging.info(f"assigned whether complex or isolated")


# In[16]:


# normalise time by lifetime


# In[17]:


with timer('by lifetime'):
    lifetime_stats = ['cloud_area','cloud_depth']
    bulk_stats = [x for x in ds.data_vars if x not in ['cloud_area','cloud_depth']]
    subset = ds[lifetime_stats]
    # subset = subset.drop_vars(['lat','lon','level_full'])
    obj_exists = subset.cloud_area>0
    life = helpers.normalise_by_lifetime(obj_exists, [subset.cloud_area, subset.cloud_depth], )
    init_t = (subset.cloud_area>0).idxmax('time').compute()
    life['TOI'] = init_t
    logging.info(f"normalised by lifetime")


# In[18]:


logging.info(f"saving...")
final = xr.merge([ds[bulk_stats], life])


# In[19]:


# save


# In[20]:


final["TOI"].attrs = {}
final["TOI"].encoding = {
    "units": "seconds since 1970-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "int64",
}


# In[21]:


with timer('save'):
    final.to_netcdf(outpath)


# In[22]:


logging.info(f"done")


# In[ ]:





# In[ ]:




