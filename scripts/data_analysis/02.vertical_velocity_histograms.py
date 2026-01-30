#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pathlib
import intake
import global3d_track as g3d
src = g3d.scripts.src
import xarray as xr
import sys 

work_dir = pathlib.Path(f'/work/bb1153/b382635/plots/tracked_results_2025/dataset_paper/results_data/acp_submission/')
work_dir = work_dir / 'w_histograms'
os.makedirs(work_dir, exist_ok=True)


# In[2]:


# load surrounding data

# times
start = '20210701'
end = '20210708'
cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
ds = cat.ICON.ngc4008a(time="PT15M", zoom=9).to_dask().sel(time=slice(start, end))


# In[ ]:


# # select subset
# i = int(sys.argv[-1])
# # i = 0
# np.random.seed(12)
# random_timesteps = np.random.randint(0,ds.time.size,ds.time.size)
# ds = ds.isel(time=random_timesteps[i])


# In[ ]:


# # regrid
# region = 'amazon'
# ds = src.utils.regrid.Regrid(region).perform(dataset[['pfull','wa_phy']], zoom=9, resolution=0.1).sel(level_full=slice(40,90),level_half=slice(41,91))


# In[ ]:


# # approximate 500 hPa level

# p500 = np.abs((ds.pfull*0.01) - 500).idxmin('level_full')
# ds_500 = ds.sel(level_half=p500, level_full=p500)


# In[ ]:


# # save histogram for time

# vals = ds_500.wa_phy.values
w_bins = np.linspace(-15,15,1000)
# w_hist, bins = np.histogram(vals[~np.isnan(vals)], bins=w_bins)

# with open(work_dir + f'w500_histogram-{i}.npy', 'wb') as f:
#     np.save(f, w_hist)
# with open(work_dir + 'w500_histogram_bins.npy', 'wb') as f:
#     np.save(f, w_bins)


# In[ ]:


# # cloudy cases
# cloudy = (ds.cli+ds.clw>1e-5).sum('level_full') > 0 # more than 1 layer with cli+clw > 1e-5


# In[ ]:


# # save cloudy / cloud free histograms for time

# vals = ds_500.wa_phy.where(cloudy).values
# w_hist, bins = np.histogram(vals[~np.isnan(vals)], bins=w_bins)

# with open(work_dir + f'w500_histogram_cloudy-{i}.npy', 'wb') as f:
#     np.save(f, w_hist)

# vals = ds_500.wa_phy.where(~cloudy).values
# w_hist, bins = np.histogram(vals[~np.isnan(vals)], bins=w_bins)

# with open(work_dir + f'w500_histogram_not_cloudy-{i}.npy', 'wb') as f:
#     np.save(f, w_hist)


# In[ ]:


i = int(sys.argv[-1])


# In[14]:


# afternoon very cloudy

# - sample
arvo_times = ds.time.sel(time=ds.time.dt.hour.isin([x-4 for x in [14,15,16,17,18]]))
random_timesteps = np.random.randint(0,arvo_times.size,arvo_times.size)
ds = ds.sel(time=arvo_times.isel(time=random_timesteps[i]))

# - regrid
region = 'amazon'
ds = src.utils.regrid.Regrid(region).perform(ds[['pfull','wa_phy','cli','clw']], zoom=9, resolution=0.1).sel(level_full=slice(40,90),level_half=slice(41,91))

# - approximate 500 hPa level

p500 = np.abs((ds.pfull*0.01) - 500).idxmin('level_full')
ds_500 = ds.sel(level_half=p500, level_full=p500)

# - find
arvo_very_cloudy = (ds.cli+ds.clw>1e-3).sum('level_full') > 0 # more than 1 layer with cli+clw > 1e-3


# In[ ]:


# - save

vals = ds_500.wa_phy.where(arvo_very_cloudy).values
w_hist, bins = np.histogram(vals[~np.isnan(vals)], bins=w_bins)

with open(work_dir / f'w500_histogram_arvo_very_cloudy-{i}.npy', 'wb') as f:
    np.save(f, w_hist)


# In[ ]:




