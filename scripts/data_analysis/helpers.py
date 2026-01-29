import xarray as xr
import glob
import re
import pathlib
import logging
import numpy as np
import pandas as pd


def load_stats(fdir, variables=[], n_clouds=50, batch=None, size=None, sidx_list=None, sidx_ignore=[], apply=None):

    # all files
    files = sorted(glob.glob(fdir + '*.nc'), key=lambda x: int(re.search(r'_(\d+)\.nc$', x).group(1)))

    # choose exact
    if sidx_list is not None:
        get_these = []
        for f in files:
            sidx = int(re.search(r'cloud_(\d+)\.nc$', pathlib.Path(f).name).group(1))
            if sidx in sidx_list:
                get_these.append(f)
        files = get_these

    # subselect number of samples
    else:
        if batch:
            files = files[(batch - 1) * size:batch * size]
        else:
            files = files[:n_clouds]

        # ignore some
        for f in files:
            sidx = int(re.search(r'cloud_(\d+)\.nc$', pathlib.Path(f).name).group(1))
            if sidx in sidx_ignore:
                files.pop(files.index(f))

    def preprocess(ds):
        fname = ds.encoding["source"]
        sidx = int(re.search(r'cloud_(\d+)\.nc$', pathlib.Path(fname).name).group(1))
        ds = ds.where(ds != -999.99)
        if variables:
            ds = ds[variables]
        if callable(apply):
            ds = apply(ds)
        if 'core' in ds.dims:
            if ds.core.notnull().sum() == 0:
                ds['core'] = np.arange(0,ds.core.size) + (10000*sidx)
        return ds.expand_dims(dict(system=[sidx]))

    return xr.open_mfdataset(files, concat_dim='system', combine='nested', preprocess=preprocess)


def load_stats_old(fdir, variables=[], n_clouds=50, batch=None, size=None, sidx_ignore=[], apply=None):
    ''' Load results statistics from specified directory, for the specified sample of clouds. '''

    # get grid to map samples to
    full_mask_coords = xr.open_mfdataset('/work/bb1153/b382635/data/final_tracks/updraft_ice_only/amazon/20210701/20210701T0000_20210702T0000_system_tracks_linked_proc.nc')
    full_mask_coords['lat'] = full_mask_coords.lat.round(2)
    full_mask_coords['lon'] = full_mask_coords.lon.round(2)
    full_mask_coords = full_mask_coords.drop_dims('time').coords

    # subselect number of samples
    files = sorted(glob.glob(fdir + '*.nc'), key=lambda x: int(re.search(r'_(\d+)\.nc$', x).group(1)))
    if batch:
        files = files[(batch - 1) * size:batch * size]
    else:
        files = files[:n_clouds]

    # load
    li = []
    for f in files:
        sidx = int(re.search(r'cloud_(\d+)\.nc$', pathlib.Path(f).name).group(1))
        if sidx in sidx_ignore:
            # skip cloud
            continue
        d = xr.open_mfdataset(f)
        d = d.where(d != -999.99)
        if 'lat' in d.dims:
            if variables:
                d = d[variables]
            if callable(apply):
                d = apply(d)
            d = xr.Dataset(coords=full_mask_coords).merge(d)
        if 'core' in d.dims:
            core_vals = d['core'].values
            d = d.assign_coords(core=core_vals + sidx * 100000)
            d['core_orig'] = ('core', core_vals)
        d = d.expand_dims(dict(system=[sidx]))
        li.append(d)
        
    # concat
    return xr.concat(li, dim='system')

def by_lifetime(obj_exists, dataarrays, n_chunks=10, max_lifetime=30, chunk_var='system'):
    max_lifetime = max_lifetime * 60 # hours to minutes
    max_tsteps = max_lifetime / 15
    age = np.arange(0,max_tsteps,1)*15

    def by_lifetime_vectorized(data, exists):
        data = data[exists]
        ntimes = data.size
        data = np.append(data, [np.nan]*(age.size-ntimes))
        return data

    def apply(da, exists):
        result = xr.apply_ufunc(
            by_lifetime_vectorized,
            da,
            exists,
            input_core_dims=[['time'],['time']],
            output_core_dims=[['age']],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={"output_sizes": {"age": age.size}},
            output_dtypes=[float],
        )
        return result.assign_coords(age=age)

    dataarrays = list(d.chunk({chunk_var: n_chunks, 'time': -1}) for d in dataarrays)
    exists = obj_exists.chunk({chunk_var: n_chunks, 'time': -1})

    result = xr.Dataset()
    for da in dataarrays:
        result[da.name] = apply(da, exists)

    return result


def normalise_by_lifetime(obj_exists, dataarrays, chunk_var='system', n_chunks=10, bins=np.arange(0, 1.05, 0.05)):

    def normalise_by_lifetime_vectorized(data, exists):
        data = data[exists]
        ntimes = data.size
        if ntimes < 2:
            return np.full((len(bins),), np.nan)
        time_percentage = np.linspace(0, 1, ntimes)
        return np.interp(bins, time_percentage, data)

    def apply_normalisation(dataarray, cloud_exists):
        result = xr.apply_ufunc(
            normalise_by_lifetime_vectorized,
            dataarray,
            cloud_exists,
            input_core_dims=[['time'],['time']],
            output_core_dims=[['interp_time']],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={"output_sizes": {"interp_time": len(bins)}},
            output_dtypes=[float],
        )
        return result.assign_coords(interp_time=bins)

    dataarrays = list(d.chunk({chunk_var: n_chunks, 'time': -1}) for d in dataarrays)
    cloud_exists = obj_exists.chunk({chunk_var: n_chunks, 'time': -1})

    result = xr.Dataset()
    for d in dataarrays:
        result[d.name] = apply_normalisation(d, cloud_exists)

    return result


def load_amazon_metatdata():
    # get metadata

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
    metadata['lifetime'] = (metadata["lifetime"] / np.timedelta64(1, "s")).astype("float32") # to seconds

    return metadata