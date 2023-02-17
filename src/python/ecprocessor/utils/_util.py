"""
Utility functions for eddy covariance processing, such as functions to read campbell files and strip timestamps from their file names

Author: Alex Fox
Created: 2023-01-30

TODO: add examples
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
import xarray as xr

def get_timestamp_from_fn(fn):
    '''
    Given a raw high-frequency data file, get the starting timestamp from the filename.
    filenames look like:
    ^.*10Hz[0-9]+_yyyy_mm_dd_hhMM$
    
    Parameters
    ----------
    fn: str or path
    
    Returns
    -------
    datetime.datetime
    '''
    # get a list of [yyyy, mm, dd, hh, MM] from filename
    if not isinstance(fn, Path().__class__):
        fn = Path(fn)

    timestamp_elements = fn.stem.split('10Hz')[1].split('_')[1:]
    hh, mm = timestamp_elements[-1][:2], timestamp_elements[-1][2:]
    timestamp_elements = timestamp_elements[:-1] + [hh, mm]
    timestamp_elements = [int(i) for i in timestamp_elements]
    # convert to datetime
    timestamp = datetime(*timestamp_elements)
    return timestamp

def read_campbell_file(fn, 
                       parse_dates=['TIMESTAMP'], skiprows=[0, 2, 3], na_values=['NAN', '"NAN"', "'NAN'"], 
                       **read_csv_kwargs):
    '''
    read in a campbell scientific TOA5/TOB3 file as a pandas dataframe.
    '''
    df = pd.read_csv(
        fn, 
        skiprows=skiprows, 
        parse_dates=parse_dates, 
        na_values=na_values,
        **read_csv_kwargs
                    )
    return df

def compute_summary(fn, renaming_dict, 
                    **read_campbell_file_kwargs):
    '''Compute statistical summary of a high-frequency data file
    
    Parameters
    ----------
    fn: str or path
        path to the desired file
    renaming_dict: dict
        a mapping from raw_column_name -> standardized_column_name. Only columns in the renaming_dict will be returned. Do not include the TIMESTAMP column
    **read_campbell_file_kwargs: dict
        keyword arguments passed to read_campbell_file()
    
    Returns
    -------
    out_names : list of variables names associated with out_data entries dim 1
    out_stats :  list of summary stats associated with out_data entries dim 0
    out_data : 2d array, dims (stat, name)
    
    '''
    
    df = (
        read_campbell_file(fn, **read_campbell_file_kwargs)
        .rename(columns=renaming_dict)
    )
    
    new_names = renaming_dict.values()
    means = list(df[new_names].mean().values)
    stds = list(df[new_names].std().values)
    skws = list(df[new_names].skew().values)
    krts = list(df[new_names].kurt().values)
    frac = list((df[new_names].count()/df.shape[0]).values)   # fraction of non-na values

    out_names = list(new_names)
    out_stats = ['mean', 'std', 'skew', 'kurt', 'frac']
    out_data = np.stack((means, stds, skws, krts, frac))
    
    return out_names, out_stats, out_data

def summarize_files(
    data_dir, renaming_dict, 
    glob='*',
    dest=None,
    verbose=False,
    ):
    '''
    Read in campbell TOA5 files from data_dir following the given glob pattern, and summarize each data file.
    Currently, this only works with files that are parseable by read_campbell_file and get_timestamp_from_fn.

    Parameters
    ----------
    data_dir : path or str
        directory containing data files.
    glob : str (default '*')
        glob string to fild files within data dir.
    dest : path or str (default None)
        Path to write the summary file to. If None (default), do not write to file. Currently, write files to a netCDF NETCDF4 file
    out_format: one of ['csv', 'pickle', 'parquet'], default 'pickle'
        output file format. Currently only 'pickle' is supported.
    verbose : bool (default False)
        Whether to generate verbose output
    renaming_dict : dict or mapping
        maps raw column names to standard column names
        Options for standard column names are U, V, W, Ts, P, H2O, or CO2.

    Returns
    -------
    summary_data : xr.Dataset
        dataset providing the timestamp and filename associated with each file, plus the mean, std, skw, and krt of each provided column indexed by time.
    '''

    # record file timestamps and paths
    data_dir = Path(data_dir)
    files = list(data_dir.glob(glob))
    files_df = pd.DataFrame(dict(fn=files))
    files_df['TIMESTAMP'] = list(map(get_timestamp_from_fn, files_df['fn']))
    files_df = files_df.sort_values('TIMESTAMP').set_index('TIMESTAMP')
    
    # compute summaries: could be way faster
    iterfiles = files_df['fn']
    if verbose:
        iterfiles = tqdm(files_df['fn'])
    out = [compute_summary(fn, renaming_dict) for fn in iterfiles]
    out_names = out[0][0]
    out_stats = out[0][1]
    # dims (name, stat, file)
    out_data = np.transpose(np.array([i[2] for i in out]), axes=(2, 0, 1))
    # summarize default data
    summary_data = xr.Dataset(
        {
            name:xr.DataArray(
                data=var,
                dims=['TIMESTAMP', 'Stat'],
                coords=dict(
                    TIMESTAMP=files_df.index,
                    Stat=out_stats,
                ),
                name=name,
            )
            for var, name in zip(out_data, out_names)
        }
    )
    
    summary_data['fn'] = xr.DataArray([str(fn) for fn in files_df['fn']], dims=['TIMESTAMP'])

    if dest is not None:
        summary_data.to_netcdf(path=Path(dest))
    return summary_data

def compute_aggregate_metrics(
    summary, 
    dest=None,
    tilt_correction=None
):
    '''
    Computes the requrested aggregatte metrics from a summary file or dataset returned by summarize_files
    Currently only 

    Parameters
    -----------
    summary : str or Path, or xr.Dataset
        either path to the dataset output by summarize_files or the object returned by summarize_files
    dest : str or Path, default None
        path to write output to
    tilt_correction : str, default None
        tilt correction algorithm to apply. Options are:
            DR: Double Rotations
            TR: Triple Rotations
            PF: Planar Fit
            CPFn: n-th order Continuous Planar Fit. Replace "n" with the desired order of the fourier series fit. n=1 is identical to the standard Planar Fit. For example, CPF5 would give a 5th order fit, and CPF15 would give a 15th order fit.

    Return
    -------
    summary : xr.Dataset
        modified summary dataset with corrected values and any information needed to apply corrections to fast data
    '''

    # try to open the file, otherwise we should have been given a dataset
    if isinstance(summary, str):
        summary = xr.open_dataset(summary)
    assert isinstance(summary, xr.Dataset), f'got type(summary)={type(summary)}. Summary is not a dataset!'

    if tilt_correction:
        from .. import tiltcorrections as tc
        # get U, V, W data
        U, V, W = summary['U'].sel(Stat='mean').data, summary['V'].sel(Stat='mean').data, summary['W'].sel(Stat='mean').data

        # apply tilt correction
        if tilt_correction == 'DR':
            # compute angles
            theta, phi = tc.get_double_rotation_angles(U, V, W)
            # compute mean rotations
            UVW_rot = tc.double_rotation_fit_from_angles(U, V, W, theta, phi)
            # store angles
            summary['Theta'] = xr.DataArray(
                data=theta,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':'DR or TR'}
            )
            summary['Phi'] = xr.DataArray(
                data=phi,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':'DR or TR'}
            )
        elif tilt_correction == 'TR':
            # get rotation angles
            theta, phi, psi = tc.get_triple_rotation_angles(U, V, W)
            # get mean rotation results
            UVW_rot = tc.triple_rotation_fit_from_angles(U, V, W, theta, phi, psi)
            # store angles
            summary['Theta'] = xr.DataArray(
                data=theta,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':'DR or TR'}
            )
            summary['Phi'] = xr.DataArray(
                data=phi,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':'DR or TR'}
            )
            summary['Psi'] = xr.DataArray(
                data=psi,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':'TR'}
            )

        elif tilt_correction == 'PF':
            # get rotation angles
            theta, phi = tc.get_double_rotation_angles(U, V, W)
            theta, phi_func, phidot_func = tc.get_continuous_planar_fit_angles(theta, phi, N=1)
            phi = phi_func(theta)
            phidot = phidot_func(theta)
            # get corrected mean coords
            UVW_rot = tc.continuous_planar_fit_from_angles(U, V, W, theta, phi, phidot)

            summary['Theta'] = xr.DataArray(
                data=theta,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':'PF'}
            )
            summary['Phi'] = xr.DataArray(
                data=phi,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':'PF'}
            )
            summary['Phidot'] = xr.DataArray(
                data=phidot,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':'PF'}
            )
        elif tilt_correction[:3] == 'CPF':
            # get fit order
            n = int(tilt_correction[3:])
            # get rotation angles
            theta, phi = tc.get_double_rotation_angles(U, V, W)
            theta, phi_func, phidot_func = tc.get_continuous_planar_fit_angles(theta, phi, N=n)
            phi = phi_func(theta)
            phidot = phi_func(theta)
            # get corrected mean coords
            UVW_rot = tc.continuous_planar_fit_from_angles(U, V, W, theta, phi, phidot)
            
            summary['Theta'] = xr.DataArray(
                data=theta,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':tilt_correction}
            )
            summary['Phi'] = xr.DataArray(
                data=phi,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':tilt_correction}
            )
            summary['Phidot'] = xr.DataArray(
                data=phidot,
                dims=['TIMESTAMP'],
                attrs={'Tilt Correction':tilt_correction}
            )

        U_rot, V_rot, W_rot = UVW_rot[:, 0], UVW_rot[:, 1], UVW_rot[:, 2]
        
        # store mean rotations
        summary['U'] = xr.DataArray(U_rot, dims=['TIMESTAMP']).assign_attrs({'Tilt Correction':tilt_correction})
        summary['V'] = xr.DataArray(V_rot, dims=['TIMESTAMP']).assign_attrs({'Tilt Correction':tilt_correction})
        summary['W'] = xr.DataArray(W_rot, dims=['TIMESTAMP']).assign_attrs({'Tilt Correction':tilt_correction})
        summary.assign_attrs(dict({'Tilt Correction':tilt_correction}))

        if dest is not None:
            summary.to_netcdf(dest)
        
        return summary