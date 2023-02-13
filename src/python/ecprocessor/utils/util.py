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

__all__ = [
    'get_timestamp_from_fn',
    'read_campbell_file',
    'compute_summary'
]

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
    df_out : pd.DataFrame
        a dataframe containing the specified columns (renaming_dict.values()) with statistical suffixes attached and relevant statistics computed.
    '''
    
    df = (
        read_campbell_file(fn)
        .rename(columns=renaming_dict)
    )
    
    new_names = renaming_dict.values()
    mean_colnames = [f'{k}_mean' for k in new_names]
    std_colnames = [f'{k}_std' for k in new_names]
    skw_colnames = [f'{k}_skw' for k in new_names]
    krt_colnames = [f'{k}_krt' for k in new_names]

    means = df[new_names].mean().rename({k:v for k, v in zip(new_names, mean_colnames)}).values
    stds = df[new_names].std().rename({k:v for k, v in zip(new_names, std_colnames)}).values
    skws = df[new_names].skew().rename({k:v for k, v in zip(new_names, skw_colnames)}).values
    krts = df[new_names].kurt().rename({k:v for k, v in zip(new_names, krt_colnames)}).values
    
    columns = mean_colnames + std_colnames + skw_colnames + krt_colnames
    data = np.concatenate([means, stds, skws, krts])[:, np.newaxis]
    
    df_out = pd.DataFrame({k:v for k, v in zip(columns, data)})
    
    return df_out

def summarize_files(
    data_dir, glob='*',
    out=None,
    **variable_names
    ):
    '''
    Read in campbell TOA5 files from data_dir following the given glob pattern, and summarize each data file.
    Currently, this only works with files that are parseable by read_campbell_file and get_timestamp_from_fn.

    This method will attempt to standardize column names by mapping raw column names to standardized names.
    See **variable_names kwargs help for more details.

    Parameters
    ----------
    data_dir : path or str
        directory containing data files.
    glob : str (default '*')
        glob string to fild files within data dir.
    out : path or str (default None)
        Path to write the summary file to. If None (default), do not write to file.
    **variable_names: str or list of str
        Groups raw data column names under standardized names.
        Use the following format when providing variable_names:
            U='Ux_17m' will rename the column 'Ux_17m' to 'U_0' in the summary data frame.
            V=['Uy_17m', 'Uy_3m'] will rename the column 'Ux_17m' to 'U_0' and 'Ux_3m' to 'U_1'
            Ts=['T_SONIC', 'T_SONIC_3'], will rename 'T_SONIC' to 'Ts_0' and 'T_SONIC_3' to 'Ts_1' etc...
        Options are U, V, W, Ts, P, H2O, or CO2. 

    Returns
    -------
    files_df : pd.DataFrame
        dataframe providing the timestamp and filename associated with each file, plus the mean, std, skw, and krt of each provided column.
    '''

    # record file timestamps and paths
    data_dir = Path(data_dir)
    files = list(data_dir.glob(glob))
    files_df = pd.DataFrame(dict(fn=files))
    files_df['TIMESTAMP'] = list(map(get_timestamp_from_fn, files_df['fn']))
    files_df = files_df.sort_values('TIMESTAMP').set_index('TIMESTAMP')

    # create a dictionary to use for renaming raw data columns to standardized names.
    standard_names = ['U', 'V', 'W', 'Ts', 'P', 'H2O', 'CO2', 'Diag']
    renaming_dict = {}
    for new_name, old_names in variable_names.items():
        if new_name not in standard_names:
            raise Exception(f"Provided name {new_name} not in list of standard names. Must be one of {standard_names}")
        if not isinstance(old_names, list): old_names = [old_names]
        renaming_dict.update({old_name:f'{new_name}_{i}' for i, old_name in enumerate(old_names)})
    
    # compute summaries
    print(renaming_dict)
    summary_data = pd.concat([compute_summary(fn, renaming_dict) for fn in tqdm(files_df['fn'])])
    summary_data = summary_data.set_index(files_df.index)
    files_df = files_df.merge(summary_data, left_index=True, right_index=True)

    if out is not None:
        out = Path(out)
        filetype = out.suffix
        write_out = {
            '.csv':files_df.to_csv, 
            '.parquet':files_df.to_parquet, 
            '.pickle':files_df.to_pickle
        }
        write_out[filetype](out)

    return files_df