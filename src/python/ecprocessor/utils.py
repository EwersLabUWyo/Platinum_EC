"""
Utility functions for eddy covariance processing, such as functions to read campbell files and strip timestamps from their file names

Author: Alex Fox
Created: 2023-01-30
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

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