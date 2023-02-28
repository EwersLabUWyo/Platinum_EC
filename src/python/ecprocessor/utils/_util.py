"""
Utility functions for eddy covariance processing, such as functions to read campbell files and strip timestamps from their file names

Author: Alex Fox
Created: 2023-01-30

TODO: add examples
"""

from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import numpy as np
from tqdm import tqdm
import xarray as xr

def convert_units(unit):
    """
    Convert into SI base units. TEMPERATURES ARE HANDLED AS INTERVALS. e.g. 35C will be converted to 63F as an interval

    units :  list or tuple of lists or tuples
        Units associated with a value are specified as a list/tuple of lists/tuples:
            [(Unit1, exponent1), (Unit2, exponent2), ...]
        A (Unit, exponent) tuple specifies Unit**exponent
        A [(Unit1, exponent1), (Unit2, exponent2), ...] list specifies Unit1**exponent1 * Unit2**exponent2

        Example:
            [('m', 1), ('s', -1)],  # m s-1 or m/s
            [('C', 1)],  # C
            [('mg', 1), ('m', 3)]  # mg m-3 or mg/m^3

        Supported units:
            Length: m, cm, mm, in, ft, yd
            Time: s, min, hr, day, ms, um
            Mass: kg, g, mg, um, lbm
            Quantity: mol, mmol, umol, nmol
            Temperature: K, F, C
            Pressure: Pa, hPa, kPa, MPa, bar, mbar, atm, torr, inHg, mmHg
            Force: N, lbf
            Energy: J, kJ, MJ
            Power: W, kW
            Electric Potential: V, mV
            Frequency: Hz, kHz, MHz
            You can add additional unit support in the convert_units function.

    Returns
    -------
    mult : float
        Multiplicative factor

    Example
    -------
    >>> x = 35.21  # mm/day/K
    >>> mult = convert_units([('mm', 1), ('day', -1), ('K', -1)])
    >>> mult
    1.1574074074074074e-08
    >>> x*mult  # m/s/K
    4.0752314814814815e-07

    """

    mult = 1
    if unit is not None:
        unit_dict = {
            # length
            'm':1., 'cm':1e-2, 'mm':1e-3, 'in':0.0254, 'ft':0.0254*12, 'yd':0.0254*36,
            # time
            's':1., 'min':60., 'hr':3600., 'day':86400., 'ms':1e-3, 'um':1e-6,
            # mass
            'kg':1., 'g':1e-3, 'mg':1e-6, 'ug':1e-9, 'lbm':0.453592,
            # quantity
            'mol':1., 'mmol':1e-3, 'umol':1e-6, 'nmol':1e-9,
            # temperature interval
            'K':1., 'F':5/9, 'C':1,
            # pressure
            'Pa':1., 'hPa':100., 'kPa':1000., 'MPa':1e6, 'bar':1e5, 'mbar':1e2, 'atm':101325., 'torr':133.322, 'inHg':3386.39, 'mmHg':133.322,
            # force
            'N':1., 'lbf':4.44822,
            # energy
            'J':1., 'kJ':1e3, 'MJ':1e6,
            # power
            'W':1., 'kW':1e3,
            # electric potential
            'V':1., 'mV':1e-3,
            # frequency
            'Hz':1., 'kHz':1e3, 'MHz':1e6,
        }

        mult = np.array([unit_dict[u[0]]**u[1] for u in unit if u is not None]).prod()
    return mult

def get_timestamp_from_fn(fn, fmt, prefix_regex=None, suffix_regex=None):
    '''
    Given a raw high-frequency data file, get the starting timestamp from the filename.
    File names MUST contain the timestamp in a strptime-compatible format.
    
    Parameters
    ----------
    fn : str or pathlib.Path() instance
        path to file
    fmt : str
        strptime format string for the file timestamp
    prefix_regex : str (default "^")
        prefix to file timestamp as a regex string, not including parent directories. If "^" (default), then there is no prefix
    suffix_regex : str (default "$")
        suffix to the file timestamp as a regex string, including file extension. If "$" (default), then there is no suffix
    
    Returns
    -------
    timestamp : datetime.datetime

    Examples
    --------
    >>> fn = 'InputData/BBSF7m_14days_dataset/TOA5_9810.CPk_BBSF7m_10Hz1349_2022_11_17_2200.dat'
    >>> fmt = r'%Y_%m_%d_%H%M'
    >>> prefix_regex = r'^TOA5.*10Hz[0-9]+_'
    >>> suffix_regex = r'\.dat$'
    >>> get_timestamp_from_fn(fn, fmt, prefix_regex, suffix_regex)
    datetime.datetime(2022, 11, 17, 22, 0)

    >>> fn = '2022_11_17_2200'
    >>> fmt = r'%Y_%m_%d_%H%M'
    >>> get_timestamp_from_fn(fn, fmt)
    datetime.datetime(2022, 11, 17, 22, 0)
    '''
    
    # get a list of [yyyy, mm, dd, hh, MM] from filename
    if not isinstance(fn, Path().__class__):
        fn = Path(fn)
    fn = fn.name

    date_start = re.search(prefix_regex, fn).end()
    date_end = re.search(suffix_regex, fn).start()
    date_string = fn[date_start:date_end]
    timestamp = datetime.strptime(date_string, fmt)
    
    return timestamp

def read_campbell_file(fn, fmt='TOA5', **read_csv_kwargs):
    '''
    read in a campbell scientific file as a pandas dataframe

    TODO Add ability to parse TOA5 metadata
    TODO Add ability to record and convert units

    Parameters
    ----------
    fn : str or path
    fmt : str, one of 'TOA5', 'TOB3', or None
        File format to parse. If None, then use read_csv_kwargs to parse the file
    **read_csv_kwargs : keyword arguments passed to pd.read_csv()
        if fmt is provided (not None), these will be ignored

    Returns
    -------
    df : pd.DataFrame
    '''
    
    if fmt == 'TOA5' or fmt == 'TOB3':
        read_csv_kwargs = dict(
            parse_dates=['TIMESTAMP'], 
            skiprows=[0, 2, 3], 
            na_values=['NAN', '"NAN"', "'NAN'"], 
        )
    
    df = pd.read_csv(fn, **read_csv_kwargs)
    return df

def compute_summary(fn, renaming_dict, units=None,
                    **read_campbell_file_kwargs):
    '''Compute statistical summary of a high-frequency data file
    
    Parameters
    ----------
    fn: str or path
        path to file
    renaming_dict: dict
        Mapping to rename column names as they appear in the raw data file. 
        The mapping must go from raw column name --> standard column name.
        Standard column names are as follow:
            U, V, W: x, y, z windspeed components
            T_SONIC: sonic temperature
            CO2, H2O, CH4, O2, O3: gas densities or concentrations
            PA: air pressure
        
        Example: {'Ux_CSAT3B':'U'} 

        Note that multiple columns CANNOT be mapped to the same variable name. 
        If you have multiple columns representing T_SONIC, for example, you must choose just one.

    units :  dict, default None
        If none, do not convert units to SI base units. It is highly recommended that you work in SI base units.
        Mapping to specify column units as they appear in the raw data file.
        All units will be converted into SI base units. TEMPERATURES ARE HANDLED AS INTERVALS. e.g. 35C will be converted to 63F as an interval
        The mapping must go from column name --> units
        Units associated with a value are specified as a list/tuple of lists/tuples:
            [(Unit1, exponent1), (Unit2, exponent2), ...]
        A (Unit, exponent) tuple specifies Unit**exponent
        A [(Unit1, exponent1), (Unit2, exponent2), ...] list specifies Unit1**exponent1 * Unit2**exponent2

        Example:
            {
                'Ux_CSAT3B':[('m', 1), ('s', -1)],  # m s-1 or m/s
                'Ts_CSAT3B':[('C', 1)],  # C
                'rho_c_LI7500':[('mg', 1), ('m', 3)]  # mg m-3 or mg/m^3
            }

        See convert_units() for more information.

    **read_campbell_file_kwargs: dict
        keyword arguments passed to read_campbell_file()
    
    Returns
    -------
    out_names : list of variables names associated with out_data entries
    out_stats :  list of summary stats associated with out_data entries
    out_data : 1d array of values

    zip(out_names, out_stats, out_data) will iterate through matched names, statistics, and values
    '''
    
    df = (
        read_campbell_file(fn, **read_campbell_file_kwargs)
        .rename(columns=renaming_dict)
    )
    
    new_names = renaming_dict.values()
    
    # convert units if requested
    if units is not None:
        unit_mapping = {k:v for k, v in zip(new_names, units)}
        for col, unit in unit_mapping.items():
            df[col] *= convert_units(unit)

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