"""
Utility functions for eddy covariance processing. These functions mostly handle i/o processing.

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
            Temperature: C, K, F
            Pressure: Pa, hPa, kPa, MPa, bar, mbar, atm, torr, inHg, mmHg
            Force: N, lbf
            Energy: J, kJ, MJ
            Power: W, kW
            Electric Potential: V, mV
            Frequency: Hz, kHz, MHz
            Fractions: fraction, pct (fraction ranges from 0 to 1, and is the base unit.)
            Unitless: unitless
            You can add additional unit support in the convert_units function.

    Returns
    -------
    mult : float
        Multiplicative factor
    SI_unit : str
        Human readable unit

    Example
    -------
    >>> x = 35.21  # mm/day/K
    >>> mult, SI_unit = convert_units([('mm', 1), ('day', -1), ('K', -1)])
    >>> mult
    1.1574074074074074e-08
    >>> SI_unit
    'C-1.000 m1.000 s-1.000
    >>> x*mult  # m/s/K
    4.0752314814814815e-07

    """

    mult = 1
    if unit is not None:
        unit_dict = {
            # length
            'm':(1., 'm'), 'cm':(1e-2, 'm'), 'mm':(1e-3, 'm'), 'in':(0.0254, 'm'), 'ft':(0.0254*12, 'm'), 'yd':(0.0254*36, 'm'),
            # time
            's':(1., 's'), 'min':(60., 's'), 'hr':(3600., 's'), 'day':(86400., 's'), 'ms':(1e-3, 's'), 'us':(1e-6, 's'),
            # mass
            'kg':(1., 'kg'), 'g':(1e-3, 'kg'), 'mg':(1e-6, 'kg'), 'ug':(1e-9, 'kg'), 'lbm':(0.453592, 'kg'),
            # quantity
            'mol':(1., 'mol'), 'mmol':(1e-3, 'mol'), 'umol':(1e-6, 'mol'), 'nmol':(1e-9, 'mol'),
            # temperature interval
            'K':(1., 'C'), 'F':(5/9, 'C'), 'C':(1, 'C'),
            # pressure
            'Pa':(1., 'Pa'), 'hPa':(100., 'Pa'), 'kPa':(1000., 'Pa'), 'MPa':(1e6, 'Pa'), 'bar':(1e5, 'Pa'), 'mbar':(1e2, 'Pa'), 'atm':(101325., 'Pa'), 'torr':(133.322, 'Pa'), 'inHg':(3386.39, 'Pa'), 'mmHg':(133.322, 'Pa'),
            # force
            'N':(1., 'N'), 'lbf':(4.44822, 'N'),
            # energy
            'J':(1., 'J'), 'kJ':(1e3, 'J'), 'MJ':(1e6, 'J'),
            # power
            'W':(1., 'W'), 'kW':(1e3, 'W'),
            # electric potential
            'V':(1., 'V'), 'mV':(1e-3, 'V'),
            # frequency
            'Hz':(1., 'Hz'), 'kHz':(1e3, 'Hz'), 'MHz':(1e6, 'Hz'),
            # proportion
            'fraction': (1., 'fraction'), 'pct':(0.01, 'fraction'),
            # unitless
            'unitless': (1., 'unitless'),
        }

        # multiplicative factor
        if unit is not None:
            mult = np.array([unit_dict[u[0]][0]**u[1] for u in unit]).prod()
            # human-readable unit
            SI_unit = ' '.join(sorted([f'{unit_dict[u[0]][1]}{u[1]:.3f}' for u in unit]))
            return mult, SI_unit
        return 1, 'Unknown'

def get_timestamp_from_fn(fn, fmt='%Y_%m_%d_%H%M', prefix_regex='^', suffix_regex='$'):
    '''
    Given a string or file path containing a timestamp, extract the timestamp.
    File names MUST contain the timestamp in a strptime-compatible format.
    
    Parameters
    ----------
    fn : str or pathlib.Path() instance
        string or filepath containing the timestamp to extract.
    fmt : str (default '%Y_%m_%d_%H%M)
        strptime format string for the file timestamp. See https://docs.python.org/3/library/datetime.html for more details.
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
    >>> # strptime format string
    >>> fmt = r'%Y_%m_%d_%H%M'
    >>> # regex to identify character preceeding the timestamp. Note that we exclude any parent directories in the prefix.
    >>> prefix_regex = r'^TOA5.*10Hz[0-9]+_'
    >>> # regex to identify character appearing after the timestamp
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
        if fmt is provided (not None), these will be ignored.

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

def compute_summary(fn, 
                    renaming_dict, 
                    units=None,
                    **read_campbell_file_kwargs):
    '''Compute statistical summary of a high-frequency data file in SI units
    
    Parameters
    ----------
    fn: str or path
        path to file
    renaming_dict: dict
        Mapping to rename column names as they appear in the raw data file. 
        Columns not specified in this mapping will not be included.
        The mapping must go from raw column name --> standard column name.
        Standard column names are as follow:
            U, V, W: x, y, z windspeed components
            T_SONIC: sonic temperature
            CO2, H2O, CH4, O2, O3: gas densities or concentrations
            PA: air pressure
        
        Example: {'Ux_CSAT3B':'U'} 

        Note that multiple columns CANNOT be mapped to the same variable name. 
        If you have multiple columns representing T_SONIC, for example, you must choose just one.

    units :  dict
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
    out_names : list of variables names associated with out_data entries in dimension 1
    out_units : list of unit names associated with out_names
    out_stats :  list of summary stats associated with out_data entries in dimension 0
    out_data : 2d array of data values with dimensions (stat, name)
    '''

    read_campbell_file_kwargs.update({'usecols':list(renaming_dict.values())})
    df = (
        read_campbell_file(fn, **read_campbell_file_kwargs)
        .rename(columns=renaming_dict)
    )
    
    new_names = renaming_dict.values()
    
    # convert units if requested
    unit_mapping = {k:v for k, v in zip(new_names, units)}
    out_units = []
    for col, unit in unit_mapping.items():
        mult, unit_name = convert_units(unit)
        df[col] *= mult
        out_units.append(unit_name)

    means = list(df[new_names].mean().values)
    stds = list(df[new_names].std().values)
    skws = list(df[new_names].skew().values)
    krts = list(df[new_names].kurt().values)
    frac = list((df[new_names].count()/df.shape[0]).values)   # fraction of non-na values

    out_names = list(new_names)
    out_stats = ['mean', 'std', 'skew', 'kurt', 'frac']
    out_data = np.stack((means, stds, skws, krts, frac))
    
    return out_names, out_units, out_stats, out_data

def summarize_files(
    data_dir, renaming_dict, 
    glob='*',
    dest=None,
    verbose=False,
    fmt='%Y_%m_%d_%H%M', 
    prefix_regex='^', 
    suffix_regex='$',
    **read_campbell_file_kwargs,
    ):
    '''
    Read in csv files from data_dir following the given glob pattern, and summarize each data file.
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
    fmt, prefix_regex, suffix_regex : args passed to get_timestamp_from_fn. See get_timestamp_from_fn() for more information.
    **read_campbell_file_kwargs : kwargs passed to read_campbell_file

    Returns
    -------
    summary_data : xr.Dataset
        dataset summarizing important raw file features:
        - Coords: TIMESTAMP
        - Data Variables: 
            * fn, dims=(TIMESTAMP)
            * User-specified variables, dims=(TIMESTAMP, Stat), attrs={'Units':units}
    '''

    # find files
    data_dir = Path(data_dir)
    files = list(data_dir.absolute().glob(glob))
    # make dataframe from files
    files_df = pd.DataFrame(dict(fn=files))
    # add file timestamps to dataframe
    files_df['TIMESTAMP'] = list(map(
        lambda fn: get_timestamp_from_fn(fn, fmt, prefix_regex, suffix_regex),
        files_df['fn']
        ))
    # sort by timestamp, set index
    files_df = files_df.sort_values('TIMESTAMP').set_index('TIMESTAMP')
    # record file summary data and metadata for each file
    iterfiles = files_df['fn']
    if verbose:
        iterfiles = tqdm(files_df['fn'])
    out = [compute_summary(fn, renaming_dict, **read_campbell_file_kwargs) for fn in iterfiles]
    out_names = out[0][0]
    out_units = out[0][1]
    out_stats = out[0][2]
    # dims (name, stat, file)
    out_data = np.transpose(np.array([i[3] for i in out]), axes=(2, 0, 1))
    
    # combine into a dataset with metadata
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
                attrs=dict(Units=unit)
            )
            for var, unit, name in zip(out_data, out_units, out_names)
        }
    )
    
    summary_data['fn'] = xr.DataArray([str(fn) for fn in files_df['fn']], dims=['TIMESTAMP'])

    if dest is not None:
        summary_data.to_netcdf(path=Path(dest))
    return summary_data

def extract_stat_from_summary(fn, )

def compute_aggregate_metrics(
    summary, 
    dest=None,
    tilt_correction=None
):
    '''
    Computes the requrested aggregatte metrics from a summary file or dataset returned by summarize_files
    Currently only 

    TODO this is pretty hard to follow. Lots of things to change if we need to add a new method. If we add a new tilt correction, we need to update this list too.

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
    assert isinstance(summary, xr.Dataset), f'got type(summary)={type(summary)}. You need to pass either an xr.Dataset object or a file path to one!'

    if tilt_correction:
        from .. import tiltcorrections as tc
        U = summary['U'].sel(Stat='mean').data
        V = summary['V'].sel(Stat='mean').data
        W = summary['W'].sel(Stat='mean').data

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