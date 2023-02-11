"""
Apply tilt corrections to either a single file or to multiple files.

"""

import argparse
from pathlib import Path
import sys
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=('''
        Apply tilt corrections to either a single file or to multiple files.
        --------------------------------
        Input files can be either .csv, .pickle, or .parquet files, but their extensions should indicate such.
        '''))
    parser.add_argument(
        '-i', '--input', 
        required=True, 
        help='path to input file or directory.'
    )
    parser.add_argument(
        '-o', '--output', 
        help='path to output file or directory', 
        default=''
    )
    parser.add_argument(
        '-ov', '--overwrite', 
        action='store_true',
        help='overwrite the input files in the case that an output is not provided.\n'
            + 'If an output file is provided, this argument is ignored.'
    )
    parser.add_argument(
        '-s', '--summary',
        help='Path to the summary file, which must contain the following:\n'
            + '* columns labeled U, V, and W'
            + '* a column labelled TIMESTAMP containing datetime information'
        default=''
    )
    parser.add_argument(
        '-m', '--method', 
        required=True, 
        choices=['DR', 'TR', 'PF', 'CPF'], 
        nargs='+',
        help='Tilt correction method to apply. Choose one or several of:\n'
            + '    DR: Double Rotation\n'
            + '    TR: Triple Rotation\n'
            + '    PF: Planar Fit\n'
            + '    CPF: Continuous Planar Fit. If CPF is specified, you should specify -n as well.'
    )
    parser.add_argument(
        '-n', '--order',
        type=int,
        nargs='*',
        help='Order of the continuous planar fit method to apply.\n' 
            + 'Order 1 is identical to the standard planar fit.\n'
            + 'No effect if CPF is not specified.\n'
            + 'If multiple CPF methods are provided, then you must provide one -n for each method.'
    ) 

    args = parser.parse_args()

    if not args.output and args.overwrite:
        args.output = args.input

    methods = {
        'DR':'double_rotation_fit_from_uvw',
        'TR': 'triple_rotation_fit_from_uvw',
        'PF':  'continuous_planar_fit_from_uvw',
        'CPF': 'continuous_planar_fit_from_uvw'
    }
    
    read_file = {
        '.parquet': pd.read_parquet,
        '.csv': pd.read_csv,
        '.pickle': pd.read_pickle
    }

    if args.summary:
        summary_file = Path(args.summary)
        file_type = summary_file.suffix
        summary = read_file[file_type](summary_file)
    
    else:

    
    # args
    # input file
    # output file
    # correction method
    # file read kwargs (columns, filetype, etc.)
    #   if csv, then you must provide pandas read arguments
    #   if parquet, then...
    #   if pickle, then...
    
    

if __name__ == '__main__':
    print(dir(Path))
    main()
    