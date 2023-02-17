"""
Step 1 of applying tilt corrections. 
This will compute necessary aggregate metrics for tilt corrections.
This file generates the output that should be fed into step 2
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import xarray as xr

from ecprocessor.utils import compute_aggregate_metrics

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=('''
        Step 1 of applying tilt corrections. 
        This will compute necessary aggregate metrics for tilt corrections
        '''))
    parser.add_argument(
        '-d', '--dest', 
        help='path to output directory', 
        default=''
    )
    parser.add_argument(
        '-s', '--summary',
        required=True,
        help='Path to the summary file generated by utils.summarize_files()'
    )
    parser.add_argument(
        '-m', '--method', 
        required=True, 
        help='Tilt correction method to apply. Choose one of:\n'
            + '    DR: Double Rotation\n'
            + '    TR: Triple Rotation\n'
            + '    PF: Planar Fit\n'
            + '    CPFn: n-th order Continuous Planar Fit. Replace "n" with the desired order of the fourier series fit. n=1 is identical to the standard Planar Fit. For example, CPF5 would give a 5th order fit, and CPF15 would give a 15th order fit.'
    ) 
    
    args = parser.parse_args()

    compute_aggregate_metrics(summary=args.summary, dest=args.dest, tilt_correction=args.method)
    print(args.dest)

if __name__ == '__main__':
    main()
    