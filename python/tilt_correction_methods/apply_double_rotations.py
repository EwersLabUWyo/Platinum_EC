"""
Run the double rotation correction on the given file
Author: Alex Fox
Created: 2023-01-30
"""

import argparse
import sys

def main():
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the double rotation correction on the given file')
    parser.add_argument('--infile', help='path to the input file.')
    parser.add_argument('--outfile', help='path to the output file.')
    args = parser.parse_args()
    infile, outfile = args.infile, args.outfile
    
    main()