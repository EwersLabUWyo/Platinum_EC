#### Python script to download biomet data from the Desai Lab's website
#### Author: Alex Fox
#### Created: June 28, 2023

import argparse
from urllib import request
import urllib
from pathlib import Path

from pandas import date_range
from time import sleep
from tqdm import tqdm

def argparser():
    parser = argparse.ArgumentParser(
        description="Get data from Ankur Desai's Website",
    )
    parser.add_argument('--dest', help='Directory to download files to', required=True)
    parser.add_argument("--start", help="Start date, in yyyy-mm-dd format", required=True)
    parser.add_argument("--end", help='End date, in yyyy-mm-dd format', required=True)
    args = parser.parse_args()
    return args

def download(start, end, dest):

    dates = date_range(start, end, freq='1d')

    years = dates.year.astype(int)
    months = dates.month.astype(int)
    days = dates.day.astype(int)
    doys = dates.dayofyear.astype(int)
    remote_files = [
        f'http://co2.aos.wisc.edu/data/Allequash/{year:04d}/{year:04d}{month:02d}{day:02d}/{doy:03d}biomet.data'
        for doy, year, month, day in zip(doys, years, months, days)
    ]

    local_files = [
        f"{dest}/biomet_{year:04d}{month:02d}{day:02d}.data"
        for year, month, day in zip(years, months, days)
    ]

    for remote, local in zip(tqdm(remote_files), local_files):
        try:
            request.urlretrieve(remote, local)
            sleep(0.2)
        except urllib.error.URLError as err:
            pass

    return local_files

# def aggregate(files, dest):
#     # get header
#     with open(files[0], 'r') as f:
#         lines = f.readlines()
#         header = lines[:2]
    
#     # geat data
#     data = [['-9999,'*10]*48]*len(files)
#     for i, fn in enumerate(files):
#         with open(fn, 'r') as f:
#             lines = f.readlines()
#             data[i] = lines[2:]
    
#     # write data
#     out_file = Path(dest) / 'biomet.csv'
#     with open(dest, 'w') as f:
#         for ln in header:
#             f.write(ln)
#         for day in data:
#             for ln in day:
#                 f.write(ln)
    
def main():

    args = argparser()
    start = args.start
    end = args.end
    dest = args.dest

    local_files = download(start, end, dest)
    # aggregated_biomet = aggregate(local_files, dest)


if __name__ == '__main__':
    main()

