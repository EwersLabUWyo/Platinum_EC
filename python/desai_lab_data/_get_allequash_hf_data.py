#### Python script to download high frequency EC data from the Desai Lab's website
#### Author: Alex Fox
#### Created: June 28, 2023

import argparse
from urllib import request
import urllib

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

def main():
    args = argparser()
    start = args.start
    end = args.end
    dest = args.dest

    dates = date_range(start, end, freq='1d')

    years = dates.year.astype(int)
    months = dates.month.astype(int)
    days = dates.day.astype(int)

    remote_files = [
        f'http://co2.aos.wisc.edu/data/Allequash/output/{year:04d}/US-ALQ_{year:04d}{month:02d}{day:02d}_L0rawdata.tar.gz' 
        for year, month, day in zip(years, months, days)
    ]

    local_files = [
        f"{dest}/{year:04d}{month:02d}{day:02d}.tar.gz"
        for year, month, day in zip(years, months, days)
    ]

    for remote, local in zip(tqdm(remote_files), local_files):
        try:
            request.urlretrieve(remote, local)
            sleep(0.2)
        except urllib.error.URLError as err:
            print(f'File {remote} not found')


if __name__ == '__main__':
    main()

