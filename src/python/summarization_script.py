from ecprocessor.utils import summarize_files
import argparse

'''
Callable script for summarizing data files. This is the first step when processing new data
'''

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Read in provided files and summarize their data into a single dataframe for later use.'
        )
    parser.add_argument('-i', '--idir', required=True, help='TOA5 file directory. Currently only supports TOA5 files with a timestamp column')
    parser.add_argument('-d', '--dest', required=True, help='Output file name', default=None)
    # this method of providing file metadata could be improved
    parser.add_argument(
        '-r', '--renaming_dict', required=True, 
        help='Renaming scheme for raw file names.\n'
            + '    Example: Ux_3m=U,Uy_3m=V,Ts_17m=Ts\n'
            + 'Refer to utils.summarize_files for further documentation.'
    )
    parser.add_argument(
        '--verbose', help='Verbose output', action='store_true'
    )

    args = parser.parse_args()
    
    try: Path(args.dest).mkdir()
    except: pass
    
    # turn variable names into args that can be passed to summarize_files()
    renaming_dict = {v.split('=')[0]:v.split('=')[1] for v in args.renaming_dict.split(',')}
    summarize_files(data_dir=args.idir, dest=args.dest, verbose=args.verbose, renaming_dict=renaming_dict)
    
    print(args.dest)

if __name__ == '__main__':
    main()