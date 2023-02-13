from utils import summarize_files
import argparse

'''
Callable script for summarizing data files
'''

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Read in provided files and summarize their data into a single dataframe for later use.'
        )
    parser.add_argument('-d', '--dir', required=True, help='TOA5 file directory')
    parser.add_argument('-g', '--glob', required=True, help='glob string to find files in directory', default='*')
    parser.add_argument('-o', '--out', required=True, help='Output file name', default=None)
    parser.add_argument(
        '--variable_names', required=True, 
        help='Standard column names followed by names as they appear in the raw data file.\n'
            + 'Separate column types with semicolons, provide a mapping with =\n'
            + '    Example: U=Ux_3m;V=Uy_3m;Ts=Ts_17m\n'
            + 'Refer to utils.summarize_files for further documentation.'
    )

    args = parser.parse_args()

    # turn variable names into args that can be passed to summarize_files()
    variable_names = {v.split('=')[0]:v.split('=')[1] for v in args.variable_names.split(';')}
    summarize_files(args.dir, args.glob, args.out, **variable_names)
    
    print(args.out)

if __name__ == '__main__':
    main()