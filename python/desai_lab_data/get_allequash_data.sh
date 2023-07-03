#!/bin/bash

###############
# ¡Change me! #
###############
# set the download location for the high frequency and biomet data, and set the location of the python scripts.
script_dir="/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Platinum_EC/python/desai_lab_data"
fastdatadir="/Users/alex/Documents/Data/Platinum_EC/Allequash/fast"
biomdatadir="/Users/alex/Documents/Data/Platinum_EC/Allequash/biomet"
# set the date range to download files for
start="2020-06-21"
end="2020-07-21"

#####################
# ¡Don't Change me! #
#####################
# download the data, could take up to 30 minutes for a year of data
echo "Downloading High Frequency Files"

# python "$script_dir/_get_allequash_hf_data.py" --dest $fastdatadir --start $start --end $end

# find .tar.gz files in the data directory
files=$(ls $fastdatadir/*.tar.gz)
echo "Extracting Files"
n_files=$(ls $fastdatadir/*.tar.gz | wc -l)
counter=0
echo ""
for FILE in $files
do
    ((counter++))
    echo -en "\r\033[K $counter of $n_files: $FILE"
    # unzip
    tar -xf $FILE -C $fastdatadir
    # copy half hourly files to the main data directory
    mv $fastdatadir/air/incoming/Allequash/*/*/*_ts_data_*.dat $fastdatadir
    # delete the zip file
    rm $FILE 
done
rm -r $fastdatadir/air


echo "Downloading Biomet Files"
python "$script_dir/_get_allequash_biomet_data.py" --dest $biomdatadir --start $start --end $end

echo "Done"