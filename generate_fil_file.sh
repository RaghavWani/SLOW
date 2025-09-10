#!/bin/bash

# ./generate_fil_file.sh [/path/to/raw/file] [/path/to/hdr/file]

present_working_dir=$(pwd)
echo -e "Present working directory is: $present_working_dir"

cp $2 $(dirname $1)

cd $(dirname $1)
echo -e "Changed the directory to: $(pwd)\n"

filterbank="/lustre_archive/apps/tdsoft/usr/bin/filterbank"

f_name=$(basename $1)
hdr_name=$(basename $2)

echo -e "Input raw file name is: $f_name"
echo -e "Input hdr file name is: $hdr_name\n"

if test -f ${f_name}.gmrt_dat; then
    rm -r ${f_name}.gmrt_dat
fi

echo -e "Removed existing .gmrt_dat files.\n"

var3=$(ln -s ${f_name} ${f_name}.gmrt_dat)

echo -e "Linked the given raw file to a .gmrt_dat file.\n"

if test -f ${f_name}.gmrt_hdr; then
    rm -r ${f_name}.gmrt_hdr
fi

echo -e "Removed existing .gmrt_hdr files.\n"

var4=$(ln -s ${hdr_name} ${f_name}.gmrt_hdr)

echo -e "Linked the given hdr file to a .gmrt_hdr file.\n"

var5=$($filterbank ${f_name}.gmrt_dat > ${f_name}.fil)

echo -e "\nGenerated the fileterbank file: ${f_name}.fil\n"

#var6=$(rm -r ${f_name}.gmrt_dat ${f_name}.gmrt_hdr)

echo -e "Removed the generated .gmrt_dat and .gmrt_hdr files. Changing to the original working directory.\n"

cd $present_working_dir
