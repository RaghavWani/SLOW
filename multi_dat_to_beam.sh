#!/bin/bash
# Script: run_make_pc_beam.sh

BASE_DIR=$1
band=$2
input_fil="/lustre_data/spotlight/data/TST3093_20250909_022047/FilData/J0332+5434_20250909_030222/BM0.fil"

for subdir in "$BASE_DIR"/*/; do
    output_file="$subdir/22ant.dat"

    echo "▶️  Processing $subdir ..."
    python dat_to_beam.py \
	-d "$subdir"/22ant.dat \
	-f "$input_fil"\
	-o "$subdir"\
	-b $band 

done

echo "✅ All subdirectories processed."

