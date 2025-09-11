#!/bin/bash
# Script: run_make_pc_beam.sh

BASE_DIR="/lustre_scratch/spotlight/data/TST3093_20250909_022047/RawVisi"
input_fil="/lustre_data/spotlight/data/TST3093_20250909_022047/FilData/J0332+5434_20250909_030222/BM0.fil"

for subdir in "$BASE_DIR"/*/; do
    output_file="$subdir/22ant.dat"

    echo "▶️  Processing $subdir ..."
    python dat_to_beam.py \
	--dat_file "$subdir"/22ant.dat \
	--fil_file "$input_fil"\
	--out_dir "$subdir"\
	--band 4

done

echo "✅ All subdirectories processed."

