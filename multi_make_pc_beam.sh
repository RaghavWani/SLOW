#!/bin/bash
# Script: run_make_pc_beam.sh

BASE_DIR="/lustre_scratch/spotlight/data/TST3093_20250909_022047/RawVisi"

for subdir in "$BASE_DIR"/*/; do
    output_file="$subdir/22ant.dat"

    if [ -f "$output_file" ]; then
        echo "⚠️  Skipping $subdir (22ant.dat already exists)"
        continue
    fi

    echo "▶️  Processing $subdir ..."
    python3 make_pc_beam.py \
        -f "$subdir"/Visi-R*.raw \
        --freq-start 550e6 \
        --freq-end 750e6 \
        --channels 4096 \
        --output "$output_file" \
        --antmask 0x738ffff \
        --no-bandpass \
        --no-baseline
done

echo "✅ All subdirectories processed."

