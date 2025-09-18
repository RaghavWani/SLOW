#!/bin/bash
# Script: run_make_pc_beam.sh

BASE_DIR=$1
band=$2
ant=$3

if [ "$band" -eq 3 ]; then
    freq_start=$((500 * 1000000))
    freq_end=$((300 * 1000000))
elif [ "$band" -eq 4 ]; then
    freq_start=$((550 * 1000000))
    freq_end=$((750 * 1000000))
elif [ "$band" -eq 5 ]; then
    freq_start=$((1460 * 1000000))
    freq_end=$((1260 * 1000000))
else
    echo "❌ Error: Unknown band '$band'. Valid options are 3, 4, or 5."
    exit 1
fi

for subdir in "$BASE_DIR"/*/; do
    output_file="$subdir/22ant.dat"

    if [ -f "$output_file" ]; then
        echo "⚠️  Skipping $subdir (22ant.dat already exists)"
        continue
    fi

    echo "▶️  Processing $subdir ..."
    python3 make_pc_beam.py \
        -f "$subdir"/Visi-R*.raw \
        --freq-start "$freq_start" \
        --freq-end "$freq_end" \
        --channels 4096 \
        --output "$output_file" \
        --antmask $ant \
        --no-bandpass \
        --no-baseline
done

echo "✅ All subdirectories processed."

