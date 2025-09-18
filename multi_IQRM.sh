#!/bin/bash

BASE_DIR=$1

for subdir in "$BASE_DIR"/*/; do
    output_file="$subdir/PC_beam_RFI_Mitigated.fil"

    if [ -f "$output_file" ]; then
        echo "⚠️  Skipping $subdir (2RFI mitigated file already exists)"
        continue
    fi

    echo "▶️  Processing $subdir ..."
    python multi_rfi_filter.py -D $subdir

done

echo "✅ All subdirectories processed."

