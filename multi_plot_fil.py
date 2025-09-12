import os
from pathlib import Path
from plot_fil import plot_fil 
import matplotlib.pyplot as plt

BASE_DIR = "/lustre_scratch/spotlight/data/POLCAL_20250911_021107/RawVisi" 

# Iterate over all subdirectories inside BASE_DIR
for subdir in Path(BASE_DIR).iterdir():
    if subdir.is_dir():
        input_fil = subdir / "PC_beam_RFI_Mitigated.fil"
        input_ts = subdir

        if input_fil.exists():
            print(f"▶️  Processing {subdir} ...")
            plot_fil(str(input_fil), str(input_ts), 1, 0)
            plt.close()
        else:
            print(f"⚠️  Skipping {subdir}, no PC_beam.fil found.")
print("✅ All subdirectories processed.")

