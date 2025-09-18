import argparse
from pathlib import Path
from plot_fil import plot_fil
import matplotlib.pyplot as plt

def main(base_dir: str):
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"❌ Error: {base_dir} does not exist.")
        return

    # Iterate over all subdirectories inside base_dir
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            input_fil = subdir / "PC_beam_01.fil"
            input_ts = subdir

            if input_fil.exists():
                print(f"▶️  Processing {subdir} ...")
                plot_fil(str(input_fil), str(input_ts), 1, 0)
                plt.close()
            else:
                print(f"⚠️  Skipping {subdir}, no PC_beam_RFI_Mitigated.fil found.")

    print("✅ All subdirectories processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all subdirectories containing PC_beam_RFI_Mitigated.fil")
    parser.add_argument("base_dir", help="Path to the base directory containing RawVisi subdirectories")
    args = parser.parse_args()

    main(args.base_dir)

