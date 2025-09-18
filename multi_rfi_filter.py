######################################################################
# Python script to run IQRM algorithm with flagging statistics as MAD,
# replacement method as Constant (=0), and radius of 410. The code processes
# the entire data by flagging and replacing bad channels from chunks of 
# data each of roughly 2min. The code is capable of parallelizing 7 RFI 
# mitigation tasks simultaneously. This consumes a lot of CPU RAM as it 
# loads entire filterbank data in an array. The code is well tested to 
# work smoothly on any GPU node, which has CPU RAM ~750GB, with upto 50-70% 
# RAM usage for ~40min data, parallelized across 7 tasks and this took 
# almost 191min during testing. These number may vary with datasets. 
# >> Data should be of size (nfreq_sample, ntime_sample)
#
# Usage: (EXECUTE ON ANY GPU NODE ONLY)
# 	python rfi_filter.py -D /path/to/filterbank/files
# Output: 
# 	RFI mitigated filterbank files with format "*_RFI_Mitigated.fil"
#	saved at same directory as input filterbank files.
#    
# NOTE: Use appropriate env before running this script:
# $ source /lustre_archive/apps/tdsoft/env.sh
#
#  Last Update: 1st August 2025; ~ Raghav Wani
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from rich.console import Console
from iqrm import iqrm_mask
from priwo import readfil, writefil
from scipy.stats import kurtosis, skew, median_abs_deviation, iqr
from pathlib import Path
from tqdm import tqdm
import os, glob, argparse
from concurrent.futures import ProcessPoolExecutor
import gc
gc.collect()
console = Console()

def statistics(stats_ID, data):
    if stats_ID == 1:
        spectral_data = data.std(axis=0)
        stats= 'StdDev'
    elif stats_ID ==2:
        spectral_data = data.mean(axis=0)
        stats = 'Mean'
    elif stats_ID == 3:
        spectral_data = data.var(axis=0)
        stats='Variance'
    elif stats_ID == 4:
        spectral_data = kurtosis(data,axis=0)
        stats = 'Kurtosis'
    elif stats_ID == 5:
        spectral_data = skew(data,axis=0)
        stats = 'Skewness'
    elif stats_ID == 6:
        spectral_acf = []
        for x in data.T:
            x_centered = x - np.mean(x)
            numerator = np.dot(x_centered[1:], x_centered[:-1])
            denominator = np.dot(x_centered, x_centered)
            r1 = numerator / denominator if denominator != 0 else 0
            spectral_acf.append(r1)
        spectral_data = np.array(spectral_acf)
        stats = 'Acf'
    elif stats_ID == 7:
        spectral_data = median_abs_deviation(data, axis=0)
        stats = 'MAD'
    elif stats_ID == 8:
        spectral_data = iqr(data, axis=0)
        stats = 'IQR'
    else:
        raise ValueError("Invalid Statistics ID entered. Please check again.")

    return spectral_data, stats

def replacement(replace_ID, data, mask):
    rfi_mit_data = data.copy().astype(np.int64)

    if replace_ID == 1:
        # Replace with Gaussian noise
        for channum in np.where(mask)[0]:
            channel_data = data[channum]
            std = np.std(channel_data)
            median = np.median(channel_data)

            noise = np.random.normal(loc=median, scale=std, size=data.shape[1])
            rfi_mit_data[channum] = noise.astype(data.dtype)
            replace = 'Noise'

    elif replace_ID == 2:
        for channum in np.where(mask)[0]:
            channel_data = data[channum]
            median = np.median(channel_data)
            rfi_mit_data[channum] = median.astype(data.dtype)
            replace = 'Median'

    elif replace_ID == 3:
        for channum in np.where(mask)[0]:
            constant = 0 # SAME REPLACEMENT FOR ALL THE CHANNELS
            rfi_mit_data[channum] = constant
            replace = 'Constant'
    else:
        raise ValueError("Invalid replacement method choosen. Please check again.")

    return rfi_mit_data

def run_iqrm(org_data_file):
    base = os.path.basename(org_data_file)
    output_dir = os.path.dirname(org_data_file)
    basename = base.replace(".fil", "")

    print(f"Processing {basename} with process ID [{os.getpid()}]")
    meta_data, org_data = readfil(org_data_file)
    org_data = org_data.astype(np.float32)  # instead of float64
    rfi_mit_data = org_data.copy()

    time = 2 # in minutes
    chunk_size = int((time * 60 * 1000) / 1.31072)  # rounding time
    total_len = rfi_mit_data.shape[1]
    #print('Total no. of time samples: ', total_len)
    #print('Total Duration of observation (in min): ', total_len * 1.31072 * 1e-3 / 60)
    
    radius = 410 # Selected Radius value: ~10% 4096
    i = 0
    while i < total_len:
        end = min(i + chunk_size, total_len)
        chunk = rfi_mit_data[:, i:end]

        spectral_data, stats = statistics(7, data = chunk.T) # Flagging Statistics: MAD
        mask, votes = iqrm_mask(spectral_data, radius=radius)

        rfi_mit_data[:, i:end] = replacement( 3, chunk, mask) # Replacement: Constant
        i += chunk_size
        del chunk, mask, votes

    try:
        writefil(meta_data, rfi_mit_data, f"{output_dir}/{basename}_RFI_Mitigated.fil")
        del rfi_mit_data, org_data
    except Exception as e:
        print(f"[{os.getpid()}] Failed to write file: {e}")


def main():
    try:
        parser = argparse.ArgumentParser(prog=__file__)
        parser.add_argument("-D", "--input_dir", type=Path, required=True)
        args = parser.parse_args()

        console.print(f"[cyan] Input Filterbank Directory: {args.input_dir} [/cyan]", style = "bold")
        max_workers = 7
        num_files = 10
        fil_files = glob.glob(os.path.join(args.input_dir, "*.fil"))
        fil_files = [f for f in fil_files if not f.endswith("_RFI_Mitigated.fil")]
        console.print(f"📂 Found [green]{len(fil_files)}[/green] files.\n", style="bold")

        if not fil_files:
            print("No .fil files found.")
            return
        console.rule("[bold blue]🚀 Beginning RFI Mitigation using IQRM")
        console.print(f"⚙️  Starting parallel processing with [cyan]{num_files}[/cyan] files at once...\n")

        with tqdm(total=len(fil_files), desc="RFI Mitigation Progress", unit="samples") as pbar:
            i=0
            while i <= len(fil_files):
                end = min(i + num_files, len(fil_files))

                fil_files_grp = fil_files[i:end]
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    executor.map(run_iqrm, fil_files_grp)

                pbar.update(end - i)
                i += num_files

        console.print(f"\n✅ [bold green]Done with RFI mitigation.[/bold green]")
        console.print(f"📁 [bold]Check RFI-mitigated files at:[/bold] [cyan]{args.input_dir}[/cyan]")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
