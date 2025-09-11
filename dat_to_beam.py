###########################################################################
# Python script to convert PC beam dat file formed from visibilities into
# filterbank file to make it useful for further time-domain analysis using
# softwares like PRESTO. This script was made to estimate the ToA offset
# between SPOTLIGHT's PC beam and visibility data, to ensure real-time
# imaging of triggered burst(s)
#
# Last Modified: 11 Sept 2025; Raghav
###########################################################################

import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.time import Time
from priwo import *
import os


def load_visibility_header(filename):
    """Read visibility header and return metadata + number of records."""
    with open(filename, "rb") as f:
        header_format = "<5di"  # little-endian: 5 doubles + 1 int
        header_size = struct.calcsize(header_format)
        header_bytes = f.read(header_size)

        freq_start, freq_end, channel_width, integration_time, processing_timestamp, channels = struct.unpack(
            header_format, header_bytes
        )

        mjd_time = Time(processing_timestamp, format='unix').mjd

        # Determine file size
        f.seek(0, 2)
        file_size = f.tell()
        data_size = file_size - header_size
        record_size = 8 + channels * 4  # 8 bytes timestamp + N*4-byte floats
        num_records = data_size // record_size

    header = {
        "freq_start": freq_start,
        "freq_end": freq_end,
        "channel_width": channel_width,
        "integration_time": integration_time,
        "processing_timestamp": processing_timestamp,
        "processing_mjd": mjd_time,
        "channels": channels,
        "header_size": header_size,
        "num_records": num_records,
    }
    return header


def load_visibility_data(filename, header):
    """Load visibility data arrays: timestamps and power data."""
    num_records = header["num_records"]
    channels = header["channels"]
    header_size = header["header_size"]

    timestamps = np.zeros(num_records, dtype=np.float64)
    power_data = np.zeros((num_records, channels), dtype=np.float32)

    with open(filename, "rb") as f:
        f.seek(header_size)  # skip header
        for i in range(num_records):
            timestamps[i] = struct.unpack("<d", f.read(8))[0]
            power_data[i, :] = np.frombuffer(f.read(channels * 4), dtype="<f4")

    return timestamps, power_data


def plot_dynamic_spectrum(power_data):
    """Generate dynamic spectrum plots."""
    org_sub_data = power_data.T
    org_time_series = org_sub_data.sum(axis=0)
    org_freq_series = org_sub_data.sum(axis=1)

    frequencies = np.arange(1, len(org_freq_series) + 1)
    times = np.arange(1, len(org_time_series) + 1) * 1.31072 * 0.001

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[20, 4], height_ratios=[4, 1], wspace=0.005, hspace=0.005)

    # Dynamic spectrum plot
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(org_sub_data, aspect='auto', cmap='viridis',
                    extent=[times[0], times[-1], frequencies[-1], frequencies[0]])

    # Time series plot
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(times, org_time_series, color='black', linewidth=1)
    ax1.set_xlabel("Time Samples")
    ax1.set_ylabel("Power")

    # Frequency series plot
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2.plot(org_freq_series, frequencies, color='black', linewidth=1)
    ax2.set_xlabel("Power")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel("Frequency channel")

    cax = fig.add_axes([0.055, 0.28, 0.01, 0.6])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("Power")
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    plt.suptitle("Dynamic Spectrum", fontsize=18)
    plt.show()


def create_filterbank(fil_file, power_data, out_dir, band):
    """Convert power data into filterbank and save as PC_beam.fil."""
    m, d = readfil(fil_file)

    if m['nbits'] != 8:
        raise ValueError("Invalid bit size detected")
    else:
        m['nbits'] = 32

    if int(band) == 4:
        print("Its band 4! Flipping..")
        power_data = np.fliplr(power_data)

    power_data = power_data.T  # (n_freq, n_time)
    writefil(m, power_data, f"{out_dir}/PC_beam.fil")

    print('=' * 50)
    print(f"PC beam from visibility stored as filterbank file (PC_beam.fil) at {out_dir}")


def main(filename, fil_file, out_dir, band, plot=False):
    base_visi = os.path.basename(filename)
    base_beam = os.path.basename(fil_file)
    print('=' * 50)
    print(f"Processing Visibility data from {base_visi}\nProcessing Spotlight beam data from {base_beam}")

    # Load header and data
    header = load_visibility_header(filename)
    timestamps, power_data = load_visibility_data(filename, header)

    print("=" * 50)
    print("Header Info:")
    for k, v in header.items():
        print(f"  {k}: {v}")
    print('\nMin and Max Power values:', np.min(power_data), np.max(power_data))
    print("Data Shapes:")
    print("  Timestamps:", timestamps.shape)
    print("  Power Data:", power_data.shape)

    # Optional plotting
    if plot:
        plot_dynamic_spectrum(power_data)

    # Create filterbank
    create_filterbank(fil_file, power_data, out_dir, band)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert PC beam dat file into filterbank format.")
    parser.add_argument("-d", "--dat_file", required=True, help="Path for dat file (Visibility Data)")
    parser.add_argument("-f", "--fil_file", required=True, help="Path for filterbank file (Spotlight Beam Data)")
    parser.add_argument("-o", "--out_dir", required=True, help="Path for output filterbank file (Spotligth Visibility data)")
    parser.add_argument("-b", "--band", required=True, help="Freq band of observation")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of dynamic spectrum")
    args = parser.parse_args()

    main(args.dat_file, args.fil_file, args.out_dir, args.band, plot=args.plot)

