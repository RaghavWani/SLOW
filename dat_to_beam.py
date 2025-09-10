###########################################################################
# Python script to convert PC beam dat file formed from visibilities into
# filterbank file to make it useful for further time-domain analysis using 
# softwares like PRESTO. This script was made to estimate the ToA offset 
# between SPOTLIGHT's PC beam and visibility data, to ensure real-time
# imaging of triggered burst(s)
#
# Last Modified: 04 August 2025; Raghav 
###########################################################################

import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.time import Time
from priwo import *
import os

# Basic Info --------------------------------------------------------------
filename = input("Enter path for dat file (Visibility Data): ")
fil_file = input("Enter path for filterbank file (Spotlight Beam Data): ")
band = input("Enter freq band of observation: ")
timestp = input("Unix from ts file: ") # Unix time from ts file
mjd_timestp = Time(timestp, format = 'unix').mjd

base_visi = os.path.basename(filename)
base_beam = os.path.basename(fil_file)
print('='*50)
print(f"Processing Visibility data from {base_visi}\nProcessing Spotlight beam data from {base_beam}")
print(f"Start MJD (unix time): {mjd_timestp}")

# Load Visibility Data --------------------------------------------------------------

#Header
with open(filename, "rb") as f:
    header_format = "<5di"  # little-endian: 5 doubles + 1 int (4 bytes)
    header_size = struct.calcsize(header_format)
    header_bytes = f.read(header_size)

    freq_start, freq_end, channel_width, integration_time, processing_timestamp, channels = struct.unpack(header_format, header_bytes)
    
    mjd_time = Time(processing_timestamp, format = 'unix').mjd
    print("="*50)
    print("Header Info:")
    print(f"  Frequency Range: {freq_start} - {freq_end} Hz")
    print(f"  Channel Width: {channel_width} Hz")
    print(f"  Integration Time per Record: {integration_time} s")
    print(f"  Processing Timestamp: {processing_timestamp}")
    print(f"  Processing Timestamp (MJD): {mjd_time} s")
    print(f"  Channels: {channels}")

    #Determine file size
    f.seek(0, 2)  
    file_size = f.tell()
    data_size = file_size - header_size
    record_size = 8 + channels * 4  # 8 bytes timestamp + 4096 * 4-byte floats
    num_records = data_size // record_size
    print(f"  Total Records: {num_records}")
    print(f"  Header Size: {header_size}")

#Main data
timestamps = np.zeros(num_records, dtype=np.float64)
power_data = np.zeros((num_records, channels), dtype=np.float32)

with open(filename, "rb") as f:
    f.seek(header_size)  # skip header

    for i in range(num_records):
        timestamps[i] = struct.unpack("<d", f.read(8))[0]
        power_data[i, :] = np.frombuffer(f.read(channels * 4), dtype="<f4")


print('\nMin and Max Power values:', np.min(power_data), np.max(power_data))
print("Data Shapes:")
print("  Timestamps:", timestamps.shape)
print("  Power Data:", power_data.shape)  # (num_records, 4096)

# Plotting dynamic spectrum --------------------------------------------------------------
org_sub_data = power_data.T  # shape: (n_freq, n_time)
org_time_series = org_sub_data.sum(axis=0)  # shape: (n_time,)
org_freq_series = org_sub_data.sum(axis=1)  # shape: (n_freq,)
print(org_sub_data.shape, '\nExpected shape is (n_freq, n_time)')
print('\nMin and Max Power:', np.min(org_sub_data), np.max(org_sub_data))

frequencies = [i for i in range(1,len(org_freq_series)+1)]
times = [i*1.31072*0.001 for i in range(1,len(org_time_series)+1)]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[20, 4], height_ratios=[4, 1], wspace=0.005, hspace=0.005)

# Dynamic spectrum plot
ax0 = fig.add_subplot(gs[0, 0])
im = ax0.imshow(org_sub_data, aspect='auto', cmap='viridis',
                    extent=[times[0],times[-1], frequencies[-1], frequencies[0]])
# Time series plot
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
ax1.plot(times, org_time_series,  color='black', linewidth=1)
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

# Creating filterbank --------------------------------------------------------------
m, d = readfil(fil_file)
if m['nbits'] != 8:
    raise ValueError("Invalid bit size detected")
else:
    m['nbits'] = 32

if int(band) == 4:
    print("Its band 4! Flipping..")
    power_data = np.fliplr(power_data) # Flipping as Band 4 is inverted

power_data = power_data.T # Transpose to convert into shape (n_freq_samples, n_time_samples)
out_dir = os.path.dirname(fil_file)
writefil(m, power_data, f"{out_dir}/PC_beam.fil")

print('='*50)
print(f"PC beam from visibility stored as filterbank file (PC_beam.fil) at {out_dir}")

