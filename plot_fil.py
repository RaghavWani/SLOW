##################################################################
# Python script to plot the data from filterbank file. This plots:
# --Dyanmic spectrum plot (freq-time)
# --Timeseries plot (power-time)
# --Frequency series plot (freq-power)
# Source relevant env before running this script
#        (/lustre_archive/apps/tdsoft/env.sh)
#
#  Date: 09th Sept 2025; ~ Raghav Wani
##################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import pytz
from astropy.time import Time
import warnings
from priwo import readfil
import os
from astropy.visualization import ZScaleInterval

tz_ist = pytz.timezone("Asia/Kolkata")
def centisec_formatter(x, pos):
    dt = mdates.num2date(x, tz=tz_ist)
    return f"{dt:%H:%M:%S}.{int(dt.microsecond/10000):02d}"

def plot_fil(fil_data_file_path, t_start, savefig, showfig):
    m, org_data = readfil(fil_data_file_path)

    #org_sub_data = org_data  # shape: (n_freq, n_time)
    #print(np.shape(org_sub_data))
    org_sub_data = org_data[200:3800, :]  # shape: (n_freq, n_time)
    org_time_series = org_sub_data.sum(axis=0)  # shape: (n_time,)
    org_freq_series = org_sub_data.sum(axis=1)  # shape: (n_freq,)
    
    frequencies = [i for i in range(1,len(org_freq_series)+1)]
    times = [(i*1.31072/1000) + t_start for i in range(1,len(org_time_series)+1)]
    times = Time(times, format = 'unix')
    times = times.to_datetime(timezone=pytz.timezone("Asia/Kolkata"))
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(org_sub_data)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[20, 4], height_ratios=[4, 1], wspace=0.005, hspace=0.005)

    # Dynamic spectrum plot
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(org_sub_data, aspect='auto', cmap='viridis',
                    extent=[times[0],times[-1], frequencies[-1], frequencies[0]], vmin=vmin, vmax=vmax)

    # Time series plot
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(times, org_time_series,  color='black', linewidth=1)
    ax1.set_xlabel("Time (in IST)")
    ax1.set_ylabel("Power")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(centisec_formatter))
    
    # Frequency series plot
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2.plot(org_freq_series, frequencies, color='black', linewidth=1)
    ax2.set_xlabel("Power")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel("Frequency channel")
    ax2.set_xlim(max(0,np.unique(org_freq_series)[1]), )

    cax = fig.add_axes([0.055, 0.28, 0.01, 0.6])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label("Power")
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    plt.suptitle("Dynamic Spectrum", fontsize=18)
    if savefig == True:
        directory = os.path.dirname(fil_data_file_path)
        base = fil_data_file_path.replace(".fil", "")
        plt.savefig(f"{base}.fil.png")
    if showfig != False:
        plt.show()

plot_fil("../9Sept2025/PC_beam_RFI_Mitigated.fil", 1757367486.709775, 0, 1)
