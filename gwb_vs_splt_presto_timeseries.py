######################################################################
# Python script to find burst ToA offset primarily between SPOTLIGHT's
# PC beam and GWB's PC beam. The code expects dedispersed timeseries 
# data in the form of binary dat file, obtained using PRESTO's 
# 'prepdata' module. 
#
# Usage: python gwb_vs_splt_presto_timeseries.py
#       Fill in all input parameters.    
#
# Last Update: 29th Aug 2025; Raghav 
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import warnings
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')

# Basic Info
print('='*100)
print("Kindly enter correct paths (absolute or relative) below.\n")
spotlight_pc = input("Enter SPOTLIGHT's dedisperesed dat file path: ") 
gwb_pc = input("Enter GWB's dedisperesed dat file path: ")
spotlight_mjd = input("Enter SPOTLIGHT beam data start time in MJD (from fil header): ")
gwb_mjd = input("Enter GWB beam data start time in MJD (from ahdr header) : ")
src_name = input("Enter pulsar name: ")
band = input("Enter freq band of observation: ")

# Normalized cropped Plot
def crop_normal_plot(splt_crop_ts, gwb_crop_ts):
    splt_normal_crop_ts = normalize_ts(splt_crop_ts)
    gwb_normal_crop_ts = normalize_ts(gwb_crop_ts)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 6))
    ax.plot(spotlight_time,splt_normal_crop_ts, color="blue", label="PC beam from SPOTLIGHT")
    ax.plot(gwb_crop_time, gwb_normal_crop_ts, color="green", label="PC beam from GWB")
   
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xlabel("Unix Time (s)")
    ax.set_title(f"{src_name}, Band {band}")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.legend()
    plt.tight_layout()
    plt.show()

def normalize_ts(ts):
    return (ts - min(ts)) / (max(ts) - min(ts))

def find_time_lag_gwb(ts1, ts2, dt1=1.31072e-3, dt2=0.16384e-3):
    """
    Cross-correlate two time series with different sampling intervals
    and find the lag (in seconds).

    ts1, ts2: 1D arrays
    dt1, dt2: sampling intervals (seconds) of ts1 and ts2

    Returns:
        lag_sec: time lag (positive means ts2 lags behind ts1)
        corr: full cross-correlation array
        lags: array of lags (in seconds)
    """
    t1 = np.arange(len(ts1)) * dt1 # Spotlight timeseries
    t2 = np.arange(len(ts2)) * dt2 # GWB timeseries

    # interpolate ts2 onto t1's grid
    f = interp1d(t2, ts2, kind='linear', bounds_error=False, fill_value=0.0)
    ts2_interp = f(t1)

    n = min(len(ts1), len(ts2_interp))
    ts1 = ts1[:n]
    ts2_interp = ts2_interp[:n]

    # normalize
    ts1 = (ts1 - np.median(ts1)) / np.std(ts1)
    ts2_interp = (ts2_interp - np.median(ts2_interp)) / np.std(ts2_interp)

    corr = np.correlate(ts1, ts2_interp, mode="full")
    lags = np.arange(-n + 1, n) * dt1  # lag axis in seconds

    lag_sec = lags[np.argmax(corr)]  # max positive correlation
    return lag_sec, corr, lags

# Original entire timeseries
spotlight_ts = np.fromfile(spotlight_pc, dtype=np.float32)
gwb_ts = np.fromfile(gwb_pc, dtype=np.float32)

print('='*100)
print(f"Source name: {src_name}\nBand: {band}")
print("Spotlight ts data shape:", spotlight_ts.shape)
print("GWB ts data shape:", gwb_ts.shape)

# Convert to Unix
spotlight_t0 = Time(spotlight_mjd, format='mjd').unix
gwb_t0 = Time(gwb_mjd, format='mjd').unix

print("\nSPOTLIGHT start MJD: ", spotlight_mjd)
print("GWB start MJD: ", gwb_mjd)
print("Start time offset (Spotlight - GWB): ", spotlight_t0 - gwb_t0, "seconds")

dt = 1.31072e-3  # 1.31072 ms sampling
gwb_dt = 0.16384e-3 # 0.16384 ms sampling
print(f"Assuming SPOTLIGHT sampling time is {dt} sec, while for GWB is {gwb_dt} sec")
print('='*100)

spotlight_time = spotlight_t0 + np.arange(len(spotlight_ts)) * dt
gwb_time = gwb_t0  + np.arange(len(gwb_ts)) * gwb_dt 

# Normalized timeseries plots for comparison
splt_normal_ts = normalize_ts(spotlight_ts)
gwb_normal_ts = normalize_ts(gwb_ts)

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 6))
ax.plot(spotlight_time,spotlight_ts, color="blue", label="PC beam from SPOTLIGHT")
ax.plot(gwb_time, gwb_normal_ts, color="green", label="PC beam from GWB")

ax.set_ylabel("Normalized Amplitude")
ax.set_xlabel("Unix Time (s)")
ax.set_title(f"{src_name}, Band {band}")
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.legend()
plt.tight_layout()
plt.show()

# Cropping - start time offset
window_start_gwb = spotlight_time[0]; window_end_gwb = spotlight_time[-1]

gwb_start_indx_gwb = np.searchsorted(gwb_time, window_start_gwb, side="left")
gwb_end_indx_gwb   = np.searchsorted(gwb_time, window_end_gwb, side="right")
gwb_crop_ts = gwb_ts[gwb_start_indx_gwb:gwb_end_indx_gwb]
gwb_crop_time = gwb_time[gwb_start_indx_gwb:gwb_end_indx_gwb]

# Normalized cropped Plot
crop_normal_plot(spotlight_ts, gwb_crop_ts)

# Subplots of same lenght
fig1, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# FOR CROPPED TS ANALYSIS:
t_min = 1000 
t_max = 20000
t_min_gwb = int(t_min * dt / gwb_dt) 
t_max_gwb = int(t_max * dt / gwb_dt)

ax[0].plot(spotlight_time, spotlight_ts, color="blue", label="PC beam from SPOTLIGHT")
#ax[0].plot(spotlight_time[t_min:t_max], spotlight_ts[t_min:t_max], color="blue", label="PC beam from SPOTLIGHT") #UNCOMMENT FOR CROPPED TS ANALYSIS
ax[0].set_ylabel("Amplitude")
ax[0].set_title(f"{src_name}, Band {band}")
ax[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax[0].legend()

ax[1].plot(gwb_crop_time, gwb_crop_ts, color="green", label="PC beam from GWB")
#ax[1].plot(gwb_crop_time[t_min_gwb:t_max_gwb], gwb_crop_ts[t_min_gwb:t_max_gwb], color="green", label="PC beam from GWB") #UNCOMMENT FOR CROPPED TS ANALYSIS
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("Unix Time (s)")
ax[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax[1].legend()

fig1.suptitle("Burst ToA offset between PC beams")
fig1.tight_layout()

# calculation and plot cross-correlation:
#lag, corr, lags = find_time_lag_gwb(spotlight_ts[t_min:t_max], gwb_crop_ts[t_min_gwb:t_max_gwb]) #UNCOMMENT FOR CROPPED TS ANALYSIS
lag, corr, lags = find_time_lag_gwb(spotlight_ts, gwb_crop_ts)
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

msg = f" Estimated time lag: {lag*1000:.6f} msec "
line = "─" * len(msg)
print(f"\n{CYAN}┌{line}┐")
print(f"│{YELLOW}{msg}{CYAN}│")
print(f"└{line}┘{RESET}")

fig2 = plt.figure(figsize=(8, 4))
plt.plot(lags, corr)
plt.axvline(lag, color="red", linestyle="--", label=f"Lag = {lag:.4f} s")
plt.xlabel("Lag (s)")
plt.ylabel("Cross-correlation")
plt.suptitle("Cross-correlation between Spotlight and GWB data")
if lag >=0:
    plt.title("GWB timeseries is delayed relative to spotlight timeseries", fontsize=10)
else:
    plt.title("GWB timeseries leads Spotlight timeseries", fontsize=10)

plt.legend()
plt.show()
