######################################################################
# Python script to find burst ToA offset primarily between SPOTLIGHT's
# PC beam and Visibility converted PC beam. The script also has 
# provision to compare the timeseries with GWB PC beam. The code expects
# dedispersed timeseries data in the form of binary dat file, obtained
# using PRESTO's 'prepdata' module. 
#
# Usage: python presto_timeseries.py
#       Fill in all input parameters. One can skip plotting GWB data.    
#
# Late Update: 21st Aug 2025; Raghav 
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import warnings
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')

# Basic Info
print('='*100)
print("Kindly enter correct paths (absolute or relative) below. If you don't want to plot GWB data for comparison, press ENTER in respective fields\n")
spotlight_pc = input("Enter SPOTLIGHT's dedisperesed dat file path: ") 
visi_pc = input("Enter Visibility's dedisperesed dat file path: ")
gwb_pc = input("Enter GWB's dedisperesed dat file path: ")
spotlight_mjd = input("Enter SPOTLIGHT beam data start time in MJD (from fil header): ")
gwb_mjd = input("Enter GWB beam data start time in MJD (from ahdr header) : ")
visi_unix = input("Enter visibility data start time in Unix Time (from Visi-R.raw.ts file): ")
visi_mjd = Time(visi_unix, format = 'unix').mjd 
src_name = input("Enter pulsar name: ")
band = input("Enter freq band of observation: ")

# Normalized cropped Plot
def crop_normal_plot(splt_crop_ts, visi_ts, gwb_crop_ts=None):
    splt_normal_crop_ts = normalize_ts(splt_crop_ts)
    visi_normal_ts = normalize_ts(visi_ts)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 6))
    ax.plot(splt_crop_time,splt_normal_crop_ts, color="blue", label="PC beam from SPOTLIGHT")
    ax.plot(visi_time, visi_normal_ts, color="red", label="PC beam from visibility")
    if gwb_crop_ts is not None:
        gwb_normal_crop_ts = normalize_ts(gwb_crop_ts)
        ax.plot(gwb_crop_time, gwb_normal_crop_ts, color="green", label="PC beam from GWB")
    else:
        print("Skipping GWB analysis")
   
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xlabel("Unix Time (s)")
    ax.set_title(f"{src_name}, Band {band}")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.legend()
    plt.tight_layout()
    plt.show()

def normalize_ts(ts):
    return (ts - min(ts)) / (max(ts) - min(ts))

def find_time_lag(ts1, ts2, dt=1.31072e-3):
    """
    Cross-correlate two normalized time series and find lag (in seconds).
    
    ts1, ts2: 1D arrays (same length or will be truncated)
    dt: sampling interval (seconds)
    
    Returns:
        lag_sec: time lag (positive means ts2 lags behind ts1)
        corr: full cross-correlation array
        lags: array of lags (in seconds)
    """
    n = min(len(ts1), len(ts2))
    ts1 = ts1[:n]
    ts2 = ts2[:n]
    
    ts1 = (ts1 - np.median(ts1)) / np.std(ts1)
    ts2 = (ts2 - np.median(ts2)) / np.std(ts2)

    corr = np.correlate(ts1, ts2, mode="full")
    lags = np.arange(-n + 1, n) * dt

    #lag_sec = lags[np.argmax(corr)]
    lag_sec = lags[np.argmax(np.abs(corr))]
    return lag_sec, corr, lags

# Original entire timeseries
spotlight_ts = np.fromfile(spotlight_pc, dtype=np.float32)
visi_ts = np.fromfile(visi_pc, dtype=np.float32)
if gwb_pc.strip() != "":
    gwb_ts = np.fromfile(gwb_pc, dtype=np.float32)

print('='*100)
print(f"Source name: {src_name}\nBand: {band}")
print("Spotlight ts data shape:", spotlight_ts.shape)
print("Visibility ts data shape:", visi_ts.shape)
if gwb_pc.strip() != "":
    print("GWB ts data shape:", gwb_ts.shape)

# Convert to Unix
spotlight_t0 = Time(spotlight_mjd, format='mjd').unix
visi_t0 = Time(visi_mjd, format='mjd').unix
if gwb_pc.strip() != "":
    gwb_t0 = Time(gwb_mjd, format='mjd').unix

print("\nSPOTLIGHT start MJD: ", spotlight_mjd)
print("Visibility start MJD: ", visi_mjd)
if gwb_pc.strip() != "":
    print("GWB start MJD: ", gwb_mjd)
    print("Start time offset (Spotlight - GWB): ", spotlight_t0 - gwb_t0, "seconds")
print("Start time offset (Spotlight - Visibility): ", spotlight_t0 - visi_t0, "seconds\n")

dt = 1.31072e-3  # 1.31072 ms sampling
gwb_dt = 0.16384e-3 # 0.16384 ms sampling
print(f"Assuming SPOTLIGHT and Visibility sampling time is {dt} sec, while for GWB is {gwb_dt} sec")
print('='*100)

spotlight_time = spotlight_t0 + np.arange(len(spotlight_ts)) * dt
visi_time = visi_t0 + np.arange(len(visi_ts)) * dt
if gwb_pc.strip() != "":
    gwb_time = gwb_t0  + np.arange(len(gwb_ts)) * gwb_dt 

# Normalized timeseries plots for comparison
splt_normal_ts = normalize_ts(spotlight_ts)
visi_normal_ts = normalize_ts(visi_ts)
if gwb_pc.strip() != "":
    gwb_normal_ts = normalize_ts(gwb_ts)

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 6))
ax.plot(spotlight_time,spotlight_ts, color="blue", label="PC beam from SPOTLIGHT")
ax.plot(visi_time, visi_ts, color="red", label="PC beam from visibility")
if gwb_pc.strip() != "":
    ax.plot(gwb_time, gwb_normal_ts, color="green", label="PC beam from GWB")

ax.set_ylabel("Normalized Amplitude")
ax.set_xlabel("Unix Time (s)")
ax.set_title(f"{src_name}, Band {band}")
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.legend()
plt.tight_layout()
plt.show()

# Cropping - start time offset
window_start = visi_time[0]; window_end = visi_time[-1]
window_start_gwb = spotlight_time[0]; window_end_gwb = spotlight_time[-1]

splt_start_indx = np.searchsorted(spotlight_time, window_start, side="left")
splt_end_indx   = np.searchsorted(spotlight_time, window_end, side="right")
splt_crop_ts = spotlight_ts[splt_start_indx:splt_end_indx]
splt_crop_time = spotlight_time[splt_start_indx:splt_end_indx]

if gwb_pc.strip() != "":
    gwb_start_indx_gwb = np.searchsorted(gwb_time, window_start_gwb, side="left")
    gwb_end_indx_gwb   = np.searchsorted(gwb_time, window_end_gwb, side="right")
    gwb_crop_ts = gwb_ts[gwb_start_indx_gwb:gwb_end_indx_gwb]
    gwb_crop_time = gwb_time[gwb_start_indx_gwb:gwb_end_indx_gwb]

# Normalized cropped Plot
if gwb_pc.strip() != "":
    crop_normal_plot(spotlight_ts, visi_ts, gwb_crop_ts)
else:
    crop_normal_plot(splt_crop_ts, visi_ts)

# Subplots of same lenght
fig1, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# FOR CROPPED TS ANALYSIS:
t_min = 14000 
t_max = 17500
print("splt:" , len(spotlight_ts), len(spotlight_time))
print("gwb:" , len(gwb_crop_ts), len(gwb_crop_time))

#ax[0].plot(splt_crop_time[t_min:t_max], splt_crop_ts[t_min:t_max], color="blue", label="PC beam from SPOTLIGHT")  #UNCOMMENT FOR CROPPED TS ANALYSIS
ax[0].plot(splt_crop_time, splt_crop_ts, color="blue", label="PC beam from SPOTLIGHT")
ax[0].set_ylabel("Amplitude")
ax[0].set_title(f"{src_name}, Band {band}")
ax[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax[0].legend()

#ax[1].plot(visi_time[t_min:t_max], visi_ts[t_min:t_max], color="red", label="PC beam from visibility")  #UNCOMMENT FOR CROPPED TS ANALYSIS
ax[1].plot(visi_time, visi_ts, color="red", label="PC beam from visibility")
ax[1].set_ylabel("Amplitude")
ax[1].set_xlabel("Unix Time (s)")
ax[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax[1].legend()

fig1.suptitle("Burst ToA offset between PC beams")
fig1.tight_layout()

# calculation and plot cross-correlation:
#lag, corr, lags = find_time_lag(splt_crop_ts[t_min:t_max], visi_ts[t_min:t_max], dt) #UNCOMMENT FOR CROPPED TS ANALYSIS
lag, corr, lags = find_time_lag(splt_crop_ts, visi_ts, dt)
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
plt.suptitle("Cross-correlation between Spotlight and Visibility data")
if lag >=0:
    plt.title("Visi timeseries is delayed relative to spotlight timeseries", fontsize=10)
else:
    plt.title("Visi timeseries leads Spotlight timeseries", fontsize=10)

plt.legend()
plt.show()
