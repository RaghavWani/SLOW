#!/usr/bin/env python3
"""
Standalone Post-Correlation Beam Generator for GMRT SPOTLIGHT
Reads 16 raw visibility files and generates a coherent beam with bandpass correction.

Usage:
    python pcbeam_generator.py -f file1.raw file2.raw ... file16.raw 
                               --freq-start 550e6 --freq-end 750e6 
                               --channels 4096 --output beam.dat
 ~ ARPAN PAL
"""

import numpy as np
import struct
import os
import argparse
import time
from datetime import datetime
import sys

class PCBeamGenerator:
    def __init__(self, freq_start, freq_end, channels, integration_time=1.31072e-3, 
                 antmask=1073741823, do_bandpass=True, do_baseline=True):
        """
        Initialize PC Beam Generator
        
        Parameters:
        -----------
        freq_start : float
            Start frequency in Hz
        freq_end : float
            End frequency in Hz  
        channels : int
            Number of channels
        integration_time : float
            Integration time in seconds (default: 1.31072 ms)
        antmask : int
            Antenna mask (default: 30 antennas)
        do_bandpass : bool
            Apply bandpass correction
        do_baseline : bool
            Subtract baseline (off-source mean)
        """
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.channels = channels
        self.integration_time = integration_time
        self.antmask = antmask
        self.do_bandpass = do_bandpass
        self.do_baseline = do_baseline
        
        # Hardcode antenna configuration (from antsamp.hdr)
        self.antenna_names = [
            'C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C08', 'C09', 'C10',
            'C11', 'C12', 'C13', 'C14', 'E02', 'E03', 'E04', 'E05', 'E06', 'S01',
            'S02', 'S03', 'S04', 'S06', 'W01', 'W02', 'W03', 'W04', 'W05', 'W06',
            'C07', 'S05'
        ]
        
        # Antenna positions (bx, by, bz in meters)
        self.antenna_positions = np.array([
            [6.95, 687.88, -20.04],      # C00
            [13.25, 326.45, -40.35],     # C01
            [0.00, 0.00, 0.00],          # C02
            [-51.20, -372.71, 133.59],   # C03
            [-51.01, -565.96, 123.43],   # C04
            [79.12, 67.81, -246.59],     # C05
            [71.25, -31.43, -220.58],    # C06
            [130.80, 280.68, -400.33],   # C08
            [48.61, 41.95, -151.65],     # C09
            [191.35, -164.87, -587.49],  # C10
            [102.49, -603.25, -321.56],  # C11
            [209.28, 174.85, -635.54],   # C12
            [368.67, -639.50, -1117.92], # C13
            [207.37, -473.69, -628.63],  # C14
            [-348.18, 2814.55, 953.67],  # E02
            [-707.56, 4576.04, 1932.46], # E03
            [-1037.59, 7780.57, 2903.29], # E04
            [-1177.96, 10199.90, 3343.20], # E05
            [-1572.05, 12073.32, 4543.13], # E06
            [942.99, 633.96, -2805.93],  # S01
            [1452.91, -367.22, -4279.16], # S02
            [2184.63, 333.10, -6404.96], # S03
            [3072.95, 947.79, -8979.50], # S04
            [4592.83, -369.09, -13382.48], # S06
            [-201.35, -1591.95, 591.32], # W01
            [-482.34, -3099.44, 1419.39], # W02
            [-991.46, -5200.01, 2899.11], # W03
            [-1733.91, -7039.06, 5067.53], # W04
            [-2705.69, -8103.26, 7817.14], # W05
            [-3101.52, -11245.77, 8916.26], # W06
            [-3102.11, -11245.60, 8916.26], # C07
            [-3102.11, -11245.60, 8916.26]  # S05
        ])
        
        # Compute derived parameters
        self.channel_width = (freq_end - freq_start) / channels
        self.frequencies = np.linspace(freq_start, freq_end, channels, endpoint=False)
        
        # SPOTLIGHT specific parameters
        self.rec_per_slice = 50
        self.max_ants = 32
        self.max_baselines = self.max_ants * (self.max_ants + 1) // 2 * 2  # RR + LL
        self.slice_duration = self.rec_per_slice * integration_time
        
        # Generate baseline mapping (same as original code)
        self.baseline_map = self._generate_baseline_map()
        
        # Count enabled antennas
        enabled_ants = bin(antmask).count('1')
        
        print(f"Frequency range: {freq_start/1e6:.1f} - {freq_end/1e6:.1f} MHz")
        print(f"Channel width: {self.channel_width/1e3:.1f} kHz")
        print(f"Integration time: {integration_time*1e3:.3f} ms")
        print(f"Antenna mask: 0x{antmask:08X} ({enabled_ants} antennas enabled)")
        print(f"Bandpass correction: {do_bandpass}")
        print(f"Baseline subtraction: {do_baseline}")
        
    def _generate_baseline_map(self):
        """
        Generate baseline mapping exactly like original code:
        All RR baselines first, then all LL baselines
        Each baseline: (ant0, ant1, polarization, baseline_index)
        """
        baseline_map = []
        baseline_idx = 0
        
        # RR baselines (polarization 0)
        for ant0 in range(self.max_ants):
            for ant1 in range(ant0, self.max_ants):
                baseline_map.append((ant0, ant1, 0, baseline_idx))
                baseline_idx += 1
        
        # LL baselines (polarization 1) 
        for ant0 in range(self.max_ants):
            for ant1 in range(ant0, self.max_ants):
                baseline_map.append((ant0, ant1, 1, baseline_idx))
                baseline_idx += 1
                
        return baseline_map

    def half_to_float_array(self, half_array):
        """Convert array of half-precision floats to single precision"""
        # Simple numpy conversion - much faster than manual conversion
        return half_array.view(np.float16).astype(np.float32)

    def read_slice_timestamp(self, filepath, slice_idx):
        """
        Read timestamp from the beginning of a slice
        
        Returns:
        --------
        timestamp : float
            Unix timestamp in seconds (with microsecond precision)
        """
        timeval_size = 16  # sizeof(struct timeval) - 16 bytes as per GMRT diagram
        recl = self.max_baselines * self.channels * 4  # 4 bytes per complex vis (2 half-floats)
        
        # Offset to slice
        offset = slice_idx * (timeval_size + self.rec_per_slice * recl)
        
        try:
            with open(filepath, 'rb') as f:
                f.seek(offset)
                timeval_data = f.read(timeval_size)
                if len(timeval_data) < timeval_size:
                    return None
                
                # Read first 8 bytes as timestamp (little endian)
                sec, usec = struct.unpack('<II', timeval_data[:8])
                timestamp = sec + usec / 1e6
                return timestamp
        except (IOError, struct.error) as e:
            print(f"Error reading timestamp from {filepath}: {e}")
            return None

    def read_slice_data(self, filepath, slice_idx):
        """
        Read visibility data for one slice (50 records)
        
        Returns:
        --------
        vis_data : ndarray
            Shape: (rec_per_slice, max_baselines, channels, 2) - real, imag
        timestamp : float
            Timestamp of the slice
        """
        timeval_size = 16  # 16 bytes as per GMRT diagram
        recl = self.max_baselines * self.channels * 4  # 4 bytes per vis
        
        # Read timestamp
        timestamp = self.read_slice_timestamp(filepath, slice_idx)
        if timestamp is None:
            return None, None
            
        # Read visibility data
        offset = slice_idx * (timeval_size + self.rec_per_slice * recl) + timeval_size
        data_size = self.rec_per_slice * recl
        
        try:
            with open(filepath, 'rb') as f:
                f.seek(offset)
                raw_data = f.read(data_size)
                if len(raw_data) < data_size:
                    return None, None
                
                # Convert to numpy array of uint16 (half-floats)
                half_data = np.frombuffer(raw_data, dtype=np.uint16)
                
                # Reshape: (records, baselines, channels, 2)
                half_data = half_data.reshape(self.rec_per_slice, self.max_baselines, 
                                            self.channels, 2)
                
                # Convert to float32
                vis_data = self.half_to_float_array(half_data)
                
                return vis_data, timestamp
                
        except (IOError, struct.error) as e:
            print(f"Error reading data from {filepath}: {e}")
            return None, None

    def compute_bandpass_correction(self, vis_data):
        """
        Compute bandpass correction and baseline subtraction - VECTORIZED VERSION
        Uses all channels for calibration (no burst channel exclusion)
        
        Parameters:
        -----------
        vis_data : ndarray
            Visibility data shape: (records, baselines, channels, 2)
            
        Returns:
        --------
        bandpass : ndarray
            Bandpass correction per baseline per channel
        baseline_vis : ndarray
            Mean off-source visibility per baseline per channel
        """
        n_rec, n_base, n_chan, _ = vis_data.shape
        
        # Initialize outputs
        bandpass = np.ones((n_base, n_chan), dtype=np.float32)
        baseline_vis = np.zeros((n_base, n_chan, 2), dtype=np.float32)
        
        if not (self.do_bandpass or self.do_baseline):
            return bandpass, baseline_vis
        
        print(f"Computing calibration using all {n_chan} channels (vectorized)")
        
        # Replace NaN/Inf with zeros for stable computation
        vis_clean = np.nan_to_num(vis_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute baseline subtraction (mean over records)
        if self.do_baseline:
            baseline_vis = np.mean(vis_clean, axis=0)  # Shape: (baselines, channels, 2)
        
        # Compute bandpass correction (mean amplitude over records)
        if self.do_bandpass:
            # Compute amplitudes: sqrt(real² + imag²)
            amplitudes = np.sqrt(np.sum(vis_clean**2, axis=3))  # Shape: (records, baselines, channels)
            
            # Average over records
            bandpass = np.mean(amplitudes, axis=0)  # Shape: (baselines, channels)
            
            # Normalize per baseline (make median = 1.0)
            for b in range(n_base):
                positive_mask = bandpass[b] > 0
                if np.any(positive_mask):
                    bp_median = np.median(bandpass[b, positive_mask])
                    if bp_median > 0:
                        bandpass[b] /= bp_median
                    else:
                        bandpass[b] = 1.0
                else:
                    bandpass[b] = 1.0
        
        return bandpass, baseline_vis

    def apply_calibration(self, vis_data, bandpass, baseline_vis):
        """Apply bandpass and baseline corrections to visibility data"""
        if not (self.do_bandpass or self.do_baseline):
            return vis_data
        
        calibrated = vis_data.copy()
        
        for b in range(vis_data.shape[1]):  # baselines
            for c in range(vis_data.shape[2]):  # channels
                if self.do_baseline:
                    # Subtract baseline
                    calibrated[:, b, c, :] -= baseline_vis[b, c]
                
                if self.do_bandpass and bandpass[b, c] > 0:
                    # Apply bandpass correction
                    calibrated[:, b, c, :] /= bandpass[b, c]
        
        return calibrated

    def compute_beam(self, vis_data):
        """
        Compute post-correlation beam by coherently summing visibilities - VECTORIZED VERSION
        Apply antenna mask exactly like original code
        
        Parameters:
        -----------
        vis_data : ndarray
            Calibrated visibility data: (records, baselines, channels, 2)
            
        Returns:
        --------
        beam : ndarray
            Beam power per record per channel
        """
        n_rec, n_base, n_chan, _ = vis_data.shape
        
        print(f"Computing beam for {n_rec} records, {n_chan} channels...")
        
        # Create baseline mask based on antenna mask
        baseline_mask = np.zeros(n_base, dtype=bool)
        used_baselines = 0
        
        for baseline_idx, (ant0, ant1, pol, data_idx) in enumerate(self.baseline_map):
            if data_idx >= n_base:
                continue
                
            # Apply antenna mask - skip if either antenna not in mask
            if not ((1 << ant0) & self.antmask) or not ((1 << ant1) & self.antmask):
                continue
            
            # Skip autocorrelations (ant0 == ant1)
            if ant0 == ant1:
                continue
            
            baseline_mask[data_idx] = True
            used_baselines += 1
        
        print(f"Using {used_baselines} baselines for beam computation")
        
        if used_baselines == 0:
            print("Warning: No baselines selected!")
            return np.zeros((n_rec, n_chan), dtype=np.float32)
        
        # Vectorized beam computation
        # Select valid baselines: shape (records, valid_baselines, channels, 2)
        valid_vis = vis_data[:, baseline_mask, :, :]
        
        # Replace NaN/Inf with zeros
        valid_vis = np.nan_to_num(valid_vis, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Coherent sum over baselines (axis=1)
        # Result shape: (records, channels, 2)
        coherent_sum = np.sum(valid_vis, axis=1)
        
        # Compute beam power: sqrt(real² + imag²)
        # Result shape: (records, channels)
        beam = np.sqrt(np.sum(coherent_sum**2, axis=2))
        
        return beam

    def process_files(self, file_paths, output_path, max_slices=None):
        """
        Process all visibility files and generate beam
        
        CORRECT processing order for GMRT SPOTLIGHT:
        Slice 0: File 0, File 1, File 2, ..., File 15 (chronological)
        Slice 1: File 0, File 1, File 2, ..., File 15 (chronological)
        etc.
        
        Parameters:
        -----------
        file_paths : list
            List of paths to 16 visibility files
        output_path : str
            Output file path for beam data
        max_slices : int or None
            Maximum number of slices to process per file (for testing)
        """
        n_files = len(file_paths)
        if n_files != 16:
            print(f"Warning: Expected 16 files, got {n_files}")
        
        print(f"Processing {n_files} files in chronological order...")
        
        # Write header to output file
        self.write_header(output_path)
        
        all_timestamps = []
        all_beams = []
        
        # Get reference timestamp from first file, first slice only
        first_file_timestamp = self.read_slice_timestamp(file_paths[0], 0)
        if first_file_timestamp is None:
            print("Error: Could not read timestamp from first file")
            return np.array([]), np.array([])
        
        with open(output_path, 'ab') as out_file:
            slice_idx = 0
            
            # Process slices in chronological order
            while True:
                if max_slices and slice_idx >= max_slices:
                    break
                
                print(f"\nProcessing slice {slice_idx} across all files...")
                
                files_with_data = 0
                slice_data = []
                
                # Read slice_idx from all files (chronological order)
                for file_idx, filepath in enumerate(file_paths):
                    vis_data, base_timestamp = self.read_slice_data(filepath, slice_idx)
                    
                    if vis_data is not None:
                        slice_data.append((file_idx, filepath, vis_data, base_timestamp))
                        files_with_data += 1
                    else:
                        print(f"  File {file_idx}: No data at slice {slice_idx}")
                
                if files_with_data == 0:
                    print(f"No more data at slice {slice_idx} - stopping")
                    break
                
                print(f"  Found data in {files_with_data} files")
                
                # Read timestamp from first file of this slice and validate
                if slice_data:
                    actual_slice_timestamp = slice_data[0][3]  # timestamp from first file
                    expected_slice_timestamp = first_file_timestamp + slice_idx * (n_files * self.rec_per_slice) * self.integration_time
                    
                    print(f"  Slice {slice_idx} - Actual: {actual_slice_timestamp:.6f}, Expected: {expected_slice_timestamp:.6f}")
                    
                    # Use expected timestamp (calculated) rather than actual
                    slice_base_timestamp = expected_slice_timestamp
                else:
                    slice_base_timestamp = first_file_timestamp
                
                # Process each file's slice data in chronological order
                for file_idx, filepath, vis_data, base_timestamp in slice_data:
                    # Calculate correct timestamp for this file within the slice
                    file_offset = file_idx * self.rec_per_slice * self.integration_time
                    corrected_base_timestamp = slice_base_timestamp + file_offset
                    
                    print(f"  Processing file {file_idx}: {os.path.basename(filepath)}, timestamp: {corrected_base_timestamp:.6f}")
                    
                    # Compute calibration for this slice (vectorized)
                    print(f"    Computing calibration...")
                    bandpass, baseline_vis = self.compute_bandpass_correction(vis_data)
                    
                    # Apply calibration (vectorized)
                    print(f"    Applying calibration...")
                    calibrated_vis = self.apply_calibration(vis_data, bandpass, baseline_vis)
                    
                    # Compute beam for each record in this slice (vectorized)
                    print(f"    Computing beam...")
                    beam = self.compute_beam(calibrated_vis)
                    
                    # Save beam data for each record
                    for rec_idx in range(self.rec_per_slice):
                        # Calculate exact timestamp for this record using corrected timestamp
                        rec_timestamp = corrected_base_timestamp + rec_idx * self.integration_time
                        
                        # Write timestamp and beam data
                        out_file.write(struct.pack('<d', rec_timestamp))  # 8 bytes
                        out_file.write(beam[rec_idx].astype('<f4').tobytes())  # channels * 4 bytes
                        
                        all_timestamps.append(rec_timestamp)
                        all_beams.append(beam[rec_idx])
                
                slice_idx += 1
        
        print(f"\nProcessed {len(all_timestamps)} records total")
        print(f"Output written to: {output_path}")
        
        # Data should already be in chronological order now
        timestamps_array = np.array(all_timestamps)
        beams_array = np.array(all_beams)
        
        # Verify chronological order
        if len(timestamps_array) > 1:
            time_diffs = np.diff(timestamps_array)
            if np.any(time_diffs < 0):
                print("Warning: Some timestamps are not in chronological order!")
            else:
                print("✓ All timestamps are in chronological order")
        
        return timestamps_array, beams_array

    def write_header(self, output_path):
        """Write header information to output file"""
        header = struct.pack('<ddddd',  # Little endian doubles (40 bytes)
                           self.freq_start,      # Start frequency
                           self.freq_end,        # End frequency  
                           self.channel_width,   # Channel width
                           self.integration_time, # Integration time
                           time.time())          # Processing timestamp
        header += struct.pack('<I', self.channels)  # Number of channels (4 bytes)
        # Total header size: 44 bytes
        
        with open(output_path, 'wb') as f:
            f.write(header)

def main():
    parser = argparse.ArgumentParser(description='Generate Post-Correlation Beam from GMRT SPOTLIGHT data')
    
    # Input files
    parser.add_argument('-f', '--files', nargs='+', required=True,
                       help='List of 16 raw visibility files')
    
    # Frequency settings
    parser.add_argument('--freq-start', type=float, required=True,
                       help='Start frequency in Hz (e.g., 550e6)')
    parser.add_argument('--freq-end', type=float, required=True,
                       help='End frequency in Hz (e.g., 750e6)')
    parser.add_argument('--channels', type=int, default=4096,
                       help='Number of frequency channels (default: 4096)')
    
    # Timing
    parser.add_argument('--integration-time', type=float, default=1.31072e-3,
                       help='Integration time in seconds (default: 1.31072e-3)')
    
    # Calibration options
    parser.add_argument('--no-bandpass', action='store_true',
                       help='Disable bandpass correction')
    parser.add_argument('--no-baseline', action='store_true', 
                       help='Disable baseline subtraction')
    
    # Output
    parser.add_argument('-o', '--output', default='pcbeam.dat',
                       help='Output file path (default: pcbeam.dat)')
    
    # Antenna mask
    parser.add_argument('--antmask', default=0x3fffffff,
                       help='Antenna mask (default: 1073741823 = 30 antennas)')
    
    # Testing
    parser.add_argument('--max-slices', type=int,
                       help='Maximum number of slices to process (for testing)')
    
    args = parser.parse_args()
    
    antmask_int = int(args.antmask, 16) #Hexadecimal to int
    # Validate inputs
    if len(args.files) != 16:
        print(f"Warning: Expected 16 files, got {len(args.files)}")
    
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
    
    # Create beam generator
    generator = PCBeamGenerator(
        freq_start=args.freq_start,
        freq_end=args.freq_end,
        channels=args.channels,
        integration_time=args.integration_time,
        antmask=antmask_int,
        do_bandpass=not args.no_bandpass,
        do_baseline=not args.no_baseline
    )
    
    # Process files
    try:
        timestamps, beams = generator.process_files(
            args.files, 
            args.output,
            max_slices=args.max_slices
        )
        
        print("\nProcessing complete!")
        print(f"Generated beam for {len(timestamps)} time samples")
        print(f"Time range: {timestamps[0]:.3f} - {timestamps[-1]:.3f} seconds")
        print(f"Output file: {args.output}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
