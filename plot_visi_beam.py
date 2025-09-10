import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from datetime import datetime

def read_beam_file(filepath, max_samples=10000):
    """
    Read the binary beam file with the specified format.
    
    Args:
        filepath (str): Path to the 30ant.dat file
        max_samples (int): Maximum number of samples to read
    
    Returns:
        dict: Dictionary containing header info, timestamps, and beam data
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        # Read header (44 bytes total)
        # 5 doubles (8 bytes each) + 1 integer (4 bytes) = 44 bytes
        header_data = f.read(44)
        
        if len(header_data) != 44:
            raise ValueError("Invalid header size")
        
        # Unpack header: 5 doubles + 1 integer (little-endian)
        header = struct.unpack('<5d1i', header_data)
        
        freq_start = header[0]
        freq_end = header[1]
        channel_width = header[2]
        integration_time = header[3]
        processing_timestamp = header[4]
        channels = header[5]
        
        print(f"Header Information:")
        print(f"  Frequency Start: {freq_start} MHz")
        print(f"  Frequency End: {freq_end} MHz")
        print(f"  Channel Width: {channel_width} MHz")
        print(f"  Integration Time: {integration_time} ms")
        print(f"  Processing Timestamp: {processing_timestamp}")
        print(f"  Channels: {channels}")
        print(f"  Processing Time: {datetime.fromtimestamp(processing_timestamp)}")
        
        # Verify channels matches expected 4096
        if channels != 4096:
            print(f"Warning: Expected 4096 channels, got {channels}")
        
        # Calculate record size: 8 bytes (timestamp) + 4096 * 4 bytes (beam data)
        record_size = 8 + channels * 4
        
        # Read data records
        timestamps = []
        beam_data = []
        
        sample_count = 0
        while sample_count < max_samples:
            # Read one record
            record_data = f.read(record_size)
            
            if len(record_data) != record_size:
                print(f"End of file reached after {sample_count} samples")
                break
            
            # Unpack timestamp (double, 8 bytes)
            timestamp = struct.unpack('<d', record_data[:8])[0]
            
            # Unpack beam power data (4096 floats, 4 bytes each)
            beam_powers = struct.unpack(f'<{channels}f', record_data[8:])
            
            timestamps.append(timestamp)
            beam_data.append(beam_powers)
            sample_count += 1
            
            if sample_count % 1000 == 0:
                print(f"Read {sample_count} samples...")
    
    print(f"Successfully read {len(timestamps)} samples")
    
    return {
        'header': {
            'freq_start': freq_start,
            'freq_end': freq_end,
            'channel_width': channel_width,
            'integration_time': integration_time,
            'processing_timestamp': processing_timestamp,
            'channels': channels
        },
        'timestamps': np.array(timestamps),
        'beam_data': np.array(beam_data)
    }

def plot_beam_data(data, plot_type='waterfall'):
    """
    Plot the beam data in various formats.
    
    Args:
        data (dict): Data dictionary from read_beam_file
        plot_type (str): Type of plot - 'waterfall', 'spectrum', 'timeseries', or 'all'
    """
    
    timestamps = data['timestamps']
    beam_data = data['beam_data']
    header = data['header']
    
    # Convert timestamps to relative time in seconds
    time_rel = (timestamps - timestamps[0]) * 1000  # Convert to milliseconds
    
    # Create frequency array
    frequencies = np.linspace(header['freq_start'], header['freq_end'], header['channels'])
    
    if plot_type == 'waterfall' or plot_type == 'all':
        plt.figure(figsize=(12, 8))
        
        # Waterfall plot (time vs frequency)
        plt.subplot(2, 2, 1)
        plt.imshow(beam_data.T, aspect='auto', origin='lower', 
                   extent=[time_rel[0], time_rel[-1], frequencies[0], frequencies[-1]],
                   cmap='viridis')
        plt.colorbar(label='Beam Power')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (MHz)')
        plt.title('Beam Power Waterfall Plot')
        
        # Average spectrum
        plt.subplot(2, 2, 2)
        avg_spectrum = np.mean(beam_data, axis=0)
        plt.plot(frequencies, avg_spectrum)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Average Beam Power')
        plt.title('Average Spectrum')
        plt.grid(True)
        
        # Time series (total power)
        plt.subplot(2, 2, 3)
        total_power = np.sum(beam_data, axis=1)
        plt.plot(time_rel, total_power)
        plt.xlabel('Time (ms)')
        plt.ylabel('Total Power')
        plt.title('Total Power vs Time')
        plt.grid(True)
        
        # Sample spectrum at different times
        plt.subplot(2, 2, 4)
        n_samples = len(timestamps)
        for i in [0, n_samples//4, n_samples//2, 3*n_samples//4, -1]:
            plt.plot(frequencies, beam_data[i], alpha=0.7, 
                    label=f't={time_rel[i]:.1f}ms')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Beam Power')
        plt.title('Sample Spectra at Different Times')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('beam_data_analysis.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'beam_data_analysis.png'")
        plt.show()
    
    elif plot_type == 'spectrum':
        plt.figure(figsize=(10, 6))
        avg_spectrum = np.mean(beam_data, axis=0)
        plt.plot(frequencies, avg_spectrum)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Average Beam Power')
        plt.title('Average Beam Spectrum')
        plt.grid(True)
        plt.savefig('beam_spectrum.png', dpi=300, bbox_inches='tight')
        print("Spectrum plot saved as 'beam_spectrum.png'")
        plt.show()
        
    elif plot_type == 'timeseries':
        plt.figure(figsize=(10, 6))
        total_power = np.sum(beam_data, axis=1)
        plt.plot(time_rel, total_power)
        plt.xlabel('Time (ms)')
        plt.ylabel('Total Power')
        plt.title('Total Beam Power vs Time')
        plt.grid(True)
        plt.savefig('beam_timeseries.png', dpi=300, bbox_inches='tight')
        print("Time series plot saved as 'beam_timeseries.png'")
        plt.show()

# Main execution
if __name__ == "__main__":
    # Set the file path
    filepath = "/lustre_archive/spotlight/raghav/visibility_vs_beam/TST2_1Aug2025/30ant.dat"
    
    try:
        # Read the beam data
        print("Reading beam file...")
        data = read_beam_file(filepath, max_samples=10000)
        
        print(f"\nData shape: {data['beam_data'].shape}")
        print(f"Time range: {data['timestamps'][0]:.6f} to {data['timestamps'][-1]:.6f}")
        print(f"Duration: {(data['timestamps'][-1] - data['timestamps'][0]) * 1000:.2f} ms")
        
        # Plot the data
        print("\nGenerating plots...")
        plot_beam_data(data, plot_type='all')
        
        # Optional: Save data for further analysis
        # np.savez('beam_data.npz', 
        #          timestamps=data['timestamps'],
        #          beam_data=data['beam_data'],
        #          header=data['header'])
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the file path and ensure the file exists.")
