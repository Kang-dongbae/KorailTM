import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timezone
from scipy.stats import skew, kurtosis
from scipy.signal import stft
from dataclasses import dataclass
from functools import partial
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class BearingParameters:
    """Bearing parameters configuration"""
    RPM: float = 1800
    N: int = 8
    PD: float = 0.05
    BD: float = 0.01
    phi: float = 0.0

    def calculate_frequencies(self) -> Dict[str, float]:
        """Calculate bearing defect frequencies"""
        phi_rad = np.radians(self.phi)
        rpm_per_sec = self.RPM / 60
        bd_pd_ratio = self.BD / self.PD
        
        return {
            'BPFO': (self.N * rpm_per_sec / 2) * (1 - bd_pd_ratio * np.cos(phi_rad)),
            'BPFI': (self.N * rpm_per_sec / 2) * (1 + bd_pd_ratio * np.cos(phi_rad)),
            'BSF': (self.PD / self.BD) * rpm_per_sec * np.sqrt(1 - (bd_pd_ratio * np.cos(phi_rad)) ** 2),
            'FTF': (rpm_per_sec / 2) * (1 - bd_pd_ratio * np.cos(phi_rad))
        }

class SignalProcessor:
    """Signal processing and statistical analysis for bearing data"""
    
    def __init__(self):
        self.stats_functions = {
            # Built-in pandas aggregations as strings
            'mean': 'mean',
            'max': 'max',
            'min': 'min',
            'var': 'var',
            # Custom functions
            'rms': lambda x: np.sqrt(np.mean(np.square(x))),
            'sra': lambda x: np.square(np.mean(np.sqrt(np.abs(x)))),
            'aa': lambda x: np.mean(np.abs(x)),
            'ptp': lambda x: np.ptp(x),
            'skewness': partial(skew, bias=False),
            'kurtosis': partial(kurtosis, bias=False),
            'crest': lambda x: np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x))),
            'shape': lambda x: np.sqrt(np.mean(np.square(x))) / np.mean(np.abs(x)),
            'impulse': lambda x: np.max(np.abs(x)) / np.mean(np.abs(x)),
            'coefficient_of_variation': lambda x: np.max(np.abs(x)) / np.square(np.mean(np.sqrt(np.abs(x)))),
            'coefficient_of_skewness': lambda x: skew(x, bias=False) / np.var(x)**3,
            'coefficient_of_kurtosis': lambda x: kurtosis(x, bias=False) / np.var(x)**4
        }

    @staticmethod
    def parse_timestamp_to_utc(timestamps: pd.Series) -> pd.Series:
        """Convert millisecond timestamps to UTC datetime"""
        try:
            return pd.to_datetime(timestamps / 1000, unit='s', utc=True)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid timestamp format: {str(e)}")

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data with efficient filtering"""
        df = data.copy()
        df['time'] = self.parse_timestamp_to_utc(df['timestamp'])
        df['second'] = (df['timestamp'] / 1000).astype(np.int64)
        
        second_counts = df['second'].value_counts()
        valid_seconds = second_counts[second_counts == 1001].index
        return df[df['second'].isin(valid_seconds)]

    def extract_defect_stats(self, signal: np.ndarray, sampling_rate: float, 
                           target_freq: float, bandwidth: float = 1.0) -> Tuple[float, float]:
        """Extract defect statistics using FFT"""
        n = len(signal)
        fft_data = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(n, 1/sampling_rate)
        magnitude_spectrum = np.abs(fft_data)
        
        mask = (fft_freq >= target_freq - bandwidth) & (fft_freq <= target_freq + bandwidth)
        if not np.any(mask):
            return np.nan, np.nan
            
        filtered_mag = magnitude_spectrum[mask]
        idx = np.argmax(filtered_mag)
        
        fft_filtered = np.zeros(n, dtype=complex)
        fft_filtered[:n//2] = fft_data[:n//2]
        fft_filtered[n//2:] = np.conj(fft_filtered[:0:-1])
        signal_filtered = np.fft.ifft(fft_filtered).real
        
        return np.mean(signal_filtered), np.max(signal_filtered)

    def extract_stft_features(self, signal: np.ndarray, sampling_rate: float, 
                             target_freq: float, bandwidth: float = 1.0) -> Tuple[float, float]:
        """Extract STFT features for a target frequency band"""
        nperseg = min(256, len(signal))  # Window size for STFT
        f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg//2)
        
        # Find frequency indices within the target band
        mask = (f >= target_freq - bandwidth) & (f <= target_freq + bandwidth)
        if not np.any(mask):
            return np.nan, np.nan
        
        # Compute magnitude spectrum
        mag_spectrum = np.abs(Zxx[mask, :])
        
        # Return mean and max across time for the frequency band
        return np.mean(mag_spectrum), np.max(mag_spectrum)

    def compute_channel_stats(self, df: pd.DataFrame, 
                            channels: List[str] = ['ch1', 'ch2', 'ch3'], 
                            group_col: str = 'second') -> Dict[str, pd.DataFrame]:
        """Compute statistics, FFT, and STFT features for each channel"""
        results = {}
        grouped = df.groupby(group_col)
        
        # Built-in pandas aggregations (as strings)
        built_in_aggs = ['mean', 'max', 'min', 'var']
        # Custom aggregations (as callables)
        custom_aggs = {k: v for k, v in self.stats_functions.items() if k not in built_in_aggs}
        
        # Calculate bearing frequencies
        bearing_params = BearingParameters()
        defect_freqs = bearing_params.calculate_frequencies()
        
        # Calculate sampling rate once
        sampling_rate = 1 / np.diff(df['timestamp'].values / 1000).mean()
        
        for ch in channels:
            # Compute built-in and custom aggregations
            stats_df = grouped[ch].agg(built_in_aggs).reset_index()
            custom_stats = grouped[ch].agg(list(custom_aggs.values())).reset_index()
            custom_stats.columns = [group_col] + list(custom_aggs.keys())
            
            # Combine built-in and custom stats
            stats_df = pd.concat([stats_df, custom_stats.drop(columns=[group_col])], axis=1)
            
            # Add defect frequencies and FFT/STFT stats for first group
            first_group_signal = df[df[group_col] == df[group_col].iloc[0]][ch].values
            for freq_name, freq_value in defect_freqs.items():
                # FFT-based stats
                fft_mean, fft_max = self.extract_defect_stats(first_group_signal, sampling_rate, freq_value)
                stats_df[f'{freq_name}_Freq'] = freq_value
                stats_df[f'{freq_name}_FFT_Mean'] = fft_mean
                stats_df[f'{freq_name}_FFT_Max'] = fft_max
                
                # STFT-based stats
                stft_mean, stft_max = self.extract_stft_features(first_group_signal, sampling_rate, freq_value)
                stats_df[f'{freq_name}_STFT_Mean'] = stft_mean
                stats_df[f'{freq_name}_STFT_Max'] = stft_max
                
            results[ch] = stats_df
            
        return results

def main():
    """Main execution function"""
    try:
        # Load and process data
        processor = SignalProcessor()
        data = pd.read_csv(r"C:\Dev\TMBearing\testdata.csv")
        processed_data = processor.preprocess_data(data)
        results = processor.compute_channel_stats(processed_data)
        
        # Print results for ch1
        print(results['ch1'].columns)
        print(results['ch1'].head())
        
    except FileNotFoundError:
        logging.error("Input CSV file not found")
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()