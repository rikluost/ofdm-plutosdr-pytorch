import adi
import numpy as np
import torch
from scipy.signal import butter, filtfilt

class SDR:
    def __init__(self, SDR_TX_IP='ip:192.168.11.11', SDR_TX_FREQ=435E6, SDR_TX_GAIN=-80, SDR_RX_GAIN=0, SDR_TX_SAMPLERATE=1e6, SDR_TX_BANDWIDTH=1e6):
        """
        Initialize the SDR class with the specified parameters.

        Args:
            SDR_TX_IP (str): IP address of the TX SDR device.
            SDR_TX_FREQ (float): TX center frequency in Hz.
            SDR_TX_GAIN (float): TX gain in dB.
            SDR_RX_GAIN (float): RX gain in dB.
            SDR_TX_SAMPLERATE (float): TX sample rate (samples/second).
            SDR_TX_BANDWIDTH (float): TX bandwidth (Hz).
        """
        self.SDR_TX_IP = SDR_TX_IP
        self.SDR_TX_FREQ = int(SDR_TX_FREQ)
        self.SDR_RX_FREQ = int(SDR_TX_FREQ)
        self.SDR_TX_GAIN = int(SDR_TX_GAIN)
        self.SDR_RX_GAIN = int(SDR_RX_GAIN)
        self.SDR_TX_SAMPLERATE = int(SDR_TX_SAMPLERATE)
        self.SDR_TX_BANDWIDTH = int(SDR_TX_BANDWIDTH)
        self.num_samples = 0

    def SDR_TX_start(self):
        """
        Initialize and start the SDR transmitter with the specified configuration parameters.
        """
        self.sdr_tx = adi.ad9364(self.SDR_TX_IP)
        self.sdr_tx.tx_destroy_buffer()

        self.sdr_tx.tx_lo = self.SDR_TX_FREQ
        self.sdr_tx.rx_lo = self.SDR_TX_FREQ

        self.sdr_tx.gain_control_mode_chan0 = 'manual'
        self.sdr_tx.sample_rate = self.SDR_TX_SAMPLERATE
        self.sdr_tx.tx_rf_bandwidth = self.SDR_TX_BANDWIDTH
        self.sdr_tx.rx_rf_bandwidth = self.SDR_TX_BANDWIDTH
        self.sdr_tx.tx_hardwaregain_chan0 = self.SDR_TX_GAIN
        self.sdr_tx.rx_hardwaregain_chan0 = self.SDR_RX_GAIN

        self.sdr_tx.tx_destroy_buffer()

    def SDR_gain_set(self, tx_gain, rx_gain):
        """
        Set the TX and RX gain.

        Args:
            tx_gain (int): TX gain in dB.
            rx_gain (int): RX gain in dB.
        """
        self.sdr_tx.tx_hardwaregain_chan0 = tx_gain
        self.sdr_tx.rx_hardwaregain_chan0 = rx_gain

    def SDR_TX_send(self, SAMPLES, max_scale=1, cyclic=False):
        """
        Transmit the given signal samples through the SDR transmitter.

        Args:
            SAMPLES (Tensor or ndarray): The signal samples to be transmitted.
            max_scale (float): Scaling factor to adjust the amplitude of the signal.
            cyclic (bool): If True, the transmitted signal will be repeated in a cyclic manner.
        """
        self.sdr_tx.tx_destroy_buffer()

        if isinstance(SAMPLES, np.ndarray):
            self.num_samples = SAMPLES.size
        elif isinstance(SAMPLES, torch.Tensor):
            self.num_samples = SAMPLES.numel()
            SAMPLES = SAMPLES.numpy()

        samples = SAMPLES - np.mean(SAMPLES)
        samples = (samples / np.max(np.abs(samples))) * max_scale
        samples *= 2**14

        self.sdr_tx.tx_cyclic_buffer = cyclic
        self.sdr_tx.tx(samples)
    
    
    def SDR_TX_stop(self):
        """
        Stop the SDR transmitter.
        """
        self.sdr_tx.tx_destroy_buffer()
        self.sdr_tx.rx_destroy_buffer()
        
    def lowpass_filter(self, data):
        """
        Applies a lowpass filter to the input data.

        Parameters:
        data (numpy array): The input signal data to be filtered.

        Returns:
        numpy array: The filtered signal data.
        """
        # Define the Nyquist frequency
        nyq = 0.5 * self.SDR_TX_SAMPLERATE

        # Normalize the cutoff frequency. 
        normal_cutoff = 100 * 15000 * 1 / nyq

        # Design a 5th-order Butterworth lowpass filter.
        b, a = butter(5, normal_cutoff, btype='low', analog=False)

        # Apply the filter to the data using zero-phase filtering.
        y = filtfilt(b, a, data)

        return np.array(y)


    def SDR_RX_receive(self, n_SAMPLES=None, normalize=True):
        """
        Receive signal samples from the SDR receiver.

        Args:
            n_SAMPLES (int, optional): The number of samples to receive.
            normalize (bool): If True, normalizes the received signal to a maximum amplitude of 1.

        Returns:
            Tensor: The received signal samples as a PyTorch tensor.
        """
        if n_SAMPLES is None:
            n_SAMPLES = self.num_samples * 4
        if n_SAMPLES <= 0:
            n_SAMPLES = 1

        self.sdr_tx.rx_destroy_buffer()
        self.sdr_tx.rx_buffer_size = n_SAMPLES

        a = self.sdr_tx.rx()
        if normalize:
            a = a / np.max(np.abs(a))
            
        a = self.lowpass_filter(a)

        return torch.tensor(a, dtype=torch.complex64)
    

    
