
import adi
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

class SDR:
    def __init__(self, SDR_TX_IP, SDR_TX_FREQ, SDR_TX_GAIN=-80, SDR_RX_GAIN=0, SDR_TX_SAMPLERATE=1e6, SDR_TX_BANDWIDTH=1e6):
    
        self.SDR_TX_IP = SDR_TX_IP # IP address of the TX SDR device
        self.SDR_TX_FREQ = int(SDR_TX_FREQ) # TX center frequency in Hz
        self.SDR_RX_FREQ = int(SDR_TX_FREQ) # RX center frequency in Hz
        self.SDR_TX_GAIN = int(SDR_TX_GAIN) # TX gain in dB
        self.SDR_RX_GAIN = int(SDR_RX_GAIN) # RX gain in dB
        self.SDR_TX_SAMPLERATE = int(SDR_TX_SAMPLERATE) # TX sample rate (samples/second)
        self.SDR_TX_BANDWIDTH = int(SDR_TX_BANDWIDTH) # TX bandwidth (Hz)
        self.num_samples=0
        
    def SDR_TX_start(self):
        '''
        Initialize and start the SDR transmitter.

        This method sets up the SDR with the specified configuration parameters. It includes:
        - Initializing the SDR device using the ad9364 driver and the provided IP address.
        - Destroying any existing TX buffers to prepare for new transmissions.
        - Setting the local oscillator (LO) frequencies for both TX and RX.
        - Configuring the gain control mode for the TX channel to manual.
        - Setting the sample rate, RF bandwidth for both TX and RX, and hardware gains.

        The method ensures that the SDR transmitter is ready to send data with the configured settings.
        '''
        # Initialize the SDR device using the Analog Devices ad9364 driver
        self.sdr_tx = adi.ad9364(self.SDR_TX_IP)

        # Clear any existing transmit buffers
        self.sdr_tx.tx_destroy_buffer()

        # Set the local oscillator (LO) frequencies for transmit and receive
        self.sdr_tx.tx_lo = self.SDR_TX_FREQ  # Transmit frequency
        self.sdr_tx.rx_lo = self.SDR_TX_FREQ  # Receive frequency (set to the same as TX)

        # Set gain control mode to manual for the TX channel
        self.sdr_tx.gain_control_mode_chan0 = 'manual'

        # Configure the sample rate for both TX and RX
        self.sdr_tx.sample_rate = self.SDR_TX_SAMPLERATE

        # Set the RF bandwidth for both TX and RX
        self.sdr_tx.tx_rf_bandwidth = self.SDR_TX_BANDWIDTH
        self.sdr_tx.rx_rf_bandwidth = self.SDR_TX_BANDWIDTH

        # Set the hardware gain for both TX and RX
        self.sdr_tx.tx_hardwaregain_chan0 = self.SDR_TX_GAIN  # TX gain
        self.sdr_tx.rx_hardwaregain_chan0 = self.SDR_RX_GAIN  # RX gain

        # Clear transmit buffer again to ensure a clean start
        self.sdr_tx.tx_destroy_buffer()

    def SDR_gain_set(self, tx_gain, rx_gain):
        '''
        Set the TX and RX gain
        '''
        self.sdr_tx.tx_hardwaregain_chan0 = tx_gain
        self.sdr_tx.rx_hardwaregain_chan0 = rx_gain

    def SDR_TX_send(self, SAMPLES, max_scale=1, cyclic=False):
        '''
        Transmit the given signal samples through the SDR transmitter.

        Parameters:
        SAMPLES (Tensor or ndarray): The signal samples to be transmitted. 
                                     Can be a NumPy array or a PyTorch tensor.
        max_scale (float): Scaling factor to adjust the amplitude of the signal.
        cyclic (bool): If True, the transmitted signal will be repeated in a cyclic manner.

        This method prepares the samples for transmission by performing several steps:
        - Converts PyTorch tensors to NumPy arrays, if necessary.
        - Normalizes the signal to remove any DC offset.
        - Scales the signal's amplitude.
        - Adjusts the sample values to match the expected range of the SDR hardware.
        - Configures the SDR to either transmit the signal cyclically or just once.
        - Finally, transmits the prepared samples.
        '''

        self.sdr_tx.tx_destroy_buffer()  # Clear any existing buffer

        # Determine the number of samples based on whether SAMPLES is a NumPy array or a PyTorch tensor
        if isinstance(SAMPLES, np.ndarray):
            self.num_samples = SAMPLES.size
        elif isinstance(SAMPLES, torch.Tensor):
            self.num_samples = SAMPLES.numel()
            SAMPLES = SAMPLES.numpy()  # Convert to NumPy array if it's a PyTorch tensor

        # Normalize the signal to remove DC offset
        samples = SAMPLES - np.mean(SAMPLES)

        # Scale the samples to their maximum amplitude and adjust according to max_scale
        samples = (samples / np.max(np.abs(samples))) * max_scale

        # Scale the signal to the dynamic range expected by the SDR hardware
        samples *= 2**14  # PlutoSDR, for example, expects sample values in the range -2^14 to +2^14

        # Set cyclic buffer mode if required
        self.sdr_tx.tx_cyclic_buffer = cyclic

        # Transmit the prepared samples
        self.sdr_tx.tx(samples)
    
    def SDR_TX_stop(self):
        '''
        Stop the SDR TX
        '''
        self.sdr_tx.tx_destroy_buffer() 
        self.sdr_tx.rx_destroy_buffer()

    def SDR_RX_receive(self, n_SAMPLES=None, normalize=True):
        '''
        Receive signal samples from the SDR receiver.

        Parameters:
        n_SAMPLES (int, optional): The number of samples to receive. 
                                   If not specified, defaults to four times the number of transmitted samples.
        normalize (bool): If True, normalizes the received signal to a maximum amplitude of 1.

        This method handles the reception of signal samples from the SDR hardware by:
        - Determining the number of samples to receive.
        - Clearing any existing buffer in the receiver to prepare for new data.
        - Setting the buffer size to the specified number of samples.
        - Receiving the samples from the SDR hardware.
        - Optionally normalizing the amplitude of the received signal.
        - Converting the received samples to a PyTorch tensor for further processing.

        Returns:
        Tensor: The received signal samples as a PyTorch tensor.
        '''
        # Determine the number of samples to receive
        if n_SAMPLES is None:
            n_SAMPLES = self.num_samples * 4  # Default to four times the number of transmitted samples
        if n_SAMPLES <= 0:
            n_SAMPLES = 1  # Ensure at least one sample is received

        # Clear the receiver buffer to prepare for new data
        self.sdr_tx.rx_destroy_buffer()

        # Set the buffer size for receiving data
        self.sdr_tx.rx_buffer_size = n_SAMPLES

        # Receive the samples from the SDR hardware
        a = self.sdr_tx.rx()

        # Normalize the signal amplitude if required
        if normalize:
            a = a / np.max(np.abs(a))

        # Convert the received samples to a PyTorch tensor
        return torch.tensor(a, dtype=torch.complex64)
    
    def find_good_max_TX_gain_value(self, SDR_RX_GAIN=-30, Max_SINR=20, make_plot=True):
        '''
        Find the maximum transmit gain value that does not exceed a specified SINR.

        Parameters:
        SDR_RX_GAIN (int): The receive gain setting for the SDR.
        Max_SINR (float): The maximum desired SINR (Signal-to-Noise Ratio).

        This method operates by:
        - Generating a QPSK signal to be used as a test transmission.
        - Measuring the noise power of the received signal at a low TX gain.
        - Iteratively increasing the TX gain and measuring the SINR until the desired SINR is reached.
        - Plotting SINR versus TX gain for analysis.

        Returns:
        int: The transmit gain value that achieves the desired SINR without exceeding it.
        '''
        # Generate a QPSK signal
        qpsk_symbols = np.tile(np.array([0.7+0.7j, -0.7+0.7j, -0.7-0.7j, 0.7-0.7j]), 10)
        qpsk_symbols = np.fft.ifft(np.fft.ifftshift(qpsk_symbols))

        # Initial setup: stop TX, set gains, and measure noise power
        self.SDR_TX_stop()
        self.SDR_gain_set(tx_gain=-80, rx_gain=SDR_RX_GAIN)
        time.sleep(0.3)  # Wait for settings to take effect
        noise_sample = self.SDR_RX_receive(10000, normalize=False)
        noise_power = torch.mean(torch.abs(torch.tensor(noise_sample, dtype=torch.complex64))**2)

        # Prepare to record SINR and TX gain values
        SINR = []
        TX_GAIN_t = []

        # Iteratively test different TX gain values
        for SDR_TX_GAIN in range(-70, 0):
            # Transmit the QPSK signal and set the current TX gain
            self.SDR_TX_send(qpsk_symbols, max_scale=1, cyclic=True)
            self.SDR_gain_set(tx_gain=SDR_TX_GAIN, rx_gain=SDR_RX_GAIN)
            TX_GAIN_t.append(SDR_TX_GAIN)

            # Measure the signal power and calculate SINR
            sample = self.SDR_RX_receive(10000, normalize=False)
            signal_power = torch.mean(torch.abs(torch.tensor(sample, dtype=torch.complex64))**2)
            SINR.append(10 * torch.log10(signal_power / noise_power).item())

            # Stop if the maximum desired SINR is reached
            if SINR[-1] > Max_SINR:
                self.SDR_TX_stop()
                self.SDR_gain_set(tx_gain=-80, rx_gain=SDR_RX_GAIN)
                break

        # Output and plot the results
        self.SDR_gain_set(tx_gain=-80, rx_gain=SDR_RX_GAIN)
        self.SDR_TX_stop()
        if make_plot:
            plt.figure(figsize=(10, 4))
            plt.scatter(TX_GAIN_t, SINR)
            plt.title(f'SINR vs TX Gain. RX Gain set to {SDR_RX_GAIN}\nFreq={self.SDR_TX_FREQ/1e6} MHz, BW={self.SDR_TX_BANDWIDTH/1e6} MHz, SR={self.SDR_TX_SAMPLERATE/1e6} MHz')
            plt.grid()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('PlutoSDR TX Gain')
            plt.ylabel('Measured SINR')
            plt.tight_layout()
            plt.show()
            plt.savefig('pics/SINR_vs_TX_Gain.png')

        return SDR_TX_GAIN

        
    def show_params(self):
        print(f'RX LO: {self.sdr_tx.rx_lo}')
        print(f'TX LO: {self.sdr_tx.tx_lo}')
        print(f'RX RF Bandwidth: {self.sdr_tx.rx_rf_bandwidth}')
        print(f'TX RF Bandwidth: {self.sdr_tx.tx_rf_bandwidth}')
        print(f'Sample Rate: {self.sdr_tx.sample_rate}')
        print(f'RX Buffer Size: {self.sdr_tx.rx_buffer_size}')
        print(f'Gain Control Mode: {self.sdr_tx.gain_control_mode_chan0}')
        print(f'RX Gain: {self.sdr_tx.rx_hardwaregain_chan0}')
        print(f'TX Gain: {self.sdr_tx.tx_hardwaregain_chan0}')
        print(f'Filter: {self.sdr_tx.filter}')
        print(f'TX Cyclic Buffer: {self.sdr_tx.tx_cyclic_buffer}')

