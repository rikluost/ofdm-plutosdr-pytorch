import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy import interpolate
import torch


######################################################################

def mapping_table(Qm, plotMap=False):
    '''
    Mapping table for QAM modulation in PyTorch

    Parameters
    ----------
    Qm : int
        Modulation order
    plotMap : bool
        Plot the constellation

    Returns
    -------
    dict
        Mapping table d_out
        Demapping table de_out
    '''

    # Size of the constellation
    size = int(torch.sqrt(torch.tensor(2**Qm)))

    # The constellation
    a = torch.arange(size, dtype=torch.float32)

    # Shift the constellation to the center
    b = a - torch.mean(a)

    # Use broadcasting to create the complex constellation grid
    C = (b.unsqueeze(1) + 1j * b).flatten()

    # Normalize the constellation
    C /= torch.sqrt(torch.mean(torch.abs(C)**2))

    # Function to convert index to binary
    def index_to_binary(i, Qm):
        return tuple(map(int, '{:0{}b}'.format(int(i), Qm)))

    # Create the mapping dictionary
    mapping_out = {index_to_binary(i, Qm): val for i, val in enumerate(C)}

    # Create the demapping table
    demapping_out = {v: k for k, v in mapping_out.items()}

    # Plotting the constellation
    if plotMap:
        plt.figure(figsize=(4,4))
        plt.plot(C.real, C.imag, 'o')
        plt.grid()
        plt.xlabel('real')
        plt.ylabel('imaginary')
        plt.title(f'Constellation, modulation order {Qm}')
        plt.savefig('pics/QAMconstellation.png')
        plt.show()

    return mapping_out, demapping_out # mapping table, demapping table

######################################################################

def TTI_mask(S, F, Fp, Sp, FFT_offset, plotTTI=False):
    '''
    Create TTI resource elements mask in PyTorch

    Parameters
    ----------
    S : int
        Number of symbols
    F : int
        Number of subcarriers
    Fp : int
        Pilot subcarrier spacing
    Sp : int
        Pilot symbol
    FFT_offset : int
        FFT offset
    plotTTI : bool
        Plot TTI mask

    Returns
    -------
    tensor
        TTI mask

    Legend:
    0 = DRX
    1 = PDSCH
    2 = pilots
    5 = DC
    '''
    # Create a mask with all ones
    TTI_mask = torch.ones((S, F), dtype=torch.int8) # all ones

    # Set symbol Sp for pilots
    TTI_mask[Sp, torch.arange(0, F, Fp)] = 2 # for pilots TX1

    # DC
    TTI_mask[:, F // 2] = 5 # DC to non-allocable power (oscillator phase noise)

    # Add FFT offsets
    TTI_mask = torch.cat((torch.zeros(S, FFT_offset, dtype=torch.int8), TTI_mask, torch.zeros(S, FFT_offset, dtype=torch.int8)), dim=1)

    # Plotting the TTI mask
    if plotTTI:
        plt.figure(figsize=(8, 1.5))
        plt.imshow(TTI_mask.numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.title('TTI mask')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        plt.savefig('pics/TTImask.png')
        plt.show()

    return TTI_mask


######################################################################

def pilot_set(TTI_mask, power=1.0):
    ''' 
    Create simple pilot QPSK set in PyTorch

    Parameters
    ----------
    TTI_mask : Tensor
        TTI mask
    power : float
        Power of the pilot

    Returns
    -------
    list
        pilot set
    '''
    # Define QPSK pilot values
    pilot_values = [-0.7 - 0.7j, -0.7 + 0.7j, 0.7 - 0.7j, 0.7 + 0.7j]

    # Count the number of pilot elements in the TTI mask
    num_pilots = torch.count_nonzero(TTI_mask == 2).item()

    # Create a list of pilot values repeated to match the number of pilots
    pilots = list(itertools.repeat(pilot_values, num_pilots))

    # Flatten the list and apply the power scaling
    pilots = [p * power for sublist in pilots for p in sublist][:num_pilots]

    return pilots


######################################################################

def create_PDSCH_data(TTI_mask, Qm, mapping_table, power=1, count=None):
    '''
    Create PDSCH data and symbols in PyTorch

    Parameters
    ----------
    TTI_mask : Tensor
        TTI mask
    Qm : int
        Modulation order
    mapping_table : dict
        Mapping table
    power : float
        Power scaling for symbols
    count : int
        Number of symbols to generate

    Returns
    -------
    Tensor
        PDSCH data
    Tensor
        PDSCH symbols
    '''
    # Count the number of PDSCH elements if count is not provided
    if count is None:
        count = torch.count_nonzero(TTI_mask == 1).item()

    # Generate random bits
    bits = torch.randint(0, 2, (count * Qm,), dtype=torch.int32)

    # Reshape the bits to match the modulation order
    pdsch_bits = bits.view(-1, Qm)

    # Convert the bits to symbols using the mapping table
    pdsch_symbols = torch.tensor([mapping_table[tuple(row.tolist())] for row in pdsch_bits], dtype=torch.complex64)

    # Apply power scaling to the symbols
    pdsch_symbols = pdsch_symbols * power

    return pdsch_bits, pdsch_symbols



######################################################################

def RE_mapping(TTI_mask, pilot_set, pdsch_symbols, plotTTI=False): 
    '''
    OFDM TTI in frequency domain, resource elements mapping in PyTorch

    Parameters
    ----------
    TTI_mask : Tensor
        TTI mask
    pilot_set : list
        list of pilot symbols
    pdsch_symbols : list
        list of PDSCH symbols
    plotTTI : bool
        plot TTI mask

    Returns
    -------
    Tensor
        mapped TTI
    '''

    # Create a zero tensor for the overall F subcarriers * S symbols
    TTI = torch.zeros(TTI_mask.shape, dtype=torch.complex64)

    # Allocate the payload and pilot
    TTI[TTI_mask==1] = torch.tensor(pdsch_symbols, dtype=torch.complex64)
    TTI[TTI_mask==2] = torch.tensor(pilot_set, dtype=torch.complex64)

    # Plotting the TTI modulated symbols
    if plotTTI:
        plt.figure(figsize=(8, 1.5))
        plt.imshow(torch.abs(TTI).numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.title('TTI modulated symbols')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        plt.savefig('pics/TTImod.png')
        plt.show()

    return TTI


######################################################################

def FFT(TTI):
    '''
    OFDM TTI in time domain, FFT using PyTorch

    Parameters
    ----------
    TTI : Tensor
        TTI modulated symbols

    Returns
    -------
    Tensor
        TTI in time domain
    '''
    # Apply ifftshift and IFFT using PyTorch
    return torch.fft.ifft(torch.fft.ifftshift(TTI, dim=1))


######################################################################

def CP_addition(OFDM_data, S, FFT_size, CP):
    '''
    OFDM TTI in time domain with cyclic prefix for each symbol using PyTorch

    Parameters
    ----------
    OFDM_data : Tensor
        TTI in frequency domain
    S : int
        number of symbols
    FFT_size : int
        FFT size
    CP : int
        cyclic prefix length

    Returns 
    -------
    Tensor
        TTI in time domain with cyclic prefix
    '''
    if OFDM_data is None:
        OFDM_data = OFDM_data

    # Initialize output tensor
    out = torch.zeros((S, FFT_size + CP), dtype=torch.complex64)

    # Add cyclic prefix to each symbol
    for symbol in range(S):
        out[symbol, :] = torch.cat((OFDM_data[symbol, -CP:], OFDM_data[symbol, :]), dim=0)

    return out.flatten()


######################################################################

def SP(bits, length, Qm): # serial to parallel
    '''
    Serial to parallel conversion in PyTorch

    Parameters
    ----------
    bits : Tensor
        bits
    length : int
        length of the array
    Qm : int
        Modulation order

    Returns
    -------
    Tensor
        bits reshaped into parallel format
    '''
    return bits.view(length, Qm)


######################################################################

import torch

def PS(bits): # parallel to serial
    '''
    Parallel to serial conversion in PyTorch

    Parameters
    ----------
    bits : Tensor
        bits

    Returns
    -------
    Tensor
        bits flattened into a serial format
    '''
    return bits.view(-1)


######################################################################

def DFT(rxsignal, plotDFT=False):
    '''
    Calculate the DFT of the received signal using PyTorch

    Parameters
    ----------
    rxsignal : Tensor
        OFDM data
    plotDFT : bool
        plot the DFT

    Returns
    -------
    Tensor 
        DFT of the received signal
    '''
    # Calculate DFT
    OFDM_RX_DFT = torch.fft.fftshift(torch.fft.fft(rxsignal, dim=1), dim=1)

    # Plot the DFT if required
    if plotDFT:
        plt.figure(figsize=(8, 1.5))
        plt.imshow(torch.abs(OFDM_RX_DFT).numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Symbol')
        plt.savefig('pics/TTI_RX.png')
        plt.show()

    return OFDM_RX_DFT


######################################################################

import torch

def remove_fft_Offests(RX_NO_CP, F, FFT_offset):
    '''
    Remove the FFT offsets using PyTorch

    Parameters
    ----------
    RX_NO_CP : Tensor
        OFDM data without cyclic prefix
    F : int
        number of subcarriers
    FFT_offset : int
        FFT offset

    Returns 
    ------- 
    Tensor
        OFDM data without cyclic prefix and FFT offsets
    '''

    # Calculate indices for the remaining subcarriers after removing offsets
    remaining_indices = torch.arange(FFT_offset, F + FFT_offset)

    # Remove the FFT offsets using slicing
    OFDM_demod = RX_NO_CP[:, remaining_indices]

    return OFDM_demod


######################################################################

def channelEstimate_LS(TTI_mask_RE, pilot_symbols, F, FFT_offset, Sp, OFDM_demod, plotEst=False):
    '''
    Calculate the channel estimate using least squares in PyTorch

    Parameters
    ----------
    TTI_mask_RE : Tensor
        TTI mask
    pilot_symbols : list
        list of pilot symbols
    F : int
        number of subcarriers
    FFT_offset : int
        FFT offset
    Sp : int
        pilot symbol
    OFDM_demod : Tensor
        OFDM data without cyclic prefix and FFT offsets
    plotEst : bool
        plot the channel estimate

    Returns 
    -------
    ndarray
        channel estimate
    '''

    # Pilot extraction
    pilots = OFDM_demod[TTI_mask_RE == 2]

    # Divide the pilots by the set pilot values
    H_estim_at_pilots = pilots / torch.tensor(pilot_symbols, dtype=torch.complex64)

    # Use NumPy for interpolation (PyTorch doesn't support interpolation yet)
    pilot_indices = torch.nonzero(TTI_mask_RE[Sp] == 2).numpy().flatten()
    H_estim_abs = scipy.interpolate.interp1d(pilot_indices, torch.abs(H_estim_at_pilots).numpy(), kind='linear', fill_value="extrapolate")(np.arange(FFT_offset, FFT_offset + F))
    H_estim_phase = scipy.interpolate.interp1d(pilot_indices, torch.angle(H_estim_at_pilots).numpy(), kind='linear', fill_value="extrapolate")(np.arange(FFT_offset, FFT_offset + F))
    H_estim = H_estim_abs * np.exp(1j * H_estim_phase)

    # Plotting
    if plotEst:
        plt.figure(0, figsize=(8,3))
        plt.plot(pilot_indices, torch.abs(H_estim_at_pilots).numpy(), '.-', label='Pilot estimates', markersize=20)
        plt.plot(np.arange(FFT_offset, FFT_offset + F), abs(H_estim), label='Estimated channel')
        plt.grid(True); plt.xlabel('Subcarrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
        plt.title('Estimated pilots and interpolated channel estimate')
        plt.savefig('pics/ChannelEstimate.png')
        plt.show()

    return H_estim

######################################################################


def equalize_ZF(OFDM_demod, H_estim, F, S):
    '''
    Equalize the OFDM data using ZF (Zero Forcing) in PyTorch

    Parameters
    ----------
    OFDM_demod : Tensor
        OFDM data without cyclic prefix and FFT offsets
    H_estim : Tensor
        channel estimate
    F : int
        number of subcarriers
    S : int
        number of symbols

    Returns
    ------- 
    Tensor
        Equalized OFDM data
    '''
    # Reshape the OFDM data and perform equalization
    equalized = (OFDM_demod.view(S, F) / H_estim)
    return equalized


######################################################################


def get_payload_symbols(TTI_mask_RE, equalized, FFT_offset, F, plotQAM=False):
    '''
    Extract all symbols with 1 in TTI mask using PyTorch

    Parameters
    ----------
    TTI_mask_RE : Tensor   
        TTI mask
    equalized : Tensor
        OFDM data without cyclic prefix and FFT offsets
    FFT_offset : int
        FFT offset
    F : int
        number of subcarriers
    plotQAM : bool
        plot the QAM symbols

    Returns
    -------
    Tensor
        payload symbols
    '''
    # Extract payload symbols
    out = equalized[TTI_mask_RE[:, FFT_offset:FFT_offset + F] == 1]

    # Plotting the QAM symbols
    if plotQAM:
        plt.figure(0, figsize=(8, 8))
        plt.plot(out.real, out.imag, 'b+')
        plt.ylim([-1.5, 1.5])
        plt.xlim([-1.5, 1.5])
        plt.xlabel('real')
        plt.ylabel('imaginary')
        plt.title('Received QAM symbols')
        plt.savefig('pics/RXdSymbols.png')
        plt.show()

    return out


######################################################################

def SINR(rx_signal, n_SINR, index):
    '''
    SINR calculation using PyTorch

    Parameters
    ----------
    rx_signal : Tensor
        Received signal
    n_SINR : int
        Number of samples for noise estimation
    index : int
        Start index for signal power calculation

    Returns
    -------
    float
        SINR
    '''
    # Calculate noise power
    rx_noise_0 = rx_signal[index - n_SINR:index]
    rx_noise_power_0 = torch.mean(torch.abs(rx_noise_0) ** 2)

    # Calculate signal power
    rx_signal_0 = rx_signal[index:index + (14 * 72)]
    rx_signal_power_0 = torch.mean(torch.abs(rx_signal_0) ** 2)

    # Compute SINR
    SINR = 10 * torch.log10(rx_signal_power_0 / rx_noise_power_0)
    SINR = round(SINR.item(), 1)

    return SINR

######################################################################


def Demapping_PS(QAM, de_mapping_table):
    constellation = torch.tensor(list(de_mapping_table.keys()))  # possible constellation points
    dists = torch.abs(QAM.view(-1, 1) - constellation.view(1, -1))  # distance of RX points to constellation points
    const_index = torch.argmin(dists, dim=1)  # pick the nearest constellation point
    hardDecision = constellation[const_index]  # get the transmitted constellation point

    # Convert keys to string format for consistent lookup
    string_key_table = {str(key.numpy()): value for key, value in de_mapping_table.items()}

    # Extract demapped symbols using the string formatted keys
    demapped_symbols = torch.tensor([string_key_table[str(c.numpy())] for c in hardDecision], dtype=torch.int32)

    return demapped_symbols, hardDecision



#######################################################################

def CP_removal(rx_signal, TTI_start, S, FFT_size, CP, plotsig=False):
    '''
    CP removal using PyTorch

    Parameters
    ----------
    rx_signal : Tensor
        received signal
    TTI_start : int
        TTI start index
    S : int
        number of symbols in TTI
    FFT_size : int
        FFT size
    CP : int
        cyclic prefix length

    Returns 
    -------
    Tensor
        signal without CP
    '''
    
    # Initialize a payload mask
    b_payload = torch.zeros(len(rx_signal), dtype=torch.bool)

    # Mark the payload parts of the signal
    for s in range(S):
        start_idx = TTI_start + (s + 1) * CP + s * FFT_size
        end_idx = start_idx + FFT_size
        b_payload[start_idx:end_idx] = 1

    # Plotting the received signal and payload mask
    if plotsig:
        # Assuming rx_signal is a PyTorch tensor
        # Convert to NumPy array and ensure it's in a float format to avoid overflow
        rx_signal_numpy = rx_signal.cpu().numpy()  # Use .cpu() if rx_signal is on GPU
        rx_signal_normalized = rx_signal_numpy / np.max(np.abs(rx_signal_numpy))

        plt.figure(0, figsize=(8, 4))
        plt.plot(rx_signal_normalized)
        plt.plot(b_payload, label='Payload Mask')
        plt.xlabel('Sample index')
        plt.ylabel('Amplitude')
        plt.title('Received signal and payload mask')
        plt.legend()
        plt.savefig('pics/RXsignal_sync.png')
        plt.show()

    # Remove the cyclic prefix
    rx_signal_no_CP = rx_signal[b_payload]

    return rx_signal_no_CP.view(S, FFT_size)


#######################################################################
import torch.nn.functional as F
def sync_TTI(tx_signal, rx_signal, leading_zeros, minimum_corr=0.3):
    '''
    Synchronization using cross-correlation in PyTorch.

    Parameters:
    ----------
    tx_signal : Tensor
        Transmitted signal.
    rx_signal : Tensor
        Received signal.
    leading_zeros : int
        Number of leading zeros for noise measurement.
    minimum_corr : float
        Minimum correlation for sync.

    Returns:
    -------
    int
        TTI start index or -1 if synchronization fails.
    '''

    # Take the absolute values of TX and RX signals
    tx_signal = torch.abs(tx_signal)
    rx_signal = torch.abs(rx_signal)

    # Compute the lengths of the signals
    tx_len = tx_signal.numel()
    rx_len = rx_signal.numel()

    # Calculate the cross-correlation using conv1d
    corr_result = F.conv1d(rx_signal.view(1, 1, -1), tx_signal.view(1, 1, -1)).view(-1)

    # Find the offset with the maximum correlation
    offset = torch.argmax(corr_result).item()
    if offset > rx_len - tx_len + leading_zeros:
        offset = offset - rx_len - leading_zeros

    # Adjust offset for leading zeros
    offset = offset + leading_zeros

    # Check for minimum correlation
    if corr_result[offset] < minimum_corr:
        return -1

    return offset

   

#######################################################################

def PSD_plot(signal, Fs, f, info='TX'):
    '''
    Plot the PSD of the signal

    Parameters
    ----------
    signal : Tensor or ndarray
        The signal (either PyTorch tensor or NumPy array)
    Fs : int
        Sampling frequency
    f : int
        Center frequency
    info : str
        Information string to include in the title

    Returns
    -------
    None
    '''

    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    plt.figure(figsize=(8, 3))
    plt.psd(signal, Fs=Fs, NFFT=1024, Fc=f, color='blue')
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB/Hz]')
    plt.title(f'Power Spectral Density, {info}')
    plt.savefig(f'pics/PSD_{info}.png')
    plt.show()

#######################################################################
   
def generate_cdl_c_impulse_response(tx_signal, num_samples=1000, sampling_rate=30.72e6, SINR=20, repeats=1, random_start=False):
    '''
    Generate a CDL-C channel impulse response and convolve it with the transmitted signal in PyTorch

    Parameters
    ----------
    tx_signal : Tensor
        transmitted signal
    num_samples : int
        number of samples
    sampling_rate : float
        sampling rate
    SINR : float
        signal to noise ratio   
    repeats : int
        number of times the signal is repeated
    random_start : bool
        randomize the starting point of the signal

    Returns
    -------
    Tensor
        received signal
    '''

    # CDL-C Power Delay Profile (PDP) and angles
    delays = torch.tensor([0, 31.5, 63, 94.5, 126, 157.5, 189, 220.5, 252, 283.5, 315, 346.5, 378, 409.5, 441, 472.5, 504, 536.5, 568, 600, 632, 664, 696, 728, 760, 792, 824.5, 857, 889.5, 922, 954.5, 987]) * 1e-9
    powers = torch.tensor([-7.5, -23.4, -16.3, -18.9, -21, -22.8, -22.9, -27.8, -23.9, -24.9, -25.6, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7])
    pdp = torch.pow(10, powers / 10)

    # Normalize PDP
    pdp /= torch.sum(pdp)
    tap_indices = torch.round(delays * sampling_rate).type(torch.int32)
    impulse_response = torch.zeros(num_samples, dtype=torch.complex64)
    impulse_response[tap_indices] = torch.sqrt(pdp) * torch.exp(1j * 2 * np.pi * torch.rand(len(pdp)))

    # Convolve the transmitted signal with the impulse response
    rx_signal = torch.conv1d(tx_signal.view(1, 1, -1), impulse_response.view(1, 1, -1)).view(-1)

    # Function to add noise
    def add_noise(rx_signal, SINR):
        noise = torch.randn_like(rx_signal) + 1j * torch.randn_like(rx_signal)
        noise_power = torch.mean(torch.abs(rx_signal) ** 2) / (10 ** (SINR / 10))
        noise *= torch.sqrt(noise_power)
        rx_signal += noise
        return rx_signal

    # Repeat and randomize the starting point if required
    rx_signal = rx_signal[:len(tx_signal)].repeat(repeats)
    if random_start:
        rx_signal = torch.roll(rx_signal, torch.randint(0, len(tx_signal), (1,)).item())

    rx_signal = add_noise(rx_signal, SINR)

    return rx_signal


    