import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy import interpolate


######################################################################

def mapping_table(Qm, plotMap=False):
    
    '''
    Mapping table for QAM modulation
    
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

    size = int(np.sqrt(2**Qm))  # size of the constellation
    a = np.arange(size)  # the constellation
    b = a - np.mean(a)  # the constellation shifted to the center
    # Use broadcasting to create the complex constellation grid
    C = (b[:, np.newaxis] + 1j * b).flatten()
    # Normalize the constellation
    C /= np.sqrt(np.mean(np.abs(C)**2)).astype(np.complex64)  # ensure DC offset is 0

    # Create the mapping table using np.fromfunction
    def index_to_binary(i, Qm):
        return tuple(map(int, '{:0{}b}'.format(int(i), Qm)))

    # Apply the function to each index and create the mapping dictionary
    mapping_out = {index_to_binary(i, Qm): val for i, val in enumerate(C)}

    # Create the demapping table
    demapping_out = {v: k for k, v in mapping_out.items()}

    if plotMap:
        plt.figure(figsize=(4,4))    
        plt.plot(np.real(list(mapping_out.values())), np.imag(list(mapping_out.values())), 'o')
        plt.grid()
        plt.xlabel('real')
        plt.ylabel('imaginary')
        plt.title(f'Constellation, modulation order {Qm}')
        plt.savefig('pics/QAMconstellation.png')
        plt.show()
    
    return mapping_out, demapping_out # mapping table, demapping table

######################################################################

def TTI_mask(S,F, Fp, Sp, FFT_offset, plotTTI=False):
    '''
    create TTI resource elements mask

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
    ndarray
        TTI mask

    --------------------
    0 = DRX
    1 = PDSCH
    2 = pilots
    5 = DC
    '''
    # create a mask with all ones
    TTI_mask = np.ones((S,F), dtype='int0') # all ones

    #  symbol Sp for pilots
    TTI_mask[Sp,0:F:Fp] = 2 # for pilots TX1
        
   # DC
    TTI_mask[:,int(F/2)] = 5 # DC to non-allocable power (oscillator phase noise)

    # add FFT offsets
    TTI_mask = np.hstack([np.zeros([S, FFT_offset]), TTI_mask, np.zeros([S, FFT_offset])]).astype('int0') # add FFT offsets
    
    if plotTTI:
        plt.figure(figsize=(8,1.5))
        plt.imshow(np.abs(TTI_mask))
        plt.title('TTI mask')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        plt.savefig('pics/TTImask.png')
        plt.show()
    return TTI_mask

######################################################################

def pilot_set(TTI_mask, power=1):
    ''' 
    Create simple pilot QPSK set

    Parameters
    ----------
    TTI_mask : ndarray
        TTI mask
    power : float
        Power of the pilot

    Returns
    -------
    ndarray
        pilot set

    '''
    pilots = list(itertools.repeat([-0.7-0.7j,-0.7+0.7j,0.7-0.7j,0.7+0.7j],np.count_nonzero(TTI_mask == 2)))
    pilots = [item * power for sublist in pilots for item in sublist][0:np.count_nonzero(TTI_mask == 2)]

    return pilots

######################################################################

def create_PDSCH_data(TTI_mask, Qm, mapping_table, power = 1, count=None):
    '''
    create PDSCH data and symbols

    Parameters
    ----------
    TTI_mask : ndarray
        TTI mask
    Qm : int
        Modulation order
    mapping_table : dict
        Mapping table
    count : int
        Number of symbols to generate

    Returns
    -------
    ndarray
        PDSCH data
    ndarray
        PDSCH symbols

    '''
    if count is None:
        count = np.count_nonzero(TTI_mask == 1)
    bits = np.random.binomial(n=1, p=0.5, size=(count*Qm))
    pdsch_bits = bits.reshape((int(len(bits)/Qm), Qm))
    pdsch_symbols = np.array([mapping_table[tuple(b)] for b in pdsch_bits])
    pdsch_symbols = [item * power for item in pdsch_symbols]

    return pdsch_bits, pdsch_symbols


######################################################################

def RE_mapping(TTI_mask, pilot_set, pdsch_symbols, plotTTI=False): 
    '''
    OFDM TTI in frequency domain, resource elements mapping

    Parameters
    ----------
    TTI_mask : ndarray
        TTI mask
    pilot_set : list
        list of pilot symbols
    pdsch_symbols : list
        list of PDSCH symbols
    plotTTI : bool
        plot TTI mask

    Returns
    -------
    ndarray
        mapped TTI

    '''

    TTI = np.zeros(TTI_mask.shape, dtype=complex) # the overall F subcarriers * S symbols
    TTI[TTI_mask==1] = pdsch_symbols # allocate the payload
    TTI[TTI_mask==2] = pilot_set  # allocate the pilot 

    if plotTTI:
        plt.figure(figsize=(8,1.5))
        plt.imshow(np.abs(TTI))
        plt.title('TTI modulated symbols')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        plt.savefig('pics/TTImod.png')
        plt.show()
    return TTI

######################################################################

def FFT(TTI):
    '''
    OFDM TTI in time domain, FFT

    Parameters
    ----------
    TTI : ndarray
        TTI modulated symbols

    Returns
    -------
    ndarray
        TTI in frequency domain

    '''
    return np.fft.ifft(np.fft.ifftshift(TTI, axes=1))

######################################################################

def CP_addition(OFDM_data, S, FFT_size, CP):

    '''
    OFDM TTI in time domain with cyclic prefix for each symbol

    Parameters
    ----------
    OFDM_data : ndarray
        TTI in frequency domain
    S : int
        number of symbols
    FFT_size : int
        FFT size
    CP : int
        cyclic prefix length

    Returns 
    -------
    ndarray
        TTI in time domain with cyclic prefix


    '''
    if OFDM_data is None:
        OFDM_data = OFDM_data
    out = np.zeros((S,FFT_size+CP), dtype=complex)
    for symbol in range(S):
        out[symbol,:] = np.hstack([OFDM_data[symbol, -CP:], OFDM_data[symbol, :]])
    return out.flatten()

######################################################################

def SP(bits, length, Qm): # serial to parallel
    '''
    Serial to parallel

    Parameters
    ----------
    bits : ndarray
        bits
    length : int
        length of the array
    Qm : int
        Modulation order

    Returns
    -------
    ndarray
        bits

    '''
    return bits.reshape((length, Qm))

######################################################################

def PS(bits): # parallel to serial
    '''
    Parallel to serial
    
    Parameters
    ----------
    bits : ndarray
        bits
    
    Returns
    -------
    ndarray
        bits   
            '''


    return bits.reshape((-1,))

######################################################################

def DFT(rxsignal, plotDFT=False):
    '''
    Calculate the DFT of the received signal
    
    Parameters
    ----------
    rx_signal : ndarray
        OFDM data
    plotDFT : bool
        plot the DFT

    Returns
    -------
    ndarray 
        DFT of the received signal

    '''
    OFDM_RX_DFT = np.fft.fftshift(np.fft.fft(rxsignal, axis=1), axes=1)
    
    # remove the OFDM offsets
    if plotDFT==True:
        plt.figure(figsize=(8,1.5))
        plt.imshow(abs(OFDM_RX_DFT))
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Symbol')
        plt.savefig('pics/TTI_RX.png')
        plt.show()
    return OFDM_RX_DFT

######################################################################

def remove_fft_Offests(RX_NO_CP, F, FFT_offset):
    '''
    Remove the FFT offsets

    Parameters
    ----------
    RX_NO_CP : ndarray
        OFDM data without cyclic prefix
    F : int
        number of subcarriers
    FFT_offset : int
        FFT offset

    Returns 
    ------- 
    ndarray
        OFDM data without cyclic prefix and FFT offsets
    '''

    zero_subc_idx_low = np.arange(0,FFT_offset)
    zero_subc_idx_high = np.arange(F,FFT_offset+F)
    OFDM_demod = np.delete(RX_NO_CP,zero_subc_idx_low,1)
    OFDM_demod = np.delete(OFDM_demod,zero_subc_idx_high,1)

    return OFDM_demod

######################################################################

def channelEstimate_LS(TTI_mask_RE, pilot_symbols, F, FFT_offset, Sp, OFDM_demod, plotEst=False):

    '''
    Calculate the channel estimate using least squares

    Parameters
    ----------
    TTI_mask_RE : ndarray
        TTI mask
    pilot_symbols : list
        list of pilot symbols
    F : int
        number of subcarriers
    FFT_offset : int
        FFT offset
    Sp : int
        pilot symbol
    OFDM_demod : ndarray
        OFDM data without cyclic prefix and FFT offsets
    plotEst : bool
        plot the channel estimate

    Returns 
    -------
    ndarray
        channel estimate

    '''
    pilots = OFDM_demod[TTI_mask_RE == 2]  # pilot extraction
    H_estim_at_pilots = pilots / pilot_symbols # divide the pilots by the set pilot values
    
    H_estim_abs = scipy.interpolate.interp1d(np.flatnonzero(TTI_mask_RE[Sp]==2), np.abs(H_estim_at_pilots), kind='linear', fill_value="extrapolate")(np.arange(FFT_offset, FFT_offset+F))
    H_estim_phase = scipy.interpolate.interp1d(np.flatnonzero(TTI_mask_RE[Sp]==2), np.angle(H_estim_at_pilots), kind='linear', fill_value="extrapolate")(np.arange(FFT_offset, FFT_offset+F))
    H_estim = H_estim_abs * np.exp(1j*H_estim_phase)
    
    if plotEst:
        plt.figure(0, figsize=(8,3))
        plt.plot(np.flatnonzero(TTI_mask_RE[Sp]==2), abs(H_estim_at_pilots),'.-', label='Pilot estimates', markersize=20)
        plt.plot(np.arange(np.min(np.where(TTI_mask_RE[Sp]==2)), FFT_offset+F), abs(H_estim), label='Estimated channel')
        plt.grid(True); plt.xlabel('Subcarrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
        plt.title('Estimated pilots and interpolated channel estimate')
        plt.savefig('pics/ChannelEstimate.png')
        plt.show()
    return H_estim

######################################################################


def equalize_ZF(OFDM_demod, H_estim, F, S):
    '''
    Equalize the OFDM data using ZF

    Parameters
    ----------
    OFDM_demod : ndarray
        OFDM data without cyclic prefix and FFT offsets
    H_estim : ndarray
        channel estimate
    F : int
        number of subcarriers
    S : int
        number of symbols

    Returns
    ------- 
    ndarray
        OFDM data without cyclic prefix and FFT offsets
    '''
    equalized = (OFDM_demod.reshape(S,F) / H_estim)
    return equalized

######################################################################


def get_payload_symbols(TTI_mask_RE, equalized, FFT_offset, F, plotQAM=False):
    '''
    extract all symbols with 1 in TTI mask

    Parameters
    ----------
    TTI_mask_RE : ndarray   
        TTI mask
    equalized : ndarray
        OFDM data without cyclic prefix and FFT offsets
    FFT_offset : int
        FFT offset
    F : int
        number of subcarriers
    plotQAM : bool
        plot the QAM symbols

    Returns
    -------
    ndarray
        payload symbols

    '''
    out = equalized[TTI_mask_RE[:,FFT_offset:FFT_offset+F]==1]

    if plotQAM:
        plt.figure(0, figsize=(8,8))
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
    SINR - this is not exactly correct also the null power received outside the PDSCH are calculated

    Parameters
    ----------
    TTI_mask_RE : ndarray
        TTI mask
    OFDM_demod : ndarray
        OFDM data without cyclic prefix and FFT offsets
    FFT_offset : int
        FFT offset
    F : int
        number of subcarriers

    Returns
    -------
    float
        SINR

    '''
    rx_noise_0 =  rx_signal[index-n_SINR:index-1]
    rx_noise_power_0 = np.mean(abs(rx_noise_0)**2)
    rx_signal_0 = rx_signal[index:index+(14*72)]
    rx_signal_power_0 = np.mean(abs(rx_signal_0)**2)
    SINR = 10*np.log10(rx_signal_power_0/rx_noise_power_0)
    SINR = round(SINR, 1)

    return SINR

######################################################################

def Demapping(QAM, de_mapping_table):
    '''
    Demapping of QAM symbols

    Parameters
    ----------
    QAM : ndarray
        QAM symbols
    de_mapping_table : ndarray
        de-mapping table

    Returns
    -------
    ndarray
        demapped symbols

    '''

    constellation = np.array([x for x in de_mapping_table.keys()])    # possible constellation points
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))    # distance of RX points to constellation points
    const_index = dists.argmin(axis=1) # pick the nearest constellation point
    hardDecision = constellation[const_index]    # get the transmitted constellation point
    return np.vstack([de_mapping_table[C] for C in hardDecision]), hardDecision    # transform the constellation point into the bit groups

#######################################################################

def CP_removal(rx_signal, TTI_start, S, FFT_size, CP, plotsig=False):
    '''
    CP removal

    Parameters
    ----------
    rx_signal : ndarray
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
    ndarray
        signal without CP
    '''
    
    b_payload = np.zeros(len(rx_signal))

    for s in range(S):
        b_payload[TTI_start + ( (s+1)*CP + s*FFT_size+ np.arange(FFT_size))] = 1
    b_payload = b_payload.astype(bool)
    
    if plotsig == True:
        plt.figure(0, figsize=(8,4))
        plt.plot(rx_signal/np.max(np.abs(rx_signal)))
        plt.plot(b_payload)
        plt.xlabel('Sample index')
        plt.ylabel('Amplitude')
        plt.title('Received signal and payload mask')
        plt.savefig('pics/RXsignal_sync.png')
        plt.show()

    rx_signal = rx_signal[b_payload]

    return rx_signal.reshape(S, FFT_size)

#######################################################################

def sync_TTI(tx_signal, rx_signal, leading_zeros, minimum_corr=0.3):
    '''
    Synchronization using correlation

    Parameters
    ----------
    tx_signal : ndarray
        transmitted signal
    rx_signal : ndarray
        received signal
    leading_zeros : int
        number of leading zeros for noise measurement
    minimum_corr : float
        minimum correlation for sync

    Returns
    -------
    int
        TTI start index

    '''

    corr_array = signal.correlate(abs(rx_signal),abs(tx_signal),mode='full',method='fft')
    offset = np.argmax(abs(corr_array[len(tx_signal):int(len(rx_signal)/2)]))+1+leading_zeros
    
    if abs(corr_array[offset]) < minimum_corr:
        return -1

    return offset
   

#######################################################################

def PSD_plot(signal, Fs, f, info='TX'):
    '''
    Plot the PSD of the signal

    Parameters
    ----------
    rx_signal : ndarray
        received signal
    Fs : int
        sampling frequency
    f : int
        center frequency

    Returns
    -------
    ndarray
        PSD of the  signal

    '''

    plt.figure(figsize=(8,3)) 
    plt.psd(signal, Fs=Fs, NFFT=1024, Fc=f, color='blue')
    plt.grid(True)
    plt.xlabel('Frequency [Hz]'), plt.ylabel('PSD [dB/Hz]'), plt.title(f'Power Spectral Density, {info}')
    plt.savefig(f'pics/PSD_{info}.png')
    plt.show()
    return()

#######################################################################
   
def generate_cdl_c_impulse_response(tx_signal, num_samples=1000, sampling_rate=30.72e6, SINR=20, repeats=1, random_start=False):
    
    '''
    Generate a CDL-C channel impulse response and convolve it with the transmitted signal

    Parameters
    ----------
    tx_signal : ndarray
        transmitted signal
    num_samples : int
        number of samples
    sampling_rate : int
        sampling rate
    SINR : float
        signal to noise ratio   
    repeats : int
        number of times the signal is repeated
    random_start : bool
        randomize the starting point of the signal

    Returns
    -------
    ndarray
        received signal

    '''

    # CDL-C Power Delay Profile (PDP) and angles from 3GPP TR 38.901 Table 7.7.1-3
    # Excess tap delay [ns], Power [dB]
    delays = np.array([0, 31.5, 63, 94.5, 126, 157.5, 189, 220.5, 252, 283.5, 315, 346.5, 378, 409.5, 441, 472.5, 504, 536.5, 568, 600, 632, 664, 696, 728, 760, 792, 824.5, 857, 889.5, 922, 954.5, 987]) * 1e-9
    powers = np.array([-7.5, -23.4, -16.3, -18.9, -21, -22.8, -22.9, -27.8, -23.9, -24.9, -25.6, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7, -29.7])
    pdp = 10 ** (powers / 10)
    
    # Normalize PDP
    pdp /= np.sum(pdp)
    tap_indices = np.round(delays * sampling_rate).astype(int)
    impulse_response = np.zeros(num_samples, dtype=complex)
    impulse_response[tap_indices] = np.sqrt(pdp) * np.exp(1j * 2 * np.pi * np.random.rand(len(pdp)))
    rx_signal = np.convolve(tx_signal, impulse_response, mode='full')
    def add_noise(rx_signal, SINR):
        noise = np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal))
        noise_power = np.mean(np.abs(rx_signal) ** 2) / (10 ** (SINR / 10))
        noise *= np.sqrt(noise_power)
        rx_signal += noise
        return rx_signal
    
    rx_signal = np.tile(rx_signal[:len(tx_signal)], repeats)
    if random_start:
        rx_signal = np.roll(rx_signal, np.random.randint(0, len(tx_signal)))

    rx_signal = add_noise(rx_signal, SINR)

    return rx_signal

    