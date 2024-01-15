import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as tFunc
import config
import os

save_plots = config.save_plots

######################################################################

def mapping_table(Qm, plot=False):
        
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
    mapping = {index_to_binary(i, Qm): val for i, val in enumerate(C)}
    
    # Create the demapping table
    demapping = {v: k for k, v in mapping.items()}

    if plot:
        visualize_constellation(C, Qm)    
    return mapping, demapping

######################################################################

def visualize_constellation(pts, Qm):
    import matplotlib.pyplot as plt
    plt.scatter(pts.numpy().real, pts.numpy().imag)
    plt.title(f'Modulation Order {Qm} Constellation')
    plt.ylabel('Imaginary'); plt.xlabel('Real')
    plt.tight_layout(); 
    if save_plots:
        plt.savefig(f'pics/const.png')
    
######################################################################

def TTI_mask(S, F, Fp, Sp, FFT_offset, plotTTI=False):

    # Create a mask with all ones
    TTI_mask = torch.ones((S, F), dtype=torch.int8) # all ones

    # Set symbol Sp for pilots
    TTI_mask[Sp, torch.arange(0, F, Fp)] = 2 # for pilots TX1
    # Ensure the first subcarrier is a pilot
    TTI_mask[Sp, 0] = 2

    # Ensure the last subcarrier is a pilot
    TTI_mask[Sp, F - 1] = 2

    # DC
    TTI_mask[:, F // 2] = 3 # DC to non-allocable power (oscillator phase noise)

    # Add FFT offsets
    TTI_mask = torch.cat((torch.zeros(S, FFT_offset, dtype=torch.int8), TTI_mask, torch.zeros(S, FFT_offset, dtype=torch.int8)), dim=1)

    # Plotting the TTI mask
    if plotTTI:
        plt.figure(figsize=(8, 1.5))
        plt.imshow(TTI_mask.numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.title('TTI mask')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        if save_plots:
            plt.savefig('pics/TTImask.png')
        plt.tight_layout()
        plt.show()

    return TTI_mask

######################################################################

def pilot_set(TTI_mask, power_scaling=1.0):

    # Define QPSK pilot values
    pilot_values = torch.tensor([-0.7 - 0.7j, -0.7 + 0.7j, 0.7 - 0.7j, 0.7 + 0.7j]) * power_scaling

    # Count the number of pilot elements in the TTI mask
    num_pilots = TTI_mask[TTI_mask == 2].numel() 

    # Create a list of pilot values repeated to match the number of pilots
    pilots = pilot_values.repeat(num_pilots//4+1)[:num_pilots]

    return pilots


######################################################################

def create_PDSCH_data(TTI_mask, Qm, mapping_table, power=1.):
    # Count PDSCH elements
    pdsch_elems = TTI_mask.eq(1).sum()
    
    # Generate random bits
    bits = torch.randint(0, 2, (pdsch_elems, Qm), dtype=torch.float32)

    # Flatten and reshape bits for symbol lookup
    flattened_bits = bits.reshape(-1, Qm)
   
    # Convert bits to tuples and lookup symbols
    symbs_list = [mapping_table[tuple(row.tolist())] for row in flattened_bits]
    symbs = torch.tensor(symbs_list, dtype=torch.complex64)

    # Reshape symbols back to original shape
    symbs = symbs.view(pdsch_elems, -1)
    symbs = symbs.flatten()
    
    # Apply power scaling 
    symbs *= power
    
    return bits, symbs

######################################################################

def RE_mapping(TTI_mask, pilot_set, pdsch_symbols, plotTTI=False): 

    # Create a zero tensor for the overall F subcarriers * S symbols
    TTI = torch.zeros(TTI_mask.shape, dtype=torch.complex64)

    # Allocate the payload and pilot
    TTI[TTI_mask==1] = pdsch_symbols.clone().detach()
    TTI[TTI_mask==2] = pilot_set.clone().detach()
    # Plotting the TTI modulated symbols
    if plotTTI:
        plt.figure(figsize=(8, 1.5))
        plt.imshow(torch.abs(TTI).numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.title('TTI modulated symbols')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        if save_plots:
            plt.savefig('pics/TTImod.png')
        plt.show()

    return TTI


######################################################################

def FFT(TTI):

    return torch.fft.ifft(torch.fft.ifftshift(TTI, dim=1))


######################################################################

def CP_addition(OFDM_data, S, FFT_size, CP):

    # Initialize output tensor
    out = torch.zeros((S, FFT_size + CP), dtype=torch.complex64)

    # Add cyclic prefix to each symbol
    for symbol in range(S):
        out[symbol, :] = torch.cat((OFDM_data[symbol, -CP:], OFDM_data[symbol, :]), dim=0)

    return out.flatten()


######################################################################

def SP(bits, length, Qm): # serial to parallel
    return bits.view(length, Qm)


######################################################################

def PS(bits): # parallel to serial
    return bits.view(-1)


######################################################################

def DFT(rxsignal, plotDFT=False):

    # Calculate DFT
    OFDM_RX_DFT = torch.fft.fftshift(torch.fft.fft(rxsignal, dim=1), dim=1)

    # Plot the DFT if required
    if plotDFT:
        plt.figure(figsize=(8, 1.5))
        plt.imshow(torch.abs(OFDM_RX_DFT).numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Symbol')
        if save_plots:
            plt.savefig('pics/TTI_RX.png')
        plt.show()

    return OFDM_RX_DFT


######################################################################

def remove_fft_Offests(RX_NO_CP, F, FFT_offset):

    # Calculate indices for the remaining subcarriers after removing offsets
    remaining_indices = torch.arange(FFT_offset, F + FFT_offset)

    # Remove the FFT offsets using slicing
    OFDM_demod = RX_NO_CP[:, remaining_indices]

    return OFDM_demod


######################################################################


def torch_interp(x, xp, fp):
    # Ensure xp and fp are sorted
    sorted_indices = torch.argsort(xp)
    xp = xp[sorted_indices].to(device=x.device)
    fp = fp[sorted_indices].to(device=x.device)

    # Find the indices to the left and right of x
    indices_left = torch.searchsorted(xp, x, right=True)
    indices_left = torch.clamp(indices_left, 0, len(xp) - 1)

    indices_right = torch.clamp(indices_left - 1, 0, len(xp) - 1)

    # Perform linear interpolation
    x_left, x_right = xp[indices_left], xp[indices_right]
    f_left, f_right = fp[indices_left], fp[indices_right]
    interp_values = f_left + (f_right - f_left) * (x - x_left) / (x_right - x_left)

    return interp_values

######################################################################

def channelEstimate_LS(TTI_mask_RE, pilot_symbols, F, FFT_offset, Sp, OFDM_demod, plotEst=False):
    # Pilot extraction
    pilots = OFDM_demod[TTI_mask_RE == 2]

    # Divide the pilots by the set pilot values
    H_estim_at_pilots = pilots / pilot_symbols

    # Interpolation indices
    pilot_indices = torch.nonzero(TTI_mask_RE[Sp] == 2, as_tuple=False).squeeze()

    # Interpolation for magnitude and phase
    all_indices = torch.arange(FFT_offset, FFT_offset + F)
    
    # Linear interpolation for magnitude and phase
    H_estim_abs = torch_interp(all_indices, pilot_indices, torch.abs(H_estim_at_pilots))
    H_estim_phase = torch_interp(all_indices, pilot_indices, torch.angle(H_estim_at_pilots))

    # Convert magnitude and phase to complex numbers
    H_estim_real = H_estim_abs * torch.exp(1j * H_estim_phase).real
    H_estim_imag = H_estim_abs * torch.exp(1j * H_estim_phase).imag

    # Combine real and imaginary parts to form complex numbers
    H_estim = torch.view_as_complex(torch.stack([H_estim_real, H_estim_imag], dim=-1))


    def dB(x):
        return 10 * torch.log10(torch.abs(x))

    if plotEst:
        plt.figure(figsize=(8, 4))
        plt.plot(pilot_indices.cpu().numpy(), dB(H_estim_at_pilots).cpu().numpy(), 'ro-', label='Pilot estimates', markersize=8)
        plt.plot(all_indices.cpu().numpy(), dB(torch.tensor(H_estim)).cpu().numpy(), 'b-', label='Estimated channel', linewidth=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlabel('Subcarrier Index', fontsize=12)
        plt.ylabel('Magnitude (dB)', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.title('Pilot Estimates and Interpolated Channel in dB', fontsize=14)
        if save_plots:
            plt.savefig('pics/ChannelEstimate.png')
        plt.show()

    return H_estim


######################################################################

def equalize_ZF(OFDM_demod, H_estim, F, S):

    # Reshape the OFDM data and perform equalization
    equalized = (OFDM_demod.view(S, F) / H_estim.to(device=OFDM_demod.device))
    return equalized

######################################################################

def get_payload_symbols(TTI_mask_RE, equalized, FFT_offset, F, plotQAM=False):
    # Extract payload symbols
    mask = TTI_mask_RE[:, FFT_offset:FFT_offset + F] == 1
    out = equalized[mask]

    # Plotting the QAM symbols
    if plotQAM:
        plt.figure(figsize=(8, 8))
        plt.scatter(out.cpu().real, out.cpu().imag, label='QAM Symbols')
        plt.axis('equal')
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.xlabel('Real Part', fontsize=12)
        plt.ylabel('Imaginary Part', fontsize=12)
        plt.title('Received QAM Constellation Diagram', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', fontsize=10)
        if save_plots:
            plt.savefig('pics/RXdSymbols.png')
        plt.show()
    return out

######################################################################

def get_payload_symbols_raw(TTI_mask_RE, equalized):
    # Extract payload symbols in the shape of the TTI mask
    out = equalized[TTI_mask_RE == 1]
    out_orig = equalized * (TTI_mask_RE==1).int()

    return out, out_orig

#######################################################################
def get_pilot_symbols_raw(TTI_mask_RE, equalized):
    # Extract pilot symbols in the shape of the TTI mask
    out = equalized[TTI_mask_RE == 2]
    out_orig = equalized * (TTI_mask_RE==2).int()

    return out, out_orig

######################################################################

def SINR(rx_signal, index, leading_zeros):
    # Calculate noise power

    rx_noise_0 = rx_signal[index - leading_zeros + 20 : index - 20]
    rx_noise_power_0 = torch.mean(torch.abs(rx_noise_0) ** 2)

    # Calculate signal power
    rx_signal_0 = rx_signal[index:index + (14 * 72)]
    rx_signal_power_0 = torch.mean(torch.abs(rx_signal_0) ** 2)

    # Compute SINR
    SINR = 10 * torch.log10(rx_signal_power_0 / rx_noise_power_0)

    # Round the SINR value
    SINR = round(SINR.item(), 1)

    return SINR, rx_noise_power_0, rx_signal_power_0

######################################################################


def Demapping(QAM, de_mapping_table):
    # Convert the demapping table keys (constellation points) to a tensor
    constellation = torch.tensor(list(de_mapping_table.keys())).to(QAM.device)
    dists = torch.abs(QAM.view(-1, 1) - constellation.view(1, -1))
    const_index = torch.argmin(dists, dim=1).to(QAM.device)
    hardDecision = constellation[const_index].to(QAM.device)
    string_key_table = {str(key.item()): value for key, value in de_mapping_table.items()}
    demapped_symbols = torch.tensor([string_key_table[str(c.item())] for c in hardDecision], dtype=torch.int32)
    return demapped_symbols, hardDecision


#######################################################################

def CP_removal(rx_signal, TTI_start, S, FFT_size, CP, plotsig=False):

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
        if save_plots:
            plt.savefig('pics/RXsignal_sync.png')
        plt.show()

    # Remove the cyclic prefix
    rx_signal_no_CP = rx_signal[b_payload]

    return rx_signal_no_CP.view(S, FFT_size)


#######################################################################

def sync_TTI(tx_signal, rx_signal, leading_zeros, threshold=6, plot=False):

    # time sync using correlation

    tx_len = tx_signal.numel()
    rx_len = rx_signal.numel()
    end_point=rx_len-tx_len

    rx_signal = rx_signal[leading_zeros:end_point]

    # Calculate the cross-correlation using conv1d
    corr_result_real = tFunc.conv1d(rx_signal.real.view(1, 1, -1), tx_signal.real.view(1, 1, -1)).view(-1)
    corr_result_imag = tFunc.conv1d(rx_signal.imag.view(1, 1, -1), tx_signal.imag.view(1, 1, -1)).view(-1)
    correlation =  torch.complex(corr_result_real, corr_result_imag)
    correlation = correlation.abs()
    threshold = correlation.mean()*threshold

    # Find the first peak that exceeds the threshold
    for i, value in enumerate(correlation):
        if value > threshold:
            if plot:
                plt.figure(figsize=(8, 4))
                plt.plot(correlation.abs()[i-10:i+50])
                plt.grid()
                plt.xlabel("Samples from selected index")
                plt.ylabel("Complex conjugate correlation")
                plt.axvline(x=10, color = 'r', linewidth=3)
                if save_plots:
                    plt.savefig('pics/corr.png')
                plt.show()

            return i + leading_zeros
        
    i = torch.argmax(correlation).item() + leading_zeros

    return i

#######################################################################

def PSD_plot(signal, Fs, f, info='TX'):

    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    plt.figure(figsize=(8, 3))
    plt.psd(signal, Fs=Fs, NFFT=1024, Fc=f, color='blue')
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB/Hz]')
    plt.title(f'Power Spectral Density, {info}')
    if save_plots:
        plt.savefig(f'pics/PSD_{info}.png')
    plt.show()


#######################################################################

# Function to check if file exists and return a new filename
def get_new_filename(directory, base_filename):
    os.makedirs(directory, exist_ok=True)
    nn = 0
    while True:
        new_filename = f"{base_filename}_{nn}.csv"
        if not os.path.exists(os.path.join(directory, new_filename)):
            return new_filename
        nn += 1

#######################################################################

def apply_multipath_channel(signal, n_taps, max_delay, repeats=0, random_start=True, SINR_s=30, leading_zeros=500):
    # note that the output is max_delay longer than input, due to the delayed symbols of some of the taps

    h = torch.zeros(max_delay+1, dtype=torch.complex64)

    for _ in range(n_taps):
        delay = torch.randint(1, max_delay, (1,)).item()  # Avoid delay=0 for remaining taps
        magnitude = torch.abs(torch.randn(1)) * 0.5
        phase = torch.randn(1)
        complex_gain = magnitude * torch.exp(1j * phase)
        h[delay] = complex_gain

    h[0] = torch.complex(torch.randn(1), torch.randn(1))

    # Normalize the channel response
    h = h / torch.norm(h)

    # Convolve the signal with the channel
    convolved_signal = torch.nn.functional.conv1d(signal.unsqueeze(0).unsqueeze(0), 
                                               h.unsqueeze(0).unsqueeze(0), 
                                               padding=max_delay).squeeze()
    
    # add leading zeros
    zeros = torch.zeros(leading_zeros, dtype=convolved_signal.dtype)
    convolved_signal = torch.cat((zeros, convolved_signal), dim=0)

    # Add noise based on SINR
    if SINR_s !=0:
        signal_power = torch.mean(torch.abs(convolved_signal)**2)
        noise_power = signal_power / (10 ** (SINR_s / 10))
        noise = torch.sqrt(noise_power / 2) * (torch.randn_like(convolved_signal) + 1j * torch.randn_like(convolved_signal))
        convolved_signal = convolved_signal + noise

    # Add random start if required
    if random_start:
        start_index = torch.randint(0, len(convolved_signal), (1,)).item()
        convolved_signal = torch.roll(convolved_signal, shifts=start_index)    
    
    # Repeat the signal if required
    if repeats > 0:
        convolved_signal = convolved_signal.repeat(repeats)

    return convolved_signal
