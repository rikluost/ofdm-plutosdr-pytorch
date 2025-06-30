import torch
from torch.utils.data import Dataset

# OFDM Parameters
Qm = 4  # bits per symbol
F = 102  # Number of subcarriers, including DC
S = 14  # Number of symbols
FFT_size = 128  # FFT size
FFT_size_RX = 128  # FFT size

Fp = 8 # Pilot subcarrier spacing
Sp = 2  # Pilot symbol, 0 for none
CP = 7  # Cyclic prefix
SCS = 15000  # Subcarrier spacing
P = F // Fp  # Number of pilot subcarriers
FFT_offset = int((FFT_size - F) / 2)  # FFT offset
FFT_offset_RX = int((FFT_size_RX - F) / 2)  # FFT offset

SampleRate = FFT_size * SCS  # Sample rate
Ts = 1 / (SCS * FFT_size)  # Sample duration
TTI_duration = Ts * (FFT_size + CP) * S * 1000  # TTI duration in ms


####### SDR ###################

SDR_RX_IP="ip:192.168.10.191"
SDR_TX_IP="ip:192.168.10.191"

SDR_TX_Frequency = int(3500e6)  # base band center frequency
SDR_TX_BANDWIDTH = SCS*F*4
tx_gain = -5  # Transmission gain in dB for SDR
rx_gain = 30  # Reception gain in dB for SDR

TX_Scale = 0.7

######################################

# Additional Parameters
leading_zeros = 500  # Number of symbols with zero value for noise measurement at the beginning of the transmission. Used for SINR estimation.

# Save the generated plots
save_plots = False

# custom dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.pdsch_iq = [] # pdsch symbols
        self.labels = [] # original bitstream labels
        self.sinr = [] # SINR

    def __len__(self):
        return len(self.pdsch_iq)
    
    def __getitem__(self, index):
        x1 = self.pdsch_iq[index]
        y = self.labels[index]
        z = self.sinr[index]
        return x1, y, z
    
    def add_item(self, new_pdsch_iq,  new_label, new_sinr):
        self.pdsch_iq.append(new_pdsch_iq) 
        self.labels.append(new_label) 
        self.sinr.append(new_sinr) 
        