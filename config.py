import torch
from torch.utils.data import Dataset

# running on a Mx mac?
macos_m1 = True


# Configuration for SDR and 3GPP CDL-C Channel Simulation, and custom dataset

# OFDM Parameters
Qm = 6  # Modulation order
F = 72  # Number of subcarriers, including DC
S = 14  # Number of symbols
FFT_size = 128  # FFT size
Fp = 3  # Pilot subcarrier spacing
Sp = 2  # Pilot symbol, 0 for none
CP = int(FFT_size / 10)  # Cyclic prefix
SCS = 15000  # Subcarrier spacing
P = F // Fp  # Number of pilot subcarriers
sn = 0  # Serial number starting value
Qm_sn = 2  # Serial number modulation order
FFT_offset = int((FFT_size - F) / 2)  # FFT offset
SampleRate = FFT_size * SCS  # Sample rate
Ts = 1 / (SCS * FFT_size)  # Sample duration
TTI_duration = Ts * (FFT_size + CP) * S * 1000  # TTI duration in ms
Pilot_Power = 1  # Pilot power
PDSCH_power = 1  # PDSCH power

# Additional Parameters
leading_zeros = 500  # Number of symbols with zero value for noise measurement at the beginning of the transmission. Used for SINR estimation.

# Save the generated plots
save_plots = False

# custom dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.pdsch_iq = [] # pdsch symbols
        self.pilot_iq = [] # pilot symbols
        self.labels = [] # original bitstream labels
        
    def __len__(self):
        return len(self.pdsch_iq)
    
    def __getitem__(self, index):
        x1 = self.pdsch_iq[index]
        x2 = self.pilot_iq[index] 
        y = self.labels[index]
        return x1, x2, y
    
    def add_item(self, new_pdsch_iq, new_pilot_iq, new_label):
        self.pdsch_iq.append(new_pdsch_iq) 
        self.pilot_iq.append(new_pilot_iq) 
        self.labels.append(new_label) 