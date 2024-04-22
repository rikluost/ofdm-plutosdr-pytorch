import torch
from torch.utils.data import Dataset

# OFDM Parameters
Qm = 6  # bits per symbol
F = 72  # Number of subcarriers, including DC
S = 14  # Number of symbols
FFT_size = 128  # FFT size
Fp = 3  # Pilot subcarrier spacing
Sp = 2  # Pilot symbol, 0 for none
CP = 20  # Cyclic prefix
SCS = 15000  # Subcarrier spacing
P = F // Fp  # Number of pilot subcarriers
FFT_offset = int((FFT_size - F) / 2)  # FFT offset
SampleRate = FFT_size * SCS  # Sample rate
Ts = 1 / (SCS * FFT_size)  # Sample duration
TTI_duration = Ts * (FFT_size + CP) * S * 1000  # TTI duration in ms
Pilot_Power = 1  # Pilot power
PDSCH_power = 1  # PDSCH power

# channel simulation
n_taps = 1 
max_delay = 3 #samples

# Additional Parameters
leading_zeros = 80  # Number of symbols with zero value for noise measurement at the beginning of the transmission. Used for SINR estimation.

# Save the generated plots
save_plots = True

# custom dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.pdsch_iq = [] # pdsch symbols
        #self.pilot_iq = [] # pilot symbols
        self.labels = [] # original bitstream labels
        
    def __len__(self):
        return len(self.pdsch_iq)
    
    def __getitem__(self, index):
        x1 = self.pdsch_iq[index]
        #x2 = self.pilot_iq[index] 
        y = self.labels[index]
        return x1, y
    
    #def add_item(self, new_pdsch_iq, new_pilot_iq, new_label):
    def add_item(self, new_pdsch_iq,  new_label):
        self.pdsch_iq.append(new_pdsch_iq) 
        #self.pilot_iq.append(new_pilot_iq) 
        self.labels.append(new_label) 