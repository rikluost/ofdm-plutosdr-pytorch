# Configuration for SDR and 3GPP CDL-C Channel Simulation

# OFDM Parameters
Qm = 4  # Modulation order
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
