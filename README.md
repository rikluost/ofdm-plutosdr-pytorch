# Over-the-Air OFDM Transmission with PlutoSDR

## Introduction

This repository contains a Python implementation of an Over-the-Air (OTA) Orthogonal Frequency Division Multiplexing (OFDM) communication system utilizing the Analog Devices ADALM-PLUTO (PlutoSDR) device. OFDM is a popular digital modulation technique used in various wireless communication standards due to its robustness against multipath propagation and achieving high spectral efficiency. This example aims to demonstrate the fundamental concepts of OFDM transmission and reception in a real-world scenario using the PlutoSDR. Also 3GPP CDL-C channel model is implemented for testing without SDR radio.

The provided functions and classes can facilitate an evaluation of OFDM systems in a real-world transmission scenario. The performance graph depicted below, generated using the integrated libraries in conjunction with the 20-ofdm-performance-testing.ipynb notebook, serves as an illustrative example. Examination of the graph reveals that the current implementation achieves an approximate Bit Error Rate (BER) of 10% at a Signal-to-Interference-plus-Noise Ratio (SINR) of 21 dB. This empirical data is indicative of the system's performance characteristics under the specified test conditions.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/Performance.png)

## Prerequisites

Before running the example, ensure that you have the following prerequisites installed and configured:

Python 3.10 or later, numPy, scipy, itertools, matplotlib, libiio
PlutoSDR drivers and utilities

The PlutoSDR device should also be properly connected to your computer and accessible through the network or USB interface.

## Installation
```
git clone https://github.com/rikluost/pluto
```

## Usage

```
cd pluto
open the jupyter notebook
```

### notebooks

- `00-setup-pluto-MacOS.ipynb` hints on how to set up plutoSdr on Mac M1
- `05-SINR_curve.ipynb` helper for finding appropriate rx_gain and rx_gain values
- `10-ofdm-example-func.ipynb` end to example of how OFDM transmission and reception works
- `20-ofdm-performance-testing.ipynb` loop for system testing in varying radio conditions

### libraries

- `SDR_Pluto.py` class for handling SDR functionality
- `OFDM_SDR_Functions.py` OFDM building block functions, channel simulation
- `config.py` OFDM related configuration parameters are stored here

## Brief intro to OFDM workflow

Under construction

### TTI Mask Creation
Creation of a TTI (Transmission Time Interval) mask in an OFDM system involves specifying the time-domain constraints for the transmission of different types of data symbols, e.g. pilot symbols and user data.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/TTImask.png) 

### Data stream creation
When filling a TTI with data, a data stream is created and filled with random data for the purpose of transmission. This data is then stored at the transmitting end to enable comparison with the received data at the other end of the OFDM system. This comparison helps in assessing the integrity of the received data and the performance of the communication system.

### Serial to parallel
The conversion of a data stream from serial to parallel format involves dividing the incoming serial bit stream into multiple sub-streams, which are then simultaneously transmitted on different orthogonal subcarriers.

### Modulation
The process of encoding the parallel data streams onto the different subcarriers by varying their amplitude and phase, according to the data being transmitted.

Example of the constellation of a common QAM modulation scheme:
![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/QAMconstellation.png) 

The TTI mask after filled with modulated symbols:
![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/TTImod.png) 

### Conversion of frequency domain symbols into time domain (FFT)
The parallel data streams, which have been modulated onto different subcarriers, are transformed from the frequency domain back to the time domain using an IFFT (Inverse Fast Fourier Transform).

### CP addition
The Cyclic Prefix (CP) in an OFDM system involves appending a copy of the end of an OFDM symbol to its beginning, mitigating intersymbol interference and maintaining subcarrier orthogonality. This ensures reliable data transmission over multipath channels and simplifies channel equalization at the receiver.

### Transmission of IQ signals (SDR)

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/PSD_TX.png) 

### Reception of IQ signals (SDR)

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/PSD_RX.png) 

### Synchronisation and CP Removal
Synchronisation is done by correlation of the transmitted signal with the received signal.

The added CP from each received OFDM symbol is discarded before processing.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/RXsignal_sync.png) 

### Conversion of time domain IQ symbols into frequency domain (DFT)

The received time-domain signal is converted back into the frequency domain using an DFT. This operation separates the data on the different subcarriers, allowing for demodulation and further processing to retrieve the transmitted data.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/TTI_RX.png) 

### Channel Estimation
Channel estimation involves determining the properties of the transmission channel to adaptively adjust the receiver and mitigate the adverse effects of the radio channel. This typically entails using known pilot symbols or training sequences to estimate the channel's frequency response.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/ChannelEstimate.png) 

### Equalization

Equalization is the process of compensating for the impairments introduced by the transmission channel. In OFDM systems, equalization typically involves manipulating the received signal in the frequency domain, based on channel state information determined by the channel estimation process.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/RXdSymbols.png) 

### Symbol demapping
This is the process of converting the received modulated symbols back into their corresponding bit representations. It involves translating the complex values from the demodulated subcarriers into the original transmitted bits based on the used modulation scheme.

### Data stream recreation
this entails concatenating the demapped bits from each subcarrier, converting the parallel data streams back into a single serial data stream

### Bit Error Rate (BER) calculation
Compare the original transmitted bit stream with the received bit stream, bit by bit and calculate BER.




## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special appreciation goes to Dr. Marc Lichtman for his invaluable contributions to the field of digital signal processing through his educational material available at PySDR (https://pysdr.org/index.html). His work has been a substantial resource, providing thorough explanations and practical insights on implementing signal processing concepts using Python. 

I would also like to extend my gratitude to Max, whose extensive work on digital signal processing is accessible at DSP Illustrations (https://dspillustrations.com/pages/index.html).

Several of the implementation concepts presented herein are derived from the aforementioned literature.

## Disclaimer: Legal Compliance in Radio Transmission

It is imperative to acknowledge that engaging in radio transmission activities, especially over the air, mandates strict adherence to regulatory and licensing requirements. The use of devices like the PlutoSDR for transmitting radio signals must be conducted responsibly and legally.

1. **Obtain Required Licenses**: Before initiating any radio transmission, ensure you have obtained all necessary licenses and permissions from the appropriate regulatory bodies.
2. **Follow Frequency Allocations**: Adhere to the designated frequency bands and power levels to prevent interference with licensed services and other critical communication systems.
3. **Ensure Safety**: Ensure that your transmission activities do not pose any risks to public safety or disrupt other lawful communication services.

Engaging in unauthorized radio transmission can result in severe legal consequences, including fines, legal actions, and the confiscation of equipment.
