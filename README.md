# Over-the-Air OFDM Transmission with PlutoSDR, PyTorch implementation

## Introduction

This repository contains a Python implementation of an Over-the-Air (OTA) Orthogonal Frequency Division Multiplexing (OFDM) communication system utilizing the Analog Devices ADALM-PLUTO (PlutoSDR) device where PyTorch is used as much as possible to allow easy experimentation with AIML for PHY-processing. Neural netowrk-based receiver is planned. This follows the largerly numpy-based implementation in https://github.com/rikluost/ofdm-plutosdr-numpy, which contains more documentation on basic consepts. Please note this is work in progress and in early stage. 

OFDM is a popular digital modulation technique used in various wireless communication standards due to its robustness against multipath propagation and achieving high spectral efficiency. The example in `10-ofdm-example-func.ipynb` aims to demonstrate the fundamental concepts of OFDM transmission and reception in a real-world scenario using the PlutoSDR while utilising PyTorch where possible. Not all functions can be based on PyTorch as the SDR libraries require NumPy input, and also output is in NumPy format. 3GPP CDL-C channel model is implemented for testing or faster training of models without using SDR radio.

The provided functions and classes can facilitate an evaluation of OFDM systems in a real-world transmission scenario. The performance graph depicted below, generated using the integrated libraries in conjunction with the 20-ofdm-performance-testing.ipynb notebook, serves as an illustrative example. Examination of the graph reveals that the current implementation achieves an approximate Bit Error Rate (BER) of 10% at a Signal-to-Interference-plus-Noise Ratio (SINR) of 21 dB. This empirical data is indicative of the system's performance characteristics under the specified test conditions.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/Performance.png)

## Prerequisites

Before running the example, ensure that you have the following prerequisites installed and configured:

Python 3.10 or later, numPy, scipy, itertools, matplotlib, libiio, torch
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
- `OFDM_SDR_Functions_torch.py` OFDM building block functions, channel simulation
- `config.py` OFDM related configuration parameters are stored here
- `ML_lib.py` AIML models and supporting functions, e.g. channel estimation, demodulation and equalizer can be placed here.

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

### Radio Channel

#### Radio Channel

A basic linear model for a channel is:

$\mathbf{y} = \mathbf{Hx} + \mathbf{n}$

where:

- $\mathbf{y}$ is the received signal vector
- $\mathbf{H}$ is the channel matrix
- $\mathbf{x}$ is the transmitted signal vector
- $\mathbf{n}$ is the noise vector

$\mathbf{H}$, the radio channel matrix, can be constructed here either by transmitting the signal $\mathbf{x}$ with an SDR transmitter, and then receiving the signal with an SDR receiver, or by using a simplified 3GPP CDL-C radio channel model. Noise, in case of SDR is naturally injected into the signal, while in case of using CDL-C, it is added separately.

#### Transmission of IQ signals (SDR)

Below shows the power spectral density of transmitted signal \(\mathbf{x}\):

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/PSD_TX.png) 

#### Reception of IQ signals (SDR)

Below shows the power spectral density of received signal \(\mathbf{y}\):

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/PSD_RX.png) 

### Synchronisation and CP Removal
Synchronisation is done by correlation of the transmitted signal with the received signal.

The added CP from each received OFDM symbol is discarded before processing.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/RXsignal_sync.png) 

### Conversion of time domain IQ symbols into frequency domain (DFT)

The received time-domain signal is converted back into the frequency domain using an DFT. This operation separates the data on the different subcarriers, allowing for demodulation and further processing to retrieve the transmitted data.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/TTI_RX.png) 

### Channel Estimation
Channel estimation involves determining the properties of the transmission channel to adaptively adjust the receiver and mitigate the adverse effects of the radio channel. This typically entails using known pilot symbols or training sequences to estimate the channel's frequency response. The output of the channel estimation is \(\mathbf{H}\), the estimated channel matrix.

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

## References

“PySDR: A Guide to SDR and DSP Using Python — PySDR: A Guide to SDR and DSP Using Python 0.1 Documentation.” Accessed November 15, 2021. https://pysdr.org/index.html.

“DSPIllustrations.Com.” Accessed November 15, 2021. https://dspillustrations.com/pages/index.html.

Honkala, Mikko, Dani Korpi, and Janne M. J. Huttunen. “DeepRx: Fully Convolutional Deep Learning Receiver.” IEEE Transactions on Wireless Communications 20, no. 6 (June 2021): 3925–40. https://doi.org/10.1109/TWC.2021.3054520.

Paszke, Adam, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, et al. “PyTorch: An Imperative Style, High-Performance Deep Learning Library.” In Advances in Neural Information Processing Systems 32, 8024–35. Curran Associates, Inc., 2019. http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.

“ADALM-PLUTO Evaluation Board | Analog Devices.” Accessed November 15, 2021. https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html#eb-overview.



## Disclaimer: Legal Compliance in Radio Transmission

It is imperative to acknowledge that engaging in radio transmission activities, especially over the air, mandates strict adherence to regulatory and licensing requirements. The use of devices like the PlutoSDR for transmitting radio signals must be conducted responsibly and legally.

1. **Obtain Required Licenses**: Before initiating any radio transmission, ensure you have obtained all necessary licenses and permissions from the appropriate regulatory bodies.
2. **Follow Frequency Allocations**: Adhere to the designated frequency bands and power levels to prevent interference with licensed services and other critical communication systems.
3. **Ensure Safety**: Ensure that your transmission activities do not pose any risks to public safety or disrupt other lawful communication services.

Engaging in unauthorized radio transmission can result in severe legal consequences, including fines, legal actions, and the confiscation of equipment.
