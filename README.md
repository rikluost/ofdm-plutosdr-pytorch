# Over-the-Air OFDM Transmission with PlutoSDR, PyTorch implementation

## Introduction

This repository contains a Python implementation of an Over-the-Air (OTA) Orthogonal Frequency Division Multiplexing (OFDM) communication system utilizing the Analog Devices ADALM-PLUTO (PlutoSDR) device where PyTorch is used as much as possible to allow easy experimentation with AIML for PHY-processing, and a neural netowrk-based receiver is planned in near future. This repository follows the largerly numpy-based implementation in https://github.com/rikluost/ofdm-plutosdr-numpy. Please note this is work in progress.

OFDM is a popular digital modulation technique used in various wireless communication standards due to its robustness against multipath propagation and achieving high spectral efficiency. The example in `10-ofdm-example-func.ipynb` aims to demonstrate the fundamental concepts of OFDM transmission and reception using the PlutoSDR. 3GPP CDL-C channel model is implemented for testing or faster training of models without using an SDR radio.

The provided functions and classes can facilitate an evaluation of OFDM systems, and build over the air demo set ups. The performance graph depicted below, generated using the integrated libraries in conjunction with the `20-ofdm-performance-testing.ipynb` notebook, serves as an illustrative example. 

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/Performance.png)

Fig 1. Empirical performance curve of OFDM system as measured over the air transmissions.

Examination of the example in Fig1 reveals that the current implementation achieves an approximate Bit Error Rate (BER) of 10% at a Signal-to-Interference-plus-Noise Ratio (SINR) of 21 dB. This empirical data is indicative of the system's performance characteristics under the specified test conditions.

## Prerequisites

Before running the example, ensure that you have the following prerequisites installed and configured:

Python 3.10 or later, torch, numPy, scipy, itertools, matplotlib, libiio, torch
PlutoSDR drivers and utilities

The PlutoSDR device should also be properly connected to your computer and accessible through the network or USB interface.

## Installation
```
git clone https://github.com/rikluost/pluto
```

## Usage

```
cd pluto
open the jupyter notebooks, see below.
```

### notebooks

- `00-setup-pluto-MacOS.ipynb` hints on how to set up plutoSdr on Mac M1
- `05-SINR_curve.ipynb` helper for finding appropriate rx_gain and tx_gain values for your SDR set up
- `10-ofdm-example-func.ipynb` end to example of how OFDM transmission and reception works
- `20-ofdm-performance-testing.ipynb` loop for system testing in varying radio conditions

### libraries

- `SDR_Pluto.py` class for handling SDR functionality
- `OFDM_SDR_Functions_torch.py` OFDM building block functions, channel simulation
- `config.py` OFDM related configuration parameters are stored here
- `ML_lib.py` AIML models and supporting functions, e.g. channel estimation, demodulation and equalizer can be placed here.

## Brief intro to OFDM workflow

### TTI Mask Creation

Creation of a TTI (Transmission Time Interval) mask in an OFDM system involves specifying the allocating the constraints for the transmission of different types of data symbols, e.g. pilot symbols, DC, and user data.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/TTImask.png) 

Fig 2. TTI mask.

### Data stream creation

A data stream is created for filling the TTI PDSCH resources with random data for the purpose of transmission. This data is then in the following stagtes used for modulating the TTI, and it is also stored to enable comparison with the received data for Bit Error Rate (BER) calculation at the receiver. This comparison helps in assessing the integrity of the received data and the performance of the communication system.

### Serial to parallel - codewords

The conversion of a data stream from serial to parallel format involves dividing the incoming serial bit stream into groups defined by the modulation order. Each group is called a codeword. This is done in order to be able to map the bits in the codeword into the symbols in the following stage.

### Modulation

The process of encoding the bits onto the different subcarriers by varying their amplitude and phase, according to the data being transmitted.

Example of the constellation of a common QAM modulation scheme:
![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/QAMconstellation.png) 

Fig 3. Constellation.

The TTI mask after filled with modulated symbols:
![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/TTImod.png) 

Fig 4. Modulated TTI.

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

Below shows the power spectral density of transmitted signal $\mathbf{x}$:

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/PSD_TX.png) 

Fig 5. Power spectral density of the signal fed to the SDR transmitter.

#### Reception of IQ signals (SDR)

Below shows the power spectral density of received signal $\mathbf{y}$:

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/PSD_RX.png) 

Fig 6. Power spectral density of the signal received from the SDR receiver.

### Synchronisation and CP Removal
Synchronisation is done by correlation of the transmitted signal with the received signal. The unmodulated 500 samples between the TTI's are simply to measure noise level for the SINR estimation. The earlier added CP from each received OFDM symbol is discarded before processing.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/RXsignal_sync.png) 

Fig 7. Synchronisation by correlation.

### Conversion of time domain IQ symbols into frequency domain (DFT)

The received time-domain signal is converted back into the frequency domain using DFT. This operation separates the data on the different subcarriers, allowing for demodulation and further processing to retrieve the pilot symbols and transmitted data.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/TTI_RX.png) 

Fig 8. Received TTI before the equalisation.

### Channel Estimation

Channel estimation involves determining the properties of the transmission channel to adaptively adjust the receiver and mitigate the adverse effects of the radio channel. This typically entails using known pilot symbols or training sequences to estimate the channel's frequency response. The output of the channel estimation is $\mathbf{H}$, the estimated channel matrix.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/ChannelEstimate.png) 

Fig 9. Absolute value of the received pilot symbols.

### Equalization

Equalization is the process of compensating for the impairments introduced by the transmission channel by dividing the received signal $\mathbf{y}$ by the channel estimate $\mathbf{H}$. More advanced methods do exist.

![alt text](https://github.com/rikluost/ofdm-plutosdr-numpy/blob/main/pics/RXdSymbols.png) 

Fig 10. Received symbols of the PDCCH symbols of one TTI after signal equalization.

### Symbol demapping

This is the process of converting the received modulated symbols back into their corresponding bit representations. It involves translating the complex values from the demodulated subcarriers into the original transmitted bits based on the used modulation scheme.

### Data stream recreation

this entails concatenating the demapped bits from each subcarrier, converting the parallel data streams back into a single serial data stream

### Bit Error Rate (BER) calculation

Compare the original transmitted bit stream with the received bit stream, bit by bit and calculate BER.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

Chang, Robert W. "Synthesis of Band-Limited Orthogonal Signals for Multichannel Data Transmission." Bell System Technical Journal 45 (December 1966): 1775-1796.

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
