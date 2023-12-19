# OFDM-PlutoSDR: Bridging SDR, OFDM and AI for Next-Gen Wireless Communication

## Introduction

This project presents a comprehensive Python implementation of an Over-the-Air (OTA) Orthogonal Frequency Division Multiplexing (OFDM) communication system utilizing the Analog Devices ADALM-Pluto (PlutoSDR) device for 1T1R transmission and reception. The implementation leverages the power of PyTorch, allowing for a seamless integration and experimentation with machine learning techniques for state-of-the-art Physical Layer (PHY) processing. 

In addition to a simple Least Squares (LS) channel estimator and zero forcing (ZF) equalization, a neural netowrk (NN) based receiver is implemented to predict the bits from the IQ signals right after the Discreet Fourier Transformation (DFT) block. A functionality for creating training datasets, for training models, and testing different setups over radio interface or simulated radio channel are part of the project. 

A NN-based receiver model is trained and compared against the LS/ZF receiver. The example NN-receiver follows the DeepRX (Honkala et.al., 2021) concept, but the model architecture is much simplified and lighter. The implementation demonstrates that the NN-based receiver outperforms a more traditional receiver based on least squares and zero-forcing techniques.

## Table of contents

<!-- TOC -->

- [OFDM-PlutoSDR: Bridging SDR, OFDM and AI for Next-Gen Wireless Communication](#ofdm-plutosdr-bridging-sdr-ofdm-and-ai-for-next-gen-wireless-communication)
    - [Introduction](#introduction)
        - [OFDM building blocks as python functions](#ofdm-building-blocks-as-python-functions)
        - [End to end example of OFDM system](#end-to-end-example-of-ofdm-system)
        - [Training data generator for ML-based receivers](#training-data-generator-for-ml-based-receivers)
        - [Training a ML-based receiver](#training-a-ml-based-receiver)
        - [Performance comparison of ZF/LS and NN-based receivers with testset and over the air](#performance-comparison-of-zfls-and-nn-based-receivers-with-testset-and-over-the-air)
        - [Performance comparison of ZF/LS and NN-based receivers over the air with PlutoSDR](#performance-comparison-of-zfls-and-nn-based-receivers-over-the-air-with-plutosdr)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
        - [Notebooks and other files](#notebooks-and-other-files)
            - [Jupyter notebooks](#jupyter-notebooks)
            - [Functions, configuration, models](#functions-configuration-models)
    - [Brief intro to OFDM workflow](#brief-intro-to-ofdm-workflow)
        - [TTI Mask Creation](#tti-mask-creation)
        - [Data stream creation](#data-stream-creation)
        - [Serial to parallel - codewords](#serial-to-parallel---codewords)
        - [Modulation](#modulation)
        - [Conversion of frequency domain symbols into time domain FFT](#conversion-of-frequency-domain-symbols-into-time-domain-fft)
        - [CP addition](#cp-addition)
        - [Radio Channel](#radio-channel)
            - [Transmission of IQ signals SDR](#transmission-of-iq-signals-sdr)
            - [Reception of IQ signals SDR](#reception-of-iq-signals-sdr)
        - [Synchronisation and CP Removal](#synchronisation-and-cp-removal)
        - [Conversion of time domain IQ symbols into frequency domain DFT](#conversion-of-time-domain-iq-symbols-into-frequency-domain-dft)
        - [Channel Estimation](#channel-estimation)
        - [Equalization](#equalization)
        - [Symbol demapping](#symbol-demapping)
        - [Data stream recreation](#data-stream-recreation)
        - [Bit Error Rate BER calculation](#bit-error-rate-ber-calculation)
    - [NN-based receiver](#nn-based-receiver)
        - [Training data generation](#training-data-generation)
        - [Training a NN-based receiver model](#training-a-nn-based-receiver-model)
        - [Comparing performance on testset](#comparing-performance-on-testset)
        - [Comparing performance over the air with SDR radio](#comparing-performance-over-the-air-with-sdr-radio)
    - [License](#license)
    - [References](#references)
    - [Disclaimer: Legal Compliance in Radio Transmission](#disclaimer-legal-compliance-in-radio-transmission)

<!-- /TOC -->

## OFDM building blocks as python functions

The functions and classes in `OFDM_SDR_Functions_torch.py` can facilitate the build demo set ups either with a true radio interface or with simulated radio channel. 

### End to end example of OFDM system 
The example in `10-ofdm-example-func.ipynb` aims to demonstrate the fundamental concepts of OFDM transmission and reception using the PlutoSDR. The notebook goes through the whole OFDM process, implements a simple LS channel estimator and ZF equalizer. The performance graph depicted below, generated using the integrated libraries in conjunction with the `20-ofdm-performance-testing.ipynb` notebook, serves as an illustrative example on what can be achieved. The empirically collected data is indicative of the system's performance characteristics under the test conditions in authors radio shack.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/Performance.png)
Fig 1. Performance curve of an OFDM system as determined with empirical over the air measurements with SDR radio.

### Training data generator for ML-based receivers
For building ML into the OFDM technology, `30-NN-receiver-dataset-creator.ipynb` provides an example on how to create torch datasets for training e.g. an NN based receiver, storing received pilot signals and modulated data as inputs, and associated original bitstream as lables. Simplified 3GPP CDL-C channel model is implemented for faster creation of training datasets, without using an SDR radio.

### Training a ML-based receiver
 `40-training-NN-based-receiver.ipynb` contains an example on how to build an train a NN-based receiver. Example NN-based receiver is much simplified version of the DeepRX Honkala et. al describes in (Honkala et. al. 2021). The receiver is fully convolutional, ResNet-style receiver utilising residual blocks with skip connection. Simple training loop is used for training the the receiver.

### Performance comparison of ZF/LS and NN-based receivers with testset and over the air

In `50-compare-ZF-LS-with-NN-based-RX-testset.ipynb` the performance of trained NN-based receiver and LS/ZF receiver is compared by utilising a testset created earlier. In `60-compare-ZF-LS-with-NN-based-RX-PlutoSDR.ipynb` the performance is compared with a real air interface.

### Performance comparison of ZF/LS and NN-based receivers over the air with PlutoSDR


## Prerequisites

Python > 3.10, torch, numPy, matplotlib, libiio, pylib-iio

## Installation

```
git clone https://github.com/rikluost/pluto # for the latest development version
```

### Notebooks and other files

#### Jupyter notebooks

- `00-setup-pluto-MacOS.ipynb` hints on how to set up s plutoSdr on Mac M1
- `05-SINR_curve.ipynb` helper for finding appropriate rx_gain and tx_gain values for your SDR set up
- `10-ofdm-example-func.ipynb` end to end example of an OFDM system
- `20-ofdm-performance-testing.ipynb` loop for system testing in varying radio conditions
- `30-NN-receiver-dataset-creator.ipynb` torch dataset creation tool for training NN-based receivers
- `40-training-NN-based-receiver.ipynb` training a NN-based receiver.
- `50-compare-ZF-LS-with-NN-based-RX-testset.ipynb` performance comparison between simple OFDM receiver and NN based receiver over the testset
- `60-compare-ZF-LS-with-NN-based-RX-PlutoSDR.ipynb` performance comparison between simple OFDM receiver and NN based receiver over a real radio interface using PlutoSDR

#### Functions, configuration, models

- `SDR_Pluto.py` class for handling SDR functionality
- `OFDM_SDR_Functions_torch.py` OFDM building block functions, channel simulation
- `config.py` OFDM related configuration parameters are stored here. Also the training data structure class is located here
- `models_local.py` the NN-based receiver as pyTorch model architecture is placed here
- `data/` folder contains the actual models and weights, training datasets and testset are store under 
- `pics/` any graphs for this documentation are stored here

## Brief intro to OFDM workflow

The following workflow covers TTI Mask Creation, Data Stream Creation, Modulation, CP Addition, Radio Channel Simulation/Over The Air Transmission, Synchronization, Channel Estimation, Equalization, Symbol Demapping, Data Stream Recreation, and Bit Error Rate (BER) Calculation.

### TTI Mask Creation

Creation of a TTI (Transmission Time Interval) mask in an OFDM system involves specifying the constraints for the transmission of different types of data symbols, e.g. pilot symbols, DC, and user data.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/TTImask.png) 
Fig 2. TTI mask.

### Data stream creation

A data stream is created for filling the TTI PDSCH resources with random data for the purpose of transmission. This data is then in the following stagtes used for modulating the TTI, and it is also stored to enable comparison with the received data for Bit Error Rate (BER) calculation at the receiver. This comparison helps in assessing the integrity of the received data and the performance of the communication system.

### Serial to parallel - codewords

The conversion of a data stream from serial to parallel format involves dividing the incoming serial bit stream into groups (codewords) defined by the modulation order. 

### Modulation

The process of encoding (= mapping) the bits onto the different subcarriers by varying their amplitude and phase, according to the data being transmitted.

Example of the constellation of a common QAM modulation scheme:
![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/QAMconstellation.png) 
Fig 3. Constellation.

The TTI mask after filled with modulated symbols:
![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/TTImod.png) 
Fig 4. Modulated TTI.

### Conversion of frequency domain symbols into time domain (FFT)
The parallel data streams, which have been modulated onto different subcarriers, are transformed from the frequency domain back to the time domain using an IFFT (Inverse Fast Fourier Transform).

### CP addition
The Cyclic Prefix (CP) in an OFDM system involves appending a copy of the end of an OFDM symbol to its beginning, mitigating intersymbol interference and maintaining subcarrier orthogonality. This ensures reliable data transmission over multipath channels and simplifies channel equalization and synchronisation at the receiver.

### Radio Channel

A basic linear model for a channel is:

$\mathbf{y} = \mathbf{Hx} + \mathbf{n}$

where:

- $\mathbf{y}$ is the received signal vector
- $\mathbf{H}$ is the channel matrix
- $\mathbf{x}$ is the transmitted signal vector
- $\mathbf{n}$ is the noise vector

$\mathbf{H}$, the radio channel matrix, can be constructed here either by transmitting the signal $\mathbf{x}$ with an SDR transmitter, and then receiving the signal with an SDR receiver, or by using a simplified 3GPP CDL-C radio channel model. Noise, in case of SDR is naturally injected into the signal, while in case of using CDL-C, it is added separately.

#### Transmission of IQ signals (SDR)

The signals can be passed through the SDR transmitter, or through the simulated radio channel. The below graph shows the power spectral density of transmitted signal $\mathbf{x}$:

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/PSD_TX.png) 
Fig 5. Power spectral density of the signal fed to the SDR transmitter.

#### Reception of IQ signals (SDR)

The signal can be received by the SDR receiver, which translates it to time domain IQ signals, or alternatively by using the output of simulated radio channel. The below graph shows the power spectral density of received signal $\mathbf{y}$:

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/PSD_RX.png) 
Fig 6. Power spectral density of the signal received from the SDR receiver.

### Synchronisation and CP Removal
Synchronisation is done by correlation of the transmitted signal with the received signal. The unmodulated samples between the TTI's are there simply to allow measuring noise level for the SINR estimation. The earlier added CP from each received OFDM symbol is discarded before processing.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/RXsignal_sync.png) 
Fig 7. Synchronisation by correlation.

### Conversion of time domain IQ symbols into frequency domain (DFT)

The received time-domain signal is converted back into the frequency domain using DFT. This operation separates the data on the different subcarriers, allowing for demodulation and further processing to retrieve the pilot symbols and transmitted data.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/TTI_RX.png) 
Fig 8. Received TTI before the equalisation.

### Channel Estimation

Channel estimation involves determining the properties of the transmission channel to adaptively adjust the receiver and mitigate the adverse effects of the radio channel. This typically entails using known pilot symbols or training sequences to estimate the channel's frequency response. The output of the channel estimation is $\mathbf{H}$, the estimated channel matrix.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/ChannelEstimate.png) 
Fig 9. Absolute value of the received pilot symbols.

### Equalization

Equalization is the process of compensating for the impairments introduced by the transmission channel by dividing the received signal $\mathbf{y}$ by the channel estimate $\mathbf{H}$. 

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/RXdSymbols.png) 
Fig 10. Received symbols of the PDCCH symbols of one TTI after signal equalization.

### Symbol demapping

This is the process of converting the received modulated symbols back into their corresponding bit representations. It involves translating the complex values from the demodulated subcarriers into the original transmitted bits based on the used modulation scheme.

### Data stream recreation

The data stream is entailed by concatenating the demapped bits from each subcarrier and symbol, converting the parallel data streams back into a single serial data stream

### Bit Error Rate (BER) calculation

Compare the original transmitted bit stream with the received bit stream, bit by bit and calculate BER.

## NN-based receiver

### Training data generation

The dataset creation example `30-NN-receiver-dataset-creator.ipynb` generates random data and OFDM TTIs modulated with it. The OFDM signals are sent over the simulated radio channel (simplified 3GPP CDL-C here). It stores the OFDM data after reception, cyclic prefic removal and DFT. The created dataset contains received OFDM symbols and pilot symbols as input, and original bits used for the modulation as labels. There are 6 bits per label, so 64QAM was used here.

### Training a NN-based receiver model

To demonstrate the setup, a dataset consisting of 20,000 samples with randomised data and radio channel was used for training the model as defined in `models_local.py`. The dataset could be much larger though - the training took only about 10minutes and was interrupted as the performance did not seem to improve any longer, as can be seen in Fig 10 below. An example of the training process can be found in `40-training-NN-based-receiver.ipynb`. As this is only intented to be a demonstration of the setup, the model architecture and any hyperparameters leave room for further optimisation.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/training_loss.png) 
Fig 10. The model performance during the training process

Not only the training loss and validation loss were observed, but also the BER of the system with validation data was monitored, as seen in Fig 11.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/training_ber.png) 
Fig 11. Bit Error Rate (BER) during the training process

### Comparing performance on testset

Notebook `50-compare-ZF-LS-with-NN-based-RX-testset.ipynb` loads the model and its weights, and the test dataset saved earlier. It implements both the NN-based receiver as well as the LS/ZF based receiver used in the OFDM example notebook. The perormance on the testset is compared and the distributions of the BERs are shown in Fig 12. The NN-based receiver on this dataset clearly outperforms the one of the LS/ZF based receiver.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics/BER_distribution_testset.png) 

Fig 12. Bit Error Rate (BER) distribution on the testset, comparing NN-based receiver to ZF/LS receiver

### Comparing performance over the air with SDR radio

IN WORKS

`60-compare-ZF-LS-with-NN-based-RX-PlutoSDR.ipynb`


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
