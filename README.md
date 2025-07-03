# OFDM-PlutoSDR: Testing SDR, OFDM and AI receivers for Next-Gen Wireless Communication

## Overview

This project delivers an end-to-end Python implementation of an Over-the-Air (OTA) Orthogonal Frequency Division Multiplexing (OFDM) communication system using the Analog Devices ADALM-Pluto (PlutoSDR) for single-antenna (1T1R SISO) transmission and reception. Built on PyTorch, the framework enables seamless integration and experimentation with modern machine learning techniques for next-generation Physical Layer (PHY) processing.

In addition to standard Least Squares (LS) channel estimation and Zero-Forcing (ZF) equalization, the framework introduces a neural network (NN)-based receiver capable of predicting transmitted bits directly from IQ samples after the Discrete Fourier Transform (DFT) at the receiver. The platform supports comprehensive dataset generation, model training, and performance evaluation over both simulated channels and real-world radio hardware.

A central contribution of this project is the ability to rigorously compare NN-based receivers—trained on simulated channel data and evaluated over-the-air or in simulation—against classical LS/ZF methods. The neural architecture, inspired by the DeepRx design (Honkala et al., 2021), adopts a streamlined structure to demonstrate that data-driven approaches can match or surpass traditional PHY receiver performance in practical SDR scenarios.

## Brief literature review

OFDM, originally formalized for multichannel data transmission by Chang [1], is foundational to contemporary wireless systems. Goldsmith [3] provides core theory on wireless channels, enabling practical OFDM deployment. The ADALM-Pluto board [9] have lowered the barrier for prototyping and over-the-air experimentation. Web sites "PySDR: A Guide to SDR and DSP using Python" [5] and "DSPIllustrations" [6] provide insights into use of SDR and signal processing concepts using python.

Recent research applies deep learning to the PHY. Goodfellow et al. [2] and Paszke et al. [8] presenting the theoretical and software foundations. Honkala et al. [7] show fully convolutional deep receivers (DeepRx) surpassing traditional methods in wireless environments, while Ju et al. [4] compare deep learning with iterative algorithms for joint channel estimation and detection in OFDM. 

## Python libraries and Jupyter noteboks

### Prerequisites

Python > 3.10, PyTorch, numPy, matplotlib, libiio, pylib-iio, scipy, pandas, jupyter

TODO: Some additional libraries may be required - update.

ADALM-Pluto SDR required for over-the-air experiments.

### Installation

```
git clone https://github.com/rikluost/pluto # for the latest development version
```

### Files and folders

#### Jupyter Notebooks

A series of Jupyter notebooks is provided to facilitate reproducible research, from data generation to training and benchmarking receiver architectures.

- `10-ofdm-example-func.ipynb`: Implements a complete OFDM transmitter and receiver chain. Enables side-by-side comparison of classical LS/ZF equalization with DeepRx-inspired neural network (NN) receivers, processing one Transmission Time Interval (TTI) at a time.
- `30-dataset-creator.ipynb`: Generates datasets for receiver training and evaluation. Supports dataset creation using either SDR-based over-the-air captures or simulated channel models.
- `40-receiver-training.ipynb`: Provides tools and examples for training the implemented fully convolutional neural network (FCNN)-based receiver on the generated datasets.
- `50-test-receivers.ipynb`: Evaluates and benchmarks the performance of FCNN-based neural network receivers against classical LS/ZF approaches on both simulated and real-world data.

#### Libraries, and configuration files
The following core Python modules and configuration files provide the backbone for SDR operation, OFDM processing, and neural network integration:
- `SDR_Pluto.py`: Utilities for interfacing with the ADALM-Pluto SDR, including functions for signal transmission, reception, and device configuration.
- `OFDM_SDR_Functions_torch.py`: A comprehensive suite of functions for OFDM modulation/demodulation, channel simulation, and physical layer (PHY) processing, leveraging PyTorch.
- `config.py`: Centralized configuration settings for the OFDM system and experiments, including definition of PyTorch dataset classes.
- `models_local.py`: PyTorch model definitions for NN-based receiver architectures, supporting modular development and rapid experimentation with different neural network designs.

#### Key folders
- `data/`: Trained models, weights, and datasets.
- `pics/`, `pics_doc/`: Saved plots for analysis and documentation.

## Introduction to the implemented OFDM workflow

The workflow covers TTI Mask Creation with Pilot Symbols, Data Stream Creation, Modulation, CP Addition, Radio Channel Simulation / Over The Air Transmission, Synchronization, Channel Estimation, Equalization, Symbol Demapping, and Bit Error Rate (BER) Calculation.

Most processing blocks come with illustrations, for creating these, the following parameters were used:

Radio channel:
- frequency: 3500MHz
- delay spread: 0 - 300us, 
- Channel: Tapped Delay Line (TDL) with up to 3 taps (resolution set by sampling frequency)
- velocity: up to 30m/s, 
- Doppler spread : Jakes with 16 sinusoids per tap.

OFDM:
- FFT size: 128, 100 modulated subcarriers
- Subcarrier spacing: 15kHz
- Cyclic prefix: 6
- Modulation: 16-QAM


### TTI Mask

A Transmission Time Interval (TTI) mask defines resource allocation in each OFDM symbol, specifying positions for pilot symbols, DC, guard bands, and user data. This implementation uses pilots in a single OFDM symbol per TTI for simplicity, though extension to multiple symbols is straightforward

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/TTImask.png) 

Fig 1. TTI mask: purple—unmodulated guard bands; yellow—DC; green—pilots; blue—user data.

### Data stream creation

A random data stream is generated to populate the TTI’s PDSCH resources. This stream is used for OFDM symbol modulation and is retained for post-reception comparison, enabling Bit Error Rate (BER) calculation after demodulation and demapping.

### Serial to parallel

The conversion of a data stream from serial to parallel format involves dividing the incoming serial bit stream into groups where the size is defined by the modulation order and FFT-size. 

### Modulation

Bits are mapped onto OFDM subcarriers by modulating amplitude and phase using M-QAM, according to the data stream. This enables simultaneous transmission of multiple symbols across subcarriers. Figures 2 and 3 illustrate example cases of 16-QAM constellation and fully populated TTI where the colors are defined by the magnitude of each complex vector.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/QAMconstellation.png) 

Fig 2. Example 16-QAM constellation.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/TTImod.png) 

Fig 3. Modulated TTI, color-coded by magnitude of each complex subcarrier.

### Conversion of frequency domain symbols into time domain (IFFT)
The modulated parallel data streams are transformed from the frequency domain to the time domain using the Inverse Fast Fourier Transform (IFFT), preparing the OFDM symbols for transmission.

### CP addition
A cyclic prefix, formed by copying the end of each OFDM symbol to its beginning, is appended to mitigate intersymbol interference. This preserves subcarrier orthogonality, facilitating reliable transmission and simplifying equalization and synchronization in multipath channels.

### Radio Channel

A basic linear model for a channel is:

$\mathbf{y} = \mathbf{Hx} + \mathbf{n}$

where:

- $\mathbf{y}$ is the received signal vector
- $\mathbf{H}$ is the channel matrix
- $\mathbf{x}$ is the transmitted signal vector
- $\mathbf{n}$ is the noise vector

The channel matrix $\mathbf{H}$ can be obtained either by transmitting $\mathbf{x}$ over-the-air with SDR hardware and capturing the received signal, or by applying a randomized channel model in simulation. In SDR-based experiments, noise is naturally introduced by the physical environment; in simulations, noise is explicitly added to the received signal.

#### Transmission of IQ signals (SDR)

The generated IQ signals can be transmitted either via the SDR hardware or through a simulated radio channel. Figure 4 illustrates the power spectral density of a transmitted signal $\mathbf{x}$:

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/PSD_TX.png)

Fig 4. Power spectral density of the transmitted OFDM signal.

#### Reception of IQ signals (SDR)

The received signal can be captured by the SDR hardware and converted to time-domain IQ samples, or obtained directly from the simulated radio channel output. Figure 5 shows the power spectral density of a received signal $\mathbf{y}$:

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/PSD_RX.png) 

Fig 5. Power spectral density of the received signal.

### Synchronization
Due to the absence of hardware timestamping on PlutoSDR, this implementation employs cyclic transmissions and time-domain synchronization using cross-correlation. Since both transmitter and receiver share the same physical clock, frequency correction is unnecessary. The point of maximum correlation identifies the start of the first OFDM symbol. Figure 6 illustrates the correlation between transmitted and received signals at the symbol level.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/corr.png) 

Fig 6. Symbol-level cross-correlation for synchronization.

### CP Removal

Unmodulated samples inserted between TTIs during cyclic transmission enable noise level measurement for SINR estimation. After synchronization, the cyclic prefix is removed from each received OFDM symbol before further processing. Figure 7 illustrates CP removal using a mask.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/RXsignal_sync.png) 

Fig 7. CP removal after synchronization using a mask.

### Conversion of time domain IQ symbols into frequency domain (DFT)

The synchronized, time-domain IQ signal is transformed back into the frequency domain using the Discrete Fourier Transform (DFT). This operation separates the subcarriers, enabling extraction of pilot symbols and user data for further processing and demodulation.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/TTI_RX.png) 

Fig 8. Received TTI after synchronization, before equalization.

### Channel Estimation

For conventional receivers, channel estimation is performed using known pilot symbols to estimate the channel’s frequency response and mitigate the effects of the radio channel. The result is the estimated channel matrix, $\mathbf{H}$. In this implementation, only a single OFDM symbol is allocated for pilot signals, which is suboptimal for mobile scenarios where the channel can vary rapidly; modern systems (e.g., 4G/5G) typically use two or more pilot symbols in time. As shown later, the NN-based receiver demonstrates improved robustness over LS/ZF methods in mobile environments, even with a single pilot OFDM symbol.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/ChannelEstimateAbs.png) 

Fig 9. Absolute values of received pilot symbols and their interpolations.

### Equalization

Equalization compensates for channel-induced distortions by dividing the received signal $\mathbf{y}$ by the channel estimate $\mathbf{H}$. In this implementation, a Zero Forcing (ZF) equalizer is used.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/RXdSymbols.png) 

Fig 10.  Constellation of PDCCH symbols from one TTI, before and after equalization.

### Symbol demapping

Symbol demapping translates the received, equalized constellation points back into their original bit representations according to the modulation scheme, recovering the transmitted data bits from the demodulated subcarriers.

### Data stream recreation

The final data stream is reconstructed by concatenating the demapped bits from all subcarriers and OFDM symbols, converting the parallel bit streams back into a single serial data stream.

### Bit Error Rate (BER) calculation

The BER is computed by comparing the original transmitted bit stream with the received bit stream on a bit-by-bit basis, quantifying the ratio of erroneous bits to the total number of transmitted bits.

## NN-based receiver

### Training data generation

The notebook `30-NN-receiver-dataset-creator.ipynb` generates random data and modulates it onto OFDM TTIs, which are transmitted through a simulated radio channel. After reception, cyclic prefix removal, and DFT, the processed OFDM data are stored. Each dataset entry includes the received OFDM symbols and pilot symbols as input, with the original modulation bits as labels. In this example, 16-QAM modulation is used, yielding 4 bits per label.

### Training a NN-based receiver model

A demonstration training was performed using a dataset of 20,000 samples, each with randomized data and radio channel parameters, as defined in `models_local.py`. While larger datasets can be used, this training was manually stopped after approximately 10 hours on an Apple M1-based laptop. Additional training or hyperparameter optimization may further improve results.

Details of the training process are provided in `40-training-NN-based-receiver.ipynb`. In this example, training utilized simulated channel data, while validation employed SDR-measured data (cutting the corners a bit). The model architecture and hyperparameters are intentionally kept simple to illustrate the workflow and may be further refined. Figure 11 shows the NN model’s training performance.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/training_loss.png) 

Fig 11. Model performance during training.

### Comparing performance on testset

The notebook `50-compare-ZF-LS-with-NN-based-RX-testset.ipynb` evaluates and compares the NN-based receiver and the conventional LS/ZF receiver using the trained model and a previously saved test dataset. Both receivers are implemented as described in the OFDM example notebook. Performance is benchmarked in terms of Bit Error Rate (BER) on simulated channel data (3500 MHz, 30 m/s).

Figure 12 shows the BER distribution across 5,000 test TTIs, while Figure 13 presents a detailed comparison for a single, randomly chosen TTI. The results demonstrate a significant performance advantage for the NN-based receiver compared to classical approaches, particularly with a minimal pilot configuration.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/BER_distribution_testset_log_scale.png) 

Fig 12. BER distribution on the test set: NN-based receiver vs. ZF/LS receiver (5,000 TTIs).

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/Performance_LS_nn.png) 

Fig 13. Performance comparison between NN-based and classic receivers for a single TTI.

The observed gains of the NN-based receiver suggest potential for reducing pilot overhead in practical systems. For demonstration, a minimal pilot configuration was selected. Future improvements may include frequency correction for separate SDR devices, support for 2T2R MIMO testing, and enhanced channel modeling.

## Errata

- Channel estimation is currently performed separately for magnitude and phase, which can lead to discontinuities or errors when the phase crosses $2\pi$ boundaries.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

[1] Chang, Robert W. "Synthesis of Band-Limited Orthogonal Signals for Multichannel Data Transmission." Bell System Technical Journal 45 (December 1966): 1775-1796.

[2] Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep Learning. The MIT Press, 2016. http://dx.doi.org/10.5555/3086952

[3] A. Goldsmith, “Wireless Communications,” Cambridge University Press, Cambridge, 2005. http://dx.doi.org/10.1017/CBO9780511841224

[4] Haocheng Ju, Haimiao Zhang, Lin Li, Xiao Li, Bin Dong, "A comparative study of deep learning and iterative algorithms for joint channel estimation and signal detection in OFDM systems", Signal Processing, Volume 223, 2024, 109554, ISSN 0165-1684, https://doi.org/10.1016/j.sigpro.2024.109554.

[5] Mark Lichtman, “PySDR: A Guide to SDR and DSP Using Python — PySDR: A Guide to SDR and DSP Using Python 0.1 Documentation.” Accessed November 15, 2021. https://pysdr.org/index.html.

[6] Maximilian Matthe, “DSPIllustrations.Com.” Accessed November 15, 2021. https://dspillustrations.com/pages/index.html.

[7] Honkala, Mikko, Dani Korpi, and Janne M. J. Huttunen. “DeepRx: Fully Convolutional Deep Learning Receiver.” IEEE Transactions on Wireless Communications 20, no. 6 (June 2021): 3925–40. https://doi.org/10.1109/TWC.2021.3054520.

[8] Paszke, Adam, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, et al. “PyTorch: An Imperative Style, High-Performance Deep Learning Library.” In Advances in Neural Information Processing Systems 32, 8024–35. Curran Associates, Inc., 2019. http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.

[9] “ADALM-PLUTO Evaluation Board | Analog Devices.” Accessed November 15, 2021. https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html#eb-overview.



## Disclaimer: Legal Compliance in Radio Transmission

It is imperative to acknowledge that engaging in radio transmission activities, especially over the air, mandates strict adherence to regulatory and licensing requirements. The use of devices like the PlutoSDR for transmitting radio signals must be conducted responsibly and legally.

1. **Obtain Required Licenses**: Before initiating any radio transmission, ensure you have obtained all necessary licenses and permissions from the appropriate regulatory bodies.
2. **Follow Frequency Allocations**: Adhere to the designated frequency bands and power levels to prevent interference with licensed services and other critical communication systems.
3. **Ensure Safety**: Ensure that your transmission activities do not pose any risks to public safety or disrupt other lawful communication services.

Engaging in unauthorized radio transmission can result in severe legal consequences, including fines, legal actions, and the confiscation of equipment.
