# OFDM-PlutoSDR: Bridging SDR, OFDM and AI for Next-Gen Wireless Communication

## Introduction

This project provides an end-to-end Python implementation of an Over-the-Air (OTA) Orthogonal Frequency Division Multiplexing (OFDM) communication system using the Analog Devices ADALM-Pluto (PlutoSDR) for 1T1R SISO transmission and reception using the same SDR both for transmitting and receiving. Leveraging PyTorch, the framework enables seamless integration and experimentation with modern machine learning techniques for state-of-the-art Physical Layer (PHY) processing.

Beyond classical Least Squares (LS) channel estimation and Zero Forcing (ZF) equalization, this implementation introduces a neural network (NN)-based receiver that predicts bits directly from IQ signals following the Discrete Fourier Transform (DFT) block at the receiver. The project also supports dataset generation, model training, and testing over both simulated channels and real radio interfaces.

A central contribution is the functionality allowing comparison of an NN-based receiver—trained on simulated channels and tested over-the-air or with simulated data—against the LS/ZF approach. The NN model, inspired by the DeepRx architecture (Honkala et al., 2021), uses a simplified structure to demonstrate that data-driven PHY can meet or exceed classical performance in practical settings.

## Short literature review

OFDM, originally formalized for multichannel data transmission by Chang [1], is foundational to contemporary wireless systems. Goldsmith [3] provides core theory on wireless channels, enabling practical OFDM deployment. The ADALM-Pluto board [9] have lowered the barrier for prototyping and over-the-air experimentation. Web sites "PySDR: A Guide to SDR and DSP using Python" [5] and "DSPIllustrations" [6] provide insights into use of SDR and signal processing concepts using python.

Recent research applies deep learning to the PHY. Goodfellow et al. [2] and Paszke et al. [8] presenting the theoretical and software foundations. Honkala et al. [7] show fully convolutional deep receivers (DeepRx) surpassing traditional methods in wireless environments, while Ju et al. [4] compare deep learning with iterative algorithms for joint channel estimation and detection in OFDM. 

## Python libraries and Jupyter noteboks

### Prerequisites

Python > 3.10, PyTorch, numPy, matplotlib, libiio, pylib-iio

TODO: Some additional libraries may be required - update.

PlutoSDR required for hardware experiments.

### Installation

```
git clone https://github.com/rikluost/pluto # for the latest development version
```

### Key files and folders

#### Example Notebooks
- `10-ofdm-example-func.ipynb`: Full OFDM Tx/Rx chain, comparing classical LS/ZF and DeepRx-style NN receivers.
- `30-dataset-creator.ipynb`: Dataset generator for training NN-based receivers with random data and simulated channel.
- `40-receiver-training.ipynb`: Training and validation of a ResNet-style, fully convolutional NN-based receiver.
- `50-test-receivers.ipynb`: Performance comparison between CNNNN-based and LS/ZF receivers.

#### Key libraries and files
- `SDR_Pluto.py`: SDR interface utilities.
- `OFDM_SDR_Functions_torch.py`: OFDM, channel simulation, and PHY utilities.
- `config.py`: Configurations and dataset classes.
- `models_local.py`: PyTorch architectures for NN-based receivers.

#### Key folders
- `data/`: Trained models, weights, and datasets.
- `pics/`, `pics_doc/`: Saved plots for analysis and documentation.


## Introduction to the implemented OFDM workflow

The workflow covers TTI Mask Creation with Pilot Symbols, Data Stream Creation, Modulation, CP Addition, Radio Channel Simulation / Over The Air Transmission, Synchronization, Channel Estimation, Equalization, Symbol Demapping, and Bit Error Rate (BER) Calculation.

### TTI Mask Creation

Creation of a TTI (Transmission Time Interval) mask in an OFDM system involves specifying the constraints for the transmission of different types of data symbols, e.g. pilot symbols, DC, and user data. In this implementation, pilots in only one OFDM-symbol is possible, though modifying this should be easy.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/TTImask.png) 

Fig 1. TTI mask.

### Data stream creation

A data stream is created for filling the TTI PDSCH resources with random data. This data is in the following stages used for modulating the TTI, and the data is also stored to enable comparison with the received data after demodulation and demapping for Bit Error Rate (BER) calculation at the receiver. 

### Serial to parallel

The conversion of a data stream from serial to parallel format involves dividing the incoming serial bit stream into groups where the size is defined by the modulation order. The number of the groups equals to the total number of PDSCH elements in one TTI.

### Modulation

The process of encoding the bits onto the different subcarriers by varying their amplitude and phase using M-QAM-modulation, according to the data being transmitted.

Figures 2 and 3 illustrate example cases of 16-QAM constellation and fully populated TTI where the colors are defined by the magnitude of each complex vector.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/QAMconstellation.png) 

Fig 2. Constellation.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/TTImod.png) 

Fig 3. Modulated TTI.

### Conversion of frequency domain symbols into time domain (IFFT)
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

$\mathbf{H}$, the radio channel matrix, can be constructed here either by transmitting the signal $\mathbf{x}$ with an SDR transmitter, and then receiving the signal with an SDR receiver, or by using a simplified randomized radio channel model. Noise, in case of SDR, is naturally injected into the signal, while in case of using simulated model, it is added in the received signal.

#### Transmission of IQ signals (SDR)

The signals can be passed through the SDR transmitter, or through a simulated radio channel. Figure 4 below shows an example of the power spectral density of transmitted signal $\mathbf{x}$:

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/PSD_TX.png)

Fig 4. Power spectral density of the signal ready for transmission.

#### Reception of IQ signals (SDR)

The signal can be received by the SDR receiver, which translates it to time domain IQ signals, or alternatively by using the output of simulated radio channel. The graph in Figure 5 shows an example of the power spectral density of received signal $\mathbf{y}$:

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/PSD_RX.png) 

Fig 5. Power spectral density of the signal.

### Synchronisation
PlutoSDR lacks capability of fully syncing TX and RX, e.g. with timestamps. This has been solved in this implementation by utilising cyclic transmissions, and time domain synchronisation by using cross-correlation. Frequency domain correction is not required as TX and RX utilise the same physical clock. Figure 6 shows an example of the correlation of transmitted and received signals at symbol level. Maximum correlation is used to identify the first symbol.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/corr.png) 

Fig 6. Synchronisation by correlation. ABS correlation of symbols from 10 before to 50 after the detected start of the signal.

### CP Removal

Unmodulated samples between the TTI's are injected in the cyclic transmission to allow measuring noise level for the SINR estimation. After the syncronisation, the CP from each received OFDM symbol is discarded before further processing.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/RXsignal_sync.png) 

Fig 7. CP Removal by using a mask after synchronization.

### Conversion of time domain IQ symbols into frequency domain (DFT)

The received time-domain signal is converted back into the frequency domain using DFT. This operation separates the data on the different subcarriers, allowing for demodulation and further processing to retrieve the pilot symbols and eventually the transmitted data.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/TTI_RX.png) 

Fig 8. Received TTI after synchronization, but before the equalization.

### Channel Estimation

In case of conventional receiver, channel estimation involves determining the properties of the transmission channel from pilot symbols to mitigate the adverse effects of the radio channel. This typically entails using known pilot symbols or training sequences to estimate the channel's frequency response. The output of the channel estimation is $\mathbf{H}$, the estimated channel matrix. Only one time domain OFDM symbol can be allocated for pilot signals, which is not optimal for mobile radios as the channel can change rapidly and hence for example in 4G and 5G two timedomain pilots are commonly used. As can be seen later, NN-based receiver with only one OFDM pilot symbol allocated for pilots seem to perform better in particular with mobile radio environment than the more conventional LS/ZF based receiver implemented here.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/ChannelEstimateAbs.png) 

Fig 9. Absolute values of the received pilot symbols with interpolations.

### Equalization

Equalization is the process of compensating for the impairments introduced by the transmission channel by dividing the received signal $\mathbf{y}$ by the channel estimate $\mathbf{H}$. Here Zero Forcing algorithm is implemented.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/RXdSymbols.png) 

Fig 10. Received symbols of the PDCCH symbols of one TTI before and after signal equalization.

### Symbol demapping

This is the process of converting the received modulated symbols back into their corresponding bit representations. It involves translating the complex values from the demodulated subcarriers into the original transmitted bits based on the used modulation scheme.

### Data stream recreation

The data stream is entailed by concatenating the demapped bits from each subcarrier and symbol, converting the parallel data streams back into a single serial data stream

### Bit Error Rate (BER) calculation

Compare the original transmitted bit stream with the received bit stream, bit by bit and calculate BER.

## NN-based receiver

### Training data generation

The dataset creation example `30-NN-receiver-dataset-creator.ipynb` generates random data and OFDM TTIs modulated with it. The OFDM signals are sent over the simulated radio channel. It stores the OFDM data after reception, cyclic prefic removal and DFT. The created dataset contains received OFDM symbols and pilot symbols as input, and original bits used for the modulation as labels. There are 6 bits per label, so 64QAM was used here.

### Training a NN-based receiver model

To showcase the setup, a training dataset containing 20,000 samples, each with randomized data and radio channel characteristics, was utilized to train a model as defined in the `models_local.py` file. Although the potential exists for using a much larger dataset, the training in this instance was completed only in about 10 minutes. It was halted as there appeared to be no further improvement in performance, a detail that is evident in Fig 10 included in the documentation.

The training process and its nuances are detailed in the `40-training-NN-based-receiver.ipynb` notebook. This example primarily serves as a demonstration of the setup's capabilities; the model architecture and hyperparameters used in this demonstration are basic and further optimization will potentially enhance performance. Figure 11 below shows the performance of an NN-based model during training. Training here was done using simulated radio channel, while validation was performed by using SDR data.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/training_loss.png) 

Fig 11. The model performance during the training proces

### Comparing performance on testset

Notebook `50-compare-ZF-LS-with-NN-based-RX-testset.ipynb` loads the model and its weights, and the test dataset saved earlier. It implements both the NN-based receiver as well as the LS/ZF based receiver used in the OFDM example notebook. The perormance on the testset is compared and the distributions of the BERs are shown in Fig 12, 13. 

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/BER_distribution_testset_log_scale.png) 

Fig 12. Bit Error Rate (BER) distribution on the testset, comparing NN-based receiver to ZF/LS receiver.

![alt text](https://github.com/rikluost/ofdm-plutosdr-pytorch/blob/main/pics_doc/Performance_LS_nn.png) 

Fig 13. Performance comparison between the NN-based and classic receivers over real-life radio interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References and Resources

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
