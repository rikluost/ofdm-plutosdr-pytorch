import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as tFunc
from config import *

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=(S, FFT_size_RX))
        self.conv2d_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=False)

        self.layer_norm_2 = nn.LayerNorm(normalized_shape=(S, FFT_size_RX))
        self.conv2d_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, inputs):
        z = self.layer_norm_1(inputs)
        z = tFunc.relu(z)
        z = self.conv2d_1(z)
        z = self.layer_norm_2(z)
        z = tFunc.relu(z)
        z = self.conv2d_2(z)
        z = z + inputs

        return z


class RXModel_2(nn.Module):
    def __init__(self, num_bits_per_symbol):
        super(RXModel_2, self).__init__()

        # Input convolution
        self.input_conv2d = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(3, 3), padding=1)

        # Residual blocks
        self.res_block_1 = ResidualBlock()
        self.res_block_2 = ResidualBlock()
        self.res_block_3 = ResidualBlock()
        self.res_block_4 = ResidualBlock()
        self.res_block_5 = ResidualBlock()

        # Output convolution
        self.output_conv2d = nn.Conv2d(in_channels=128, out_channels=num_bits_per_symbol, kernel_size=(3, 3), padding=1)

    def forward(self, inputs):
        y = inputs
   
        # Stack the tensors along a new dimension (axis 0) and permute to match Conv2D input shape
        #z = torch.stack([y.real, y.imag], dim=1).float()
        z = self.input_conv2d(y)

        # Residual blocks
        z = self.res_block_1(z)
        z = self.res_block_2(z)
        z = self.res_block_3(z)
        z = self.res_block_4(z)
        z = self.res_block_5(z)

        # Output convolution
        z = self.output_conv2d(z)

        # Reshape and apply sigmoid activation
        z = z.permute(0, 2, 3, 1)
        z = nn.Sigmoid()(z)

        return z

