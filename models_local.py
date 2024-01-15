import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as tFunc
from config import *

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=( S,F-1))
        self.conv2d_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=False)

        self.layer_norm_2 = nn.LayerNorm(normalized_shape=(S,F-1))
        self.conv2d_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, inputs):
        z = self.layer_norm_1(inputs)
        z = tFunc.relu(z)
        z = self.conv2d_1(z)
        z = self.layer_norm_2(z)
        z = tFunc.relu(z)
        z = self.conv2d_2(z)
        
        # Skip connection
        z = z + inputs

        return z
    

###################### Simple DeepRX-type Neural Network Receiver, PyTorch model #############################

class RXModel_2(nn.Module):

    def __init__(self, num_bits_per_symbol):
        super(RXModel_2, self).__init__()

        # Input convolution
        self.input_conv2d = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(3, 3), padding='same')

        # Residual blocks
        self.res_block_1 = ResidualBlock()
        self.res_block_2 = ResidualBlock()
        self.res_block_3 = ResidualBlock()
        self.res_block_4 = ResidualBlock()
        self.res_block_5 = ResidualBlock()

        # Output conv
        self.output_conv2d = nn.Conv2d(in_channels=128, out_channels=num_bits_per_symbol, kernel_size=(3, 3), padding='same')

    def forward(self, inputs):
        y = inputs
   
        # Stack the tensors along a new dimension (axis 0)
        z = torch.stack([y.real, y.imag], dim=0)
        z = z.permute(1, 0, 2, 3)
        z = self.input_conv2d(z)

        # Residual blocks
        z = self.res_block_1(z)
        z = self.res_block_2(z)
        z = self.res_block_3(z)
        z = self.res_block_4(z)
        z = self.res_block_5(z)
        z = self.output_conv2d(z)

        # Reshape 
        z = z.permute(0,2, 3, 1)
        z = nn.Sigmoid()(z)

        return z