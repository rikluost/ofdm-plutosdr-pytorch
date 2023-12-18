import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as tFunc

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=(128, 14, 71))
        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        #torch.nn.init.xavier_uniform(self.conv_1.weight)

        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=(128, 14, 71))
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, inputs):
        z = self.layer_norm_1(inputs)
        z = tFunc.relu(z)
        z = self.conv_1(z)
        z = self.layer_norm_2(z)
        z = tFunc.relu(z)
        z = self.conv_2(z)
        
        # Skip connection
        z = z + inputs

        return z
    

###################### Simple DeepRX-type Neural Network Receiver, PyTorch model #############################

class RXModel(nn.Module):

    def __init__(self, num_bits_per_symbol):
        super(RXModel, self).__init__()

        # Input convolution
        self.input_conv = nn.Conv2d(in_channels=4, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False)

        # Residual blocks
        self.res_block_1 = ResidualBlock()
        self.res_block_2 = ResidualBlock()
        self.res_block_3 = ResidualBlock()
        self.res_block_4 = ResidualBlock()
        self.res_block_5 = ResidualBlock()

        # Output conv
        self.output_conv = nn.Conv2d(in_channels=128, out_channels=num_bits_per_symbol, kernel_size=(3, 3), padding=(1, 1), bias=False)

    def forward(self, inputs):
        y, pilots = inputs
   
        # Stack the tensors along a new dimension (axis 0)
        z = torch.stack([y.real, y.imag, pilots.real, pilots.imag], dim=0)
        z = z.permute(1, 0, 2, 3)
        # Input conv
        z = self.input_conv(z)

        # Residual blocks
        z = self.res_block_1(z)
        z = self.res_block_2(z)
        z = self.res_block_3(z)
        z = self.res_block_4(z)
        z = self.res_block_5(z)
        z = self.output_conv(z)

        # Reshape the input to fit what the resource grid demapper is expected
        z = z.permute(0,2, 3, 1)
        z = nn.Sigmoid()(z)

        return z
