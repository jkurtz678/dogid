import torch
from torch import nn

class ConvolutionalModel(nn.Module): 
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        conv_stride = 1
        conv_padding = 1
        conv_kernel_size = 3
        pooling_kernel = 2
        pooling_stride = 2
        #conv_output = hidden_units * 56 * 56
        conv_output = hidden_units * 14 * 14
        #output_size = ((input_shape - kernel_size + 2*padding)/stride) + 1
        #print("output_size: ", output_size)
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=conv_kernel_size,
                      stride=conv_stride,
                      padding=conv_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=conv_kernel_size,
                      stride=conv_stride,
                      padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel,
                         stride=pooling_stride)     
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, conv_kernel_size, padding=conv_padding),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, conv_kernel_size, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(pooling_kernel)
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, conv_kernel_size, padding=conv_padding),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, conv_kernel_size, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(pooling_kernel)
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, conv_kernel_size, padding=conv_padding),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, conv_kernel_size, padding=conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(pooling_kernel)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conv_output,
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        
        #print(f"x.shape: {x.shape}")
        x = self.classifier(x)
        return x

class ImprovedConvolutionalModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # This will handle any input size
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_shape)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)
        return x