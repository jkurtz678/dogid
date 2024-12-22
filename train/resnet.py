import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # Set dropout rates based on the layer depth (determined by channel size)
        if in_channels <= 64:  # First layer
            dropout1_rate = 0.1
            dropout2_rate = 0.2
        elif in_channels <= 128:  # Second layer
            dropout1_rate = 0.15
            dropout2_rate = 0.25
        elif in_channels <= 256:  # Third layer
            dropout1_rate = 0.2
            dropout2_rate = 0.3
        else:  # Fourth layer
            dropout1_rate = 0.25
            dropout2_rate = 0.35
        
        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout1_rate)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout2_rate)
        
        # Shortcut connection (identity mapping or 1x1 conv)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Save input for skip connection
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Add skip connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=120):
        super(ResNet, self).__init__()
        self.in_channels = 64

         # Set specific batch norm momentum
        self.bn_momentum = 0.1  # Try 0.1 or 0.01
        def bn_momentum_adjust(m):
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = self.bn_momentum
        
        self.apply(bn_momentum_adjust)
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classifier with dropout (similar to your current model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512 * block.expansion),  # Add BN before dropout
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion, num_classes),
            #nn.BatchNorm1d(1024),  # Add BN after linear
            #nn.ReLU(),
            #nn.Dropout(0.3),
            #nn.Linear(1024, num_classes),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        # First block might have different stride
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

def create_resnet18(num_classes=120):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    
    # Initialize the Linear layers (indices 3 and 7 in the classifier sequence)
    nn.init.normal_(model.classifier[3].weight, std=0.001)  # First Linear layer
    nn.init.constant_(model.classifier[3].bias, 0)
    
    #nn.init.normal_(model.classifier[7].weight, std=0.001)  # Second Linear layer
    #nn.init.constant_(model.classifier[7].bias, 0)
    
    return model

def create_resnet34(num_classes=120):
    """Creates a ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
