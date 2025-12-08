import torch
import torch.nn as nn


class DynamicCNN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        conv_channels: list[int],     
        linear_units: list[int],       
        kernel_size: int,
        pool_type: str,
        pool_every: int,
        dropout: float,
        n_classes: int,
        img_size: int, 
    ):
        super().__init__()
        k = kernel_size
        Pool = nn.MaxPool2d if pool_type == "max" else nn.AvgPool2d

        # Convolutional-Teil (dynamisch) 
        conv_layers = []
        prev_ch = in_ch
        current_spatial = img_size 

        for i, ch in enumerate(conv_channels):
            conv_layers.append(
                nn.Conv2d(prev_ch, ch, k, padding=k // 2, bias=False)
            )
            conv_layers.append(nn.BatchNorm2d(ch))
            conv_layers.append(nn.ReLU(inplace=True))

            #nach jedem pool_every-Conv Max/Avg-Pooling
            if (i + 1) % pool_every == 0 and current_spatial > 2:
                conv_layers.append(Pool(2))
                current_spatial //= 2  

            prev_ch = ch

        self.conv = nn.Sequential(*conv_layers)

        # Fully-Connected-Teil (dynamisch) 
        head_layers = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), 
        ]
        in_features = prev_ch

        for units in linear_units:
            head_layers.append(nn.Linear(in_features, units))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(dropout))
            in_features = units

        head_layers.append(nn.Linear(in_features, n_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x
