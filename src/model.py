import torch
import torch.nn as nn


class cnn(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units), nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # halbiert
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units*4), nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(hidden_units*4, hidden_units*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units*4), nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(hidden_units*4, hidden_units*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_units*4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # nochmal halbiert
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # -> [B, C, 1, 1]
            nn.Flatten(),             # -> [B, C]
            nn.Linear(hidden_units*4, hidden_units*8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units*8, hidden_units*8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units*8, output_shape)
        )

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.layer5(x)
        x = self.head(x)
        return x