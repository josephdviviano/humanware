"""
Adopted from https://github.com/potterhsu/SVHNClassifier-PyTorch
Cloned on Feb 20th 2019 by Joseph D Viviano.
"""
import os
import glob
import torch
import torch.nn as nn


class DeepConv(nn.Module):
    CHECKPOINT_FILENAME_PATTERN = 'model-{}.tar'

    def __init__(self, dropout=0.2):
        super(DeepConv, self).__init__()

        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=48,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48,
                      out_channels=64,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=160,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160,
                      out_channels=192,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192,
                      out_channels=192,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192,
                      out_channels=192,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192,
                      out_channels=192,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        )
        hidden9 = nn.Sequential(
            nn.Linear(192 * 7 * 7, 3072),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )
        self._classifier = nn.Sequential(
            hidden9,
            hidden10
        )
        self._digit_length = nn.Sequential(nn.Linear(3072, 7))
        self._digit1 = nn.Sequential(nn.Linear(3072, 11))
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))
        self._digit3 = nn.Sequential(nn.Linear(3072, 11))
        self._digit4 = nn.Sequential(nn.Linear(3072, 11))
        self._digit5 = nn.Sequential(nn.Linear(3072, 11))

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), 192 * 7 * 7)
        x = self._classifier(x)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]
        return length_logits, digits_logits
