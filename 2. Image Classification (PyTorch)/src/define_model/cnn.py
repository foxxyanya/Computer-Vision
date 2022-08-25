import torch
import torch.nn as nn


class CNNClassificationModel(torch.nn.Module):
    
    def __init__(self):
        super(CNNClassificationModel, self).__init__()
        self.network = nn.Sequential(
            
            # input shape (3, 150, 150)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output shape (64, 75, 75)
        
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output shape (128, 37, 37)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output shape (256, 18, 18)
            
            nn.Flatten(), # output shape (82944)
            nn.Linear(82944, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )
    
    def forward(self, X):
        return self.network(X)
