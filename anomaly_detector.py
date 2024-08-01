import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def is_anomaly(frame, model, threshold=0.01):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64), antialias=True)
    ])
    frame_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        output = model(frame_tensor)
        loss = F.mse_loss(output, frame_tensor)
    return loss.item() > threshold
