import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, input_channels=1, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN : Extraction de features
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x128 -> 16x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x64 -> 8x32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)), # 8x32 -> 4x32 (On garde la largeur)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # RNN : Séquence
        # Input shape au RNN : (Batch, Width, Features)
        self.rnn = nn.LSTM(512 * 4, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        
        # Classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, 32, 128)
        features = self.cnn(x) 
        
        # Transformation pour le RNN
        # (B, 512, 4, 32) -> (B, 512*4, 32) -> (B, 32, 2048)
        b, c, h, w = features.size()
        features = features.view(b, c * h, w)
        features = features.permute(0, 2, 1) 
        
        # RNN
        output, _ = self.rnn(features)
        
        # FC
        output = self.fc(output)
        
        # Pour CTCLoss, output doit être (Time, Batch, Class)
        return output.permute(1, 0, 2)