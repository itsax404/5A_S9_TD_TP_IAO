import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, input_channels=1, hidden_size=256):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
        )

        self.rnn = nn.LSTM(512 * 4, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Extraction des features via CNN
        features = self.cnn(x)
        b, c, h, w = features.size()
        
        # Redimensionnement pour le RNN (Batch, Time, Features)
        # La hauteur (h) est fusionnée avec les canaux (c)
        features = features.view(b, c * h, w) 
        features = features.permute(0, 2, 1) 
        
        # Passage dans le RNN
        output, _ = self.rnn(features)
        output = self.fc(output)
        
        # Format [Time, Batch, Classes] requis par CTCLoss
        return output.permute(1, 0, 2)