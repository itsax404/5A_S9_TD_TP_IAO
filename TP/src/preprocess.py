import torch
import torchvision.transforms as T
from PIL import Image

class ImageTransform:
    def __init__(self, img_height=32, img_width=128):
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((img_height, img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __call__(self, img):
        return self.transform(img)

class TextEncoder:
    """Encode le texte en entiers pour le modèle et inversement."""
    def __init__(self, chars):
        self.chars = sorted(list(set(chars)))
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)} # 0 réservé pour le blank CTC
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.blank_idx = 0

    def encode(self, text):
        return torch.tensor([self.char_to_idx[c] for c in text if c in self.char_to_idx], dtype=torch.long)

    def decode(self, tokens):
        """Décodage CTC (suppression des doublons et des blanks)."""
        res = []
        last_token = self.blank_idx
        for token in tokens:
            token = token.item()
            if token != self.blank_idx and token != last_token:
                res.append(self.idx_to_char.get(token, ""))
            last_token = token
        return "".join(res)
    
    def __len__(self):
        return len(self.chars) + 1