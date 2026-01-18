import torch
import torchvision.transforms.functional as F
from PIL import Image

class ImageTransform:
    def __init__(self, img_height=32, img_width=1024):
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, img):
        # Redimensionnement proportionnel
        percent = float(self.img_height) / float(img.size[1])
        new_width = int((float(img.size[0]) * float(percent)))
        
        img = img.resize((new_width, self.img_height), Image.Resampling.LANCZOS)

        # Padding avec fond blanc (Grayscale)
        if new_width > self.img_width:
            img = img.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)
        else:
            new_img = Image.new('L', (self.img_width, self.img_height), 255)
            new_img.paste(img, (0, 0))
            img = new_img

        # Normalisation
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.5], std=[0.5])
        
        return img

class TextEncoder:
    def __init__(self, chars):
        self.chars = sorted(list(set(chars)))
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.blank_idx = 0

    def encode(self, text):
        return torch.tensor([self.char_to_idx[c] for c in text if c in self.char_to_idx], dtype=torch.long)

    def decode(self, tokens):
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