import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class OCRDataset(Dataset):
    def __init__(self, data, tokenizer, transform=None):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data

        self.df = self.df.dropna(subset=['image_path', 'label'])
        self.df = self.df[self.df['label'].apply(lambda x: isinstance(x, str))]
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = row['label']

        try:
            image = Image.open(img_path).convert("L")
            
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Erreur chargement {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        return image, self.tokenizer.encode(label), label

def collate_fn(batch):
    images, encoded_labels, original_labels = zip(*batch)
    
    images = torch.stack(images, 0)
    targets = torch.cat(encoded_labels)
    target_lengths = torch.tensor([len(l) for l in encoded_labels], dtype=torch.long)
    
    real_width = images.size(3)
    input_lengths = torch.full(size=(images.size(0),), fill_value=real_width // 4, dtype=torch.long)
    
    return images, targets, input_lengths, target_lengths, original_labels