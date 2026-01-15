import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from preprocess import ImageTransform, TextEncoder

class OCRDataset(Dataset):
    def __init__(self, csv_path, transform=None, tokenizer=None):
        self.df = pd.read_csv(csv_path)
        # Filtrage des données manquantes ou invalides
        self.df = self.df.dropna(subset=['image_path', 'label'])
        self.df = self.df[self.df['label'].apply(lambda x: isinstance(x, str))]
        
        self.transform = transform
        
        # Si aucun tokenizer n'est fourni, on en crée un basé sur tout le dataset
        if tokenizer is None:
            all_text = "".join(self.df['label'].tolist())
            self.tokenizer = TextEncoder(all_text)
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label_str = row['label']

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Erreur chargement {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self)) # Fallback simple

        label_encoded = self.tokenizer.encode(label_str)
        return image, label_encoded, label_str

def collate_fn(batch):
    """Gère les batches avec des labels de longueurs variables."""
    images, encoded_labels, original_labels = zip(*batch)
    
    images = torch.stack(images, 0)
    
    # On concatène tous les labels pour CTCLoss
    targets = torch.cat(encoded_labels)
    
    # Longueurs pour CTCLoss
    target_lengths = torch.tensor([len(l) for l in encoded_labels], dtype=torch.long)
    input_lengths = torch.full(size=(images.size(0),), fill_value=32, dtype=torch.long) # 32 est la largeur de feature map CNN
    
    return images, targets, input_lengths, target_lengths, original_labels