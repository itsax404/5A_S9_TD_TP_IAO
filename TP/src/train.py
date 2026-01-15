import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from dataset import OCRDataset, collate_fn
from model import CRNN
from preprocess import ImageTransform

def train():
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chemins
    root_dir = Path(__file__).resolve().parent.parent
    csv_path = root_dir / "data" / "rimes" / "metadata.csv"
    model_save_path = root_dir / "app" / "model.pth"

    if not csv_path.exists():
        print(f"Erreur: {csv_path} introuvable. Lancez download_data.py d'abord.")
        return

    # Données
    transform = ImageTransform()
    dataset = OCRDataset(str(csv_path), transform=transform)
    
    # Split Train/Val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Modèle
    # len(tokenizer) inclut déjà le blank CTC
    model = CRNN(num_classes=len(dataset.tokenizer)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"Démarrage entraînement sur {DEVICE} avec {len(dataset)} images.")
    
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, targets, input_lengths, target_lengths, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(images) # (T, B, C)
            
            # Log_softmax requis par CTCLoss PyTorch (sauf si inclus, mais safer ici)
            preds_log_softmax = preds.log_softmax(2)
            
            loss = criterion(preds_log_softmax, targets, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} terminée. Loss moyenne: {avg_loss:.4f}")

        # Sauvegarde du meilleur modèle
        if avg_loss < best_loss:
            best_loss = avg_loss
            # On sauvegarde le state_dict ET le tokenizer pour l'inférence
            torch.save({
                'model_state': model.state_dict(),
                'chars': dataset.tokenizer.chars
            }, model_save_path)
            print(f"Modèle sauvegardé dans {model_save_path}")

if __name__ == "__main__":
    train()