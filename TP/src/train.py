import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import sys
import time

# Gestion des chemins d'importation
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from src.dataset import OCRDataset, collate_fn
from src.model import CRNN
from src.preprocess import ImageTransform, TextEncoder

def train():
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_dir = current_dir.parent
    csv_path = root_dir / "data" / "rimes" / "metadata.csv"
    model_save_path = root_dir / "app" / "model.pth"

    print(f"Demarrage de l'entrainement sur {DEVICE}")
    print(f"Configuration : Batch={BATCH_SIZE}, Epochs={EPOCHS}, LR={LR}")

    if not csv_path.exists():
        print("Erreur : metadata.csv introuvable.")
        return

    # Chargement des données
    full_df = pd.read_csv(csv_path)
    tokenizer = TextEncoder("".join(full_df['label'].astype(str).tolist()))
    transform = ImageTransform() 

    # Séparation Train / Val
    train_df = full_df[full_df['split'] == 'train']
    val_df = full_df[full_df['split'] == 'val']
    
    if val_df.empty:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(train_df, test_size=0.1)

    print(f"Images d'entrainement : {len(train_df)}")
    print(f"Images de validation  : {len(val_df)}")

    # Création des DataLoaders
    train_ds = OCRDataset(train_df, tokenizer, transform)
    val_ds = OCRDataset(val_df, tokenizer, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Initialisation du modèle
    model = CRNN(num_classes=len(tokenizer)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_val_loss = float('inf')

    # Boucle d'entraînement
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        
        for batch_idx, (imgs, targets, input_lens, target_lens, _) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs).log_softmax(2)
            loss = criterion(preds, targets, input_lens, target_lens)
            loss.backward()
            
            # Gradient clipping pour la stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            total_train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", end="\r")

        avg_train_loss = total_train_loss / len(train_loader)

        # Phase de validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, targets, input_lens, target_lens, _ in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs).log_softmax(2)
                loss = criterion(preds, targets, input_lens, target_lens)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        duration = time.time() - start_time
        print(f"\nFin Epoch {epoch+1}/{EPOCHS} ({duration:.0f}s)")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {avg_val_loss:.4f}")

        # Mise à jour du learning rate
        scheduler.step(avg_val_loss)

        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state': model.state_dict(),
                'chars': tokenizer.chars,
                'epoch': epoch,
                'loss': best_val_loss
            }, model_save_path)
            print(f"  Modele sauvegarde (Validation Loss : {best_val_loss:.4f})")

if __name__ == "__main__":
    train()