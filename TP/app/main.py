import torch
import sys
import os
from PIL import Image
from pathlib import Path

# Ajout du dossier src au path pour les imports
current_file_path = Path(__file__).resolve()
app_dir = current_file_path.parent
project_root = app_dir.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.model import CRNN
from src.preprocess import ImageTransform, TextEncoder

def predict(image_path, model_path):
    device = torch.device("cpu") # Inférence CPU par défaut
    
    if not os.path.exists(model_path):
        print("Erreur : Modèle introuvable. Entraînez le modèle d'abord.")
        return

    # Chargement
    checkpoint = torch.load(model_path, map_location=device)
    chars = checkpoint['chars']
    tokenizer = TextEncoder(chars)
    
    model = CRNN(num_classes=len(tokenizer)).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Prétraitement
    transform = ImageTransform()
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device) # (1, 1, 32, 128)
    except Exception as e:
        print(f"Erreur ouverture image: {e}")
        return

    # Inférence
    with torch.no_grad():
        preds = model(img_tensor) # (T, 1, C)
        preds_idx = preds.argmax(dim=2) # (T, 1)
        preds_flat = preds_idx.squeeze(1)
        
        text = tokenizer.decode(preds_flat)
        print(f"Prédiction : {text}")
        return text

if __name__ == "__main__":
    # Utilisation : python main.py chemin/vers/image.png
    model_file = app_dir / "model.pth"
    
    if len(sys.argv) > 1:
        img_file = sys.argv[1]
        predict(img_file, str(model_file))
    else:
        print("Usage: python main.py <chemin_image>")