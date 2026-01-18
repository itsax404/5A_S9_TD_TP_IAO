import sys
import os
import torch
from pathlib import Path
from PIL import Image

# Configuration
IMAGE_A_TESTER = "C:/Users/joach/5A/IA/bard/5A_S9_TD_TP_IAO/TP/data/rimes/Handwritten2Text Training Dataset/Images/eval2011-0_000009.jpg"
INPUT_FOLDER_NAME = "input"

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.model import CRNN
from src.preprocess import ImageTransform, TextEncoder

def resolve_image_path(user_path=None):
    """
    Détermine le chemin de l'image à traiter selon la priorité :
    Argument CLI > Variable globale > Dossier input par défaut
    """
    candidate = user_path if user_path else IMAGE_A_TESTER
    
    if candidate and isinstance(candidate, str) and len(candidate.strip()) > 0:
        if os.path.exists(candidate):
            return candidate
        
        abs_path = project_root / candidate
        if abs_path.exists():
            return str(abs_path)
            
        print(f"Attention : Image introuvable : {candidate}")
        print(f"Tentative de bascule sur le dossier '{INPUT_FOLDER_NAME}/'...\n")
    
    input_dir = project_root / INPUT_FOLDER_NAME
    
    if not input_dir.exists():
        print(f"Erreur : Dossier '{INPUT_FOLDER_NAME}' inexistant.")
        return None

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts]
    
    if not images:
        print(f"Erreur : Le dossier '{INPUT_FOLDER_NAME}' est vide.")
        return None
        
    found_img = images[0]
    print(f"Image selectionnee : {found_img.name}")
    return str(found_img)

def predict(image_path, model_path):
    device = torch.device("cpu")
    
    if not Path(model_path).exists():
        print(f"Erreur : Modele introuvable ({model_path}).")
        return

    # Chargement du modèle
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Erreur chargement modele : {e}")
        return

    tokenizer = TextEncoder(checkpoint['chars'])
    model = CRNN(num_classes=len(tokenizer)).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    transform = ImageTransform()
    
    # Chargement et pré-traitement de l'image
    try:
        # Conversion explicite en "L" (Grayscale) pour cohérence avec l'entraînement
        img = Image.open(image_path).convert("L")
        img_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Erreur lecture image : {e}")
        return

    print(f"Traitement de : {Path(image_path).name}")
    
    # Inférence
    with torch.no_grad():
        preds = model(img_tensor)
        decoded = tokenizer.decode(preds.argmax(2).squeeze(1))
        
        print("-" * 30)
        print(f"PREDICTION : {decoded}")
        print("-" * 30)
        return decoded

if __name__ == "__main__":
    model_file = current_file.parent / "model.pth"
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    final_image_path = resolve_image_path(cli_arg)
    
    if final_image_path:
        predict(final_image_path, str(model_file))