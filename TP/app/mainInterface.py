import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import torch
from pathlib import Path
from PIL import Image

# Gestion des chemins pour trouver le dossier src/
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.model import CRNN
from src.preprocess import ImageTransform, TextEncoder

class SimpleOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Tool")
        self.root.geometry("300x150")
        
        self.center_window()

        # Label d'état
        self.label_status = tk.Label(root, text="Pret", fg="black")
        self.label_status.pack(side=tk.BOTTOM, pady=5)

        # Bouton principal
        self.btn_new = tk.Button(root, text="Nouveau", command=self.process_workflow, height=2, width=15, font=("Arial", 11, "bold"))
        self.btn_new.pack(expand=True)

        # Initialisation
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.transform = ImageTransform()
        
        self.load_model()

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def load_model(self):
        model_path = current_file.parent / "model.pth"
        
        if not model_path.exists():
            messagebox.showerror("Erreur", f"Modele introuvable :\n{model_path}\n\nVeuillez lancer l'entrainement.")
            self.root.destroy()
            return

        try:
            self.label_status.config(text="Chargement du modele...")
            self.root.update()
            
            checkpoint = torch.load(model_path, map_location=self.device)
            self.tokenizer = TextEncoder(checkpoint['chars'])
            
            self.model = CRNN(num_classes=len(self.tokenizer)).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            
            self.label_status.config(text="Modele charge.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le modele :\n{e}")
            self.root.destroy()

    def predict_text(self, image_path):
        try:
            # IMPORTANT : Conversion en 'L' (Grayscale) pour correspondre a l'entrainement
            img = Image.open(image_path).convert("L")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                preds = self.model(img_tensor)
                decoded = self.tokenizer.decode(preds.argmax(2).squeeze(1))
                return decoded
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lecture image :\n{e}")
            return None

    def process_workflow(self):
        # 1. Selection fichier
        file_path = filedialog.askopenfilename(
            title="Selectionner une image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
        )
        
        if not file_path:
            return

        self.label_status.config(text="Analyse en cours...")
        self.root.update()

        # 2. Prediction
        predicted_text = self.predict_text(file_path)
        
        if predicted_text is None:
            self.label_status.config(text="Erreur analyse.")
            return

        # 3. Sauvegarde
        save_path = filedialog.asksaveasfilename(
            title="Sauvegarder le resultat",
            defaultextension=".txt",
            filetypes=[("Fichier Texte", "*.txt")],
            initialfile=f"resultat_{Path(file_path).stem}.txt"
        )

        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(predicted_text)
                
                self.label_status.config(text="Sauvegarde terminee.")
                messagebox.showinfo("Succes", f"Resultat :\n\n{predicted_text}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'ecrire le fichier :\n{e}")
        else:
            self.label_status.config(text="Operation annulee.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleOCRApp(root)
    root.mainloop()