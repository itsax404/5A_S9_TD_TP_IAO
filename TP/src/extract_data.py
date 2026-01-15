import os
import zipfile
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import shutil

class DatasetManager:
    def __init__(self):
        current_path = Path(__file__).resolve()
        self.root_dir = current_path.parent.parent
            
        self.compressed_dir = self.root_dir / "compressed_data"
        self.data_dir = self.root_dir / "data"
        
        # Chemins RIMES
        self.rimes_zip_path = self.compressed_dir / "rimes" / "archive.zip"
        self.rimes_extract_dir = self.data_dir / "rimes"
        
        # Chemins EMNIST
        self.emnist_zip_path = self.compressed_dir / "emnist" / "archive.zip"
        self.emnist_extract_dir = self.data_dir / "emnist"

        # Création de l'arborescence de sortie (data)
        self.rimes_extract_dir.mkdir(parents=True, exist_ok=True)
        self.emnist_extract_dir.mkdir(parents=True, exist_ok=True)

    def _extract_zip(self, zip_path, extract_to):
        if not zip_path.exists():
            print(f"Erreur : Archive introuvable à {zip_path}")
            return False
            
        print(f"Extraction de {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except zipfile.BadZipFile:
            print("Erreur : Fichier zip corrompu.")
            return False

    def prepare_rimes(self):
        print("--- RIMES ---")
        csv_output = self.rimes_extract_dir / "metadata.csv"

        if csv_output.exists():
            print("ATTENTION : metadata.csv existe déjà. Supprimez-le pour régénérer.")
            return

        # Extraction si nécessaire
        if not any(self.rimes_extract_dir.iterdir()):
             if not self._extract_zip(self.rimes_zip_path, self.rimes_extract_dir):
                return
        
        # Recherche du fichier XML
        xml_files = list(self.rimes_extract_dir.rglob("*.xml"))
        data = []
        
        if not xml_files:
            print("ERREUR CRITIQUE : Aucun fichier XML trouvé dans le dossier RIMES.")
            print("Vérifiez le contenu de data/rimes/")
            return

        print(f"Fichier d'annotations trouvé : {xml_files[0].name}")
        tree = ET.parse(xml_files[0])
        root = tree.getroot()
        
        # Compteur pour vérifier si on trouve des labels
        count_found = 0
        
        # On itère sur tous les éléments pour être souple sur la structure
        for elem in root.iter():
            # RIMES standard utilise 'FileName' et 'Value'
            # D'autres versions utilisent 'text' ou 'Label'
            
            filename = elem.get('FileName')
            # Essayer plusieurs clés possibles pour le label
            label = elem.get('Value') or elem.get('Text') or elem.get('Label')
            
            if filename and label:
                # Chercher l'image correspondante
                full_path = list(self.rimes_extract_dir.rglob(filename))
                
                # Parfois le XML a l'extension .tif et l'image est .jpg
                if not full_path:
                    name_no_ext = Path(filename).stem
                    # Cherche n'importe quelle image avec ce nom
                    candidates = list(self.rimes_extract_dir.rglob(f"{name_no_ext}.*"))
                    # Filtre pour garder les images
                    full_path = [p for p in candidates if p.suffix.lower() in ['.jpg', '.png', '.tif']]

                if full_path:
                    data.append({
                        "image_path": str(full_path[0]), 
                        "label": label
                    })
                    count_found += 1

        if count_found == 0:
            print("ERREUR : Le fichier XML a été lu, mais aucun couple (FileName, Value) n'a été trouvé.")
            print("Affichez le contenu du XML pour vérifier les noms des attributs.")
            return

        # Sauvegarde
        df = pd.DataFrame(data)
        df = df[df['label'].str.strip().astype(bool)]
        df.to_csv(csv_output, index=False)
        
        print(f"SUCCÈS : metadata.csv régénéré avec {len(df)} images.")
        print(f"Exemple de label : {df.iloc[0]['label']}") 


    def prepare_emnist(self):
        print("\n--- EMNIST ---")
        raw_dir = self.emnist_extract_dir / "raw"

        # Vérification si déjà extrait
        if raw_dir.exists() and any(raw_dir.iterdir()):
            print("EMNIST déjà extrait.")
            return

        # Vérification de l'archive manuelle
        if not self.emnist_zip_path.exists():
            print(f"Erreur : Archive manquante.")
            print(f"Placez le fichier 'gzip.zip' dans : {self.emnist_zip_path.parent}")
            return

        # Extraction
        self._extract_zip(self.emnist_zip_path, raw_dir)

    def prepare_all(self):
        self.prepare_rimes()
        self.prepare_emnist()
        print("\nTerminé.")

if __name__ == "__main__":
    DatasetManager().prepare_all()