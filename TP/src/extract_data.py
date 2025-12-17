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
            print("RIMES déjà prêt.")
            return

        # Extraction si le dossier est vide
        if not any(self.rimes_extract_dir.iterdir()):
             if not self._extract_zip(self.rimes_zip_path, self.rimes_extract_dir):
                return
        
        # Parsing du XML
        xml_files = list(self.rimes_extract_dir.rglob("*.xml"))
        data = []
        
        if xml_files:
            print(f"Création du CSV depuis : {xml_files[0].name}")
            tree = ET.parse(xml_files[0])
            for elem in tree.getroot().iter():
                if 'FileName' in elem.attrib and 'Value' in elem.attrib:
                    filename = elem.attrib['FileName']
                    full_path = list(self.rimes_extract_dir.rglob(filename))
                    if full_path:
                        data.append({
                            "image_path": str(full_path[0]), 
                            "label": elem.attrib['Value']
                        })
        else:
            print("Pas de XML trouvé, indexation simple des images.")
            for img in self.rimes_extract_dir.rglob("*"):
                if img.suffix.lower() in ['.png', '.jpg']:
                    data.append({"image_path": str(img), "label": img.stem})

        if data:
            pd.DataFrame(data).to_csv(csv_output, index=False)
            print(f"metadata.csv créé ({len(data)} images).")

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