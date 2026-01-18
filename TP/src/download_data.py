import os
import zipfile
import pandas as pd
from pathlib import Path

class DatasetManager:
    def __init__(self):
        current_path = Path(__file__).resolve()
        self.root_dir = current_path.parent.parent if current_path.parent.name == 'src' else current_path.parent
            
        self.data_dir = self.root_dir / "data"
        self.rimes_dir = self.data_dir / "rimes"
        self.compressed_dir = self.root_dir / "compressed_data"

    def prepare_rimes(self):
        print("Indexation RIMES")
        csv_output = self.rimes_dir / "metadata.csv"

        if csv_output.exists():
            print("Le fichier metadata.csv existe deja.")
            return

        sets_dirs = list(self.rimes_dir.rglob("Sets"))
        trans_dirs = list(self.rimes_dir.rglob("Transcriptions"))
        imgs_dirs = list(self.rimes_dir.rglob("Images"))

        if not (sets_dirs and trans_dirs and imgs_dirs):
            print("Erreur : Dossiers Sets/Transcriptions/Images introuvables dans data/rimes.")
            return

        sets_d, trans_d, imgs_d = sets_dirs[0], trans_dirs[0], imgs_dirs[0]
        print(f"Dossiers detectes dans : {sets_d.parent.name}")

        files_map = {
            "train": ["TrainLines.txt", "train.txt"],
            "val":   ["ValidationLines.txt", "val.txt"],
            "test":  ["TestLines.txt", "test.txt"]
        }

        data = []
        
        for split, filenames in files_map.items():
            for fname in filenames:
                fpath = sets_d / fname
                if fpath.exists():
                    print(f"Traitement de {fname} ({split})...")
                    with open(fpath, 'r', encoding='utf-8') as f:
                        ids = [l.strip() for l in f.readlines() if l.strip()]
                    
                    for fid in ids:
                        txt_file = trans_d / f"{fid}.txt"
                        img_file = imgs_d / f"{fid}.jpg"
                        if not img_file.exists():
                            img_file = imgs_d / f"{fid}.png"
                        
                        if txt_file.exists() and img_file.exists():
                            try:
                                with open(txt_file, 'r', encoding='utf-8') as tf:
                                    label = tf.read().strip()
                                if label:
                                    data.append({
                                        "image_path": str(img_file),
                                        "label": label,
                                        "split": split
                                    })
                            except: pass

        if data:
            df = pd.DataFrame(data)
            df.to_csv(csv_output, index=False)
            print(f"Succes : metadata.csv genere avec {len(df)} images.")
        else:
            print("Erreur : Aucune donnee valide trouvee.")

if __name__ == "__main__":
    manager = DatasetManager()
    manager.prepare_rimes()