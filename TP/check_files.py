from pathlib import Path

# On regarde dans data/rimes
p = Path("data/rimes")

print(f"--- Scan de {p.resolve()} ---")
files = list(p.rglob("*"))

# Affiche les 20 premiers fichiers trouvés
for f in files[:20]:
    print(f)

print(f"\nTotal fichiers trouvés : {len(files)}")

# Cherche spécifiquement des fichiers texte ou xml ou csv
print("\n--- Fichiers potentiels d'annotations ---")
for ext in ['*.xml', '*.txt', '*.csv', '*.json']:
    for f in p.rglob(ext):
        if "metadata.csv" not in f.name: # On ignore celui qu'on a créé nous-même
            print(f"TROUVÉ : {f.name} (Chemin: {f})")