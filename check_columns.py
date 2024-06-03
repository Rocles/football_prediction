import pandas as pd

# Chemin vers le fichier CSV
file_path = 'matches.csv'

# Charger le dataset
df = pd.read_csv(file_path)

# Afficher les colonnes du dataset
print(df.columns)
