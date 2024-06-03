import pandas as pd
from sklearn.model_selection import train_test_split

# Charger le dataset
df = pd.read_csv('matches.csv')

# Supprimer les colonnes inutiles
columns_to_drop = ['numero', 'Date', 'HTFormPtsStr', 'ATFormPtsStr']
df = df.drop(columns=columns_to_drop)

# Vérifier s'il y a des valeurs manquantes et les traiter si nécessaire
# Par exemple, pour l'imputation des valeurs manquantes avec la moyenne
# df.fillna(df.mean(), inplace=True)

# Séparer les données en caractéristiques (features) et variable cible
X = df.drop(['FTR'], axis=1)
y = df['FTR']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarder les ensembles d'entraînement et de test
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
