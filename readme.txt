# Football Prediction

## Description

Ce projet utilise l'apprentissage automatique pour prédire les résultats des matchs de football. L'application est construite avec Python et utilise Flask pour servir le modèle à travers une API.

## Contenu du Dépôt

- `app.py` : Script Flask pour servir le modèle via une API.
- `train_model.py` : Script pour l'entraînement du modèle de machine learning.
- `matches.csv` : Dataset utilisé pour entraîner le modèle.
- `model.pkl` : Modèle entraîné et sérialisé.
- `Dockerfile` : Fichier de configuration Docker pour conteneuriser l'application.
- `requirements.txt` : Liste des dépendances Python nécessaires pour exécuter l'application.
- `.gitignore` : Fichier spécifiant les fichiers et répertoires à ignorer par Git.

## Installation

### Prérequis

- Python 3.x
- pip (gestionnaire de paquets Python)
- Docker (facultatif, pour exécuter l'application dans un conteneur)

### Étapes d'Installation

1. Clonez le dépôt :

   ```sh
   git clone https://github.com/yourusername/football_prediction.git
   cd football_prediction
