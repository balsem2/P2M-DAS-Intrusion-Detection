# DAS CNN Classification

Projet de classification de signaux DAS avec un modèle CNN 2D.

## Classes

- car
- construction
- fence
- longboard
- manipulation
- openclose
- regular
- running
- walk

## Fichiers importants

- `best_cnn_das_2d.keras` : modèle CNN entraîné.
- `classes_das_2d.npy` : noms des classes du modèle.
- `extract_dataset_2d.py` : extraction des features depuis les fichiers DAS.
- `train_from_extracted_2d.py` : entraînement du CNN.
- `voir_accuracy.py` : évaluation du modèle sauvegardé.
- `open_model.py` : affichage de l'architecture du modèle.

## Résultat

Accuracy test :

```text
94.45%
```

## Utilisation

Installer les dépendances :

```powershell
pip install -r requirements.txt
```

Afficher le résumé du modèle :

```powershell
py -3.10 open_model.py
```

Afficher l'accuracy :

```powershell
py -3.10 voir_accuracy.py
```

## Note sur les données

Le dataset brut `data/` et le fichier `X_das_2d.npy` ne sont pas inclus dans GitHub car ils sont volumineux.

Pour utiliser directement le modèle dans un backend, les fichiers minimum sont :

```text
best_cnn_das_2d.keras
classes_das_2d.npy
```
