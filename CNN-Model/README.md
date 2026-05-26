# CNN Model

Ce dossier contient la partie CNN du projet P2M pour la classification des événements DAS.

## Organisation

```text
CNN-Model/
├── Data_Preparation/
│   └── extract_dataset_2d.py
├── Model/
│   ├── train_from_extracted_2d.py
│   ├── voir_accuracy.py
│   ├── open_model.py
│   └── make_plots.py
├── Trained_Model/
│   ├── classes_das_2d.npy
│   ├── best_cnn_das_2d.keras
│   └── best_cnn_das_2d_backend.keras
├── Results/
│   ├── accuracy_result.txt
│   └── model_summary.txt
├── requirements.txt
├── ouvrir_modele.bat
└── voir_accuracy.bat
```

Les fichiers `.keras` sont disponibles dans GitHub Releases :

```text
https://github.com/balsem2/P2M-DAS-Intrusion-Detection/releases/tag/v1.0
```

## Classes

```text
car, construction, fence, longboard, manipulation,
openclose, regular, running, walk
```

## Résultat CNN

```text
Test accuracy : 94.45 %
Test loss     : 0.1484
```

## Utilisation

Extraire les données :

```powershell
py -3.10 Data_Preparation\extract_dataset_2d.py
```

Entraîner le CNN :

```powershell
py -3.10 Model\train_from_extracted_2d.py
```

Voir l'architecture :

```powershell
py -3.10 Model\open_model.py
```

Voir l'accuracy :

```powershell
py -3.10 Model\voir_accuracy.py
```

Pour le backend, utiliser :

```text
Trained_Model/best_cnn_das_2d_backend.keras
Trained_Model/classes_das_2d.npy
```
