# P2M-DAS-Intrusion-Detection

Projet P2M : système de détection, prévision et visualisation d'intrusions à partir de signaux DAS
(*Distributed Acoustic Sensing*).

Le projet combine trois blocs principaux :

- un modèle **CNN 2D** pour classifier les événements DAS ;
- un modèle **LSTM** pour prévoir l'évolution temporelle du signal ;
- une **application web React** pour visualiser la fibre, les alertes et les courbes de signal.

## Structure du projet

```text
CNN-Model/          Modèle CNN 2D organisé en préparation, modèle, résultats et modèle entraîné
LSTM-Model/         Préparation des données actives et modèle LSTM de prévision
DAS_Interface/      Application web React de visualisation
Data_Exploration/   Fichiers et scripts d'exploration du dataset
Presentation/       Présentation PDF du projet
rapport_technique.tex
```

## CNN : classification d'événements

Le CNN reçoit des patches de forme :

```text
(17, 1024, 1)
```

Il classe les événements parmi 9 classes :

```text
car, construction, fence, longboard, manipulation,
openclose, regular, running, walk
```

Résultat obtenu sur le test set :

```text
Test accuracy : 94.45 %
Test loss     : 0.1484
```

Fichiers importants :

```text
CNN-Model/Data_Preparation/extract_dataset_2d.py
CNN-Model/Model/train_from_extracted_2d.py
CNN-Model/Trained_Model/classes_das_2d.npy
CNN-Model/Results/accuracy_result.txt
```

## LSTM : prévision temporelle

Le modèle LSTM prédit l'évolution future du signal DAS. Il utilise :

```text
Entrée : 8192 points sur 8 canaux
Sortie : 1024 points futurs sur 8 canaux
```

Le modèle sauvegardé est :

```text
LSTM-Model/Trained_Model/best_lstm_forecast.keras
```

## Application web

L'interface se trouve dans :

```text
DAS_Interface/das-monitor/
```

Elle permet de visualiser :

- une page d'authentification ;
- une carte de la fibre avec 166 segments ;
- les alertes par couleur : normal, alerte, danger ;
- l'événement prédit par le CNN ;
- les courbes de signal actuel et de prédiction LSTM.

Pour lancer l'application :

```powershell
cd DAS_Interface/das-monitor
npm install
npm start
```

URL locale :

```text
http://localhost:3000
```

Sur Windows, le fichier suivant peut aussi être utilisé :

```text
DAS_Interface/das-monitor/ouvrir_application.bat
```

## Modèles entraînés CNN

Les modèles CNN entraînés sont disponibles dans la release GitHub :

```text
https://github.com/balsem2/P2M-DAS-Intrusion-Detection/releases/tag/v1.0
```

Pour l'intégration backend, utiliser :

```text
best_cnn_das_2d_backend.keras
CNN-Model/Trained_Model/classes_das_2d.npy
```

Le modèle complet est aussi disponible :

```text
best_cnn_das_2d.keras
```

## Présentation et rapport

Présentation du projet :

```text
Presentation/DAS_System_VF.pdf
```

Rapport technique LaTeX :

```text
rapport_technique.tex
```

## Notes

Le dataset brut `data/` et les gros tableaux extraits comme `X_das_2d.npy` ne sont pas inclus dans GitHub
à cause de leur taille.
