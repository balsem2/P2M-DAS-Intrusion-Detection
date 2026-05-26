# P2M-DAS-Intrusion-Detection

AI-based intrusion detection using Distributed Acoustic Sensing (DAS).

This project uses a 2D CNN model to classify DAS signal events.

## Objective

- Load and preprocess DAS signals
- Extract FFT-based 2D features
- Classify events with a CNN model
- Support backend integration for visualization applications

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

## Important Files

- `best_cnn_das_2d.keras`: trained CNN model.
- `classes_das_2d.npy`: model class names.
- `extract_dataset_2d.py`: feature extraction from DAS files.
- `train_from_extracted_2d.py`: CNN training script.
- `voir_accuracy.py`: evaluates the saved model.
- `open_model.py`: displays the model architecture.

## Result

Test accuracy:

```text
94.45%
```

## Usage

Install dependencies:

```powershell
pip install -r requirements.txt
```

Display model summary:

```powershell
py -3.10 open_model.py
```

Evaluate model accuracy:

```powershell
py -3.10 voir_accuracy.py
```

## Backend Integration

The trained model files are available in GitHub Releases:

```text
https://github.com/balsem2/P2M-DAS-Intrusion-Detection/releases/tag/v1.0
```

Minimum files needed to use the CNN in a backend:

```text
best_cnn_das_2d_backend.keras
classes_das_2d.npy
```

The full trained model is also available in the same release:

```text
best_cnn_das_2d.keras
```

Expected model input shape:

```text
(1, 17, 1024, 1)
```

## Dataset

The raw dataset `data/` and large extracted arrays such as `X_das_2d.npy` are not included in this repository because of their size.
