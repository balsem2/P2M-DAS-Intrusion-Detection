# P2M-DAS-Intrusion-Detection

AI-based intrusion detection using Distributed Acoustic Sensing (DAS).

This repository contains:

- `CNN-Model/`: 2D CNN model for DAS event classification.
- `LSTM-Model/`: LSTM model files.
- `DAS_Interface/`: visualization/backend interface.
- `Data_Exploration/`: data exploration files.

## CNN Model

The CNN model classifies DAS events into:

- car
- construction
- fence
- longboard
- manipulation
- openclose
- regular
- running
- walk

Test accuracy:

```text
94.45%
```

## Trained CNN Files

The trained Keras model files are available in GitHub Releases:

```text
https://github.com/balsem2/P2M-DAS-Intrusion-Detection/releases/tag/v1.0
```

For backend integration, use:

```text
best_cnn_das_2d_backend.keras
CNN-Model/classes_das_2d.npy
```

The full trained model is also available:

```text
best_cnn_das_2d.keras
```

Expected CNN input shape:

```text
(1, 17, 1024, 1)
```

## Notes

The raw DAS dataset and large extracted arrays are not included in the repository because of their size.
