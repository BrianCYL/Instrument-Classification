# Music Instrument Classification

This repository implements multi-label music instrument classification using
the Slakh dataset. The model uses the `m-a-p/MERT-v1-95M` pretrained audio
feature extractor and a neural network classifier to predict 9 instrument
classes for each audio segment.

## Setup

Install the dependencies listed by running:

```bash
pip install -r requirements.txt
pip install transformers
```

The code uses the pretrained
[`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M) model. The
first run downloads its weights from Hugging Face, so internet access is
required.

Place the Slakh data under `./slakh/` with these directories and label files, the dataset can be downloaded from [here](https://drive.google.com/file/d/12PDfVYclzjm0BYkeBKHm6bcVyvh-Mzu9/view?usp=sharing):

```text
slakh/
├── train/*.npy
├── validation/*.npy
├── test/*.npy
├── train_labels.json
├── validation_labels.json
└── test_labels.json
```

Please note that:

- Each `.npy` file contains an audio segment.
- Each label JSON maps an `.npy` filename to its 9-element multi-hot label.
- Run all commands from the `Instrument-Classification/` directory.
- If only needing to perform inference, one can download model checkpoints from this [Google Drive](https://drive.google.com/drive/folders/17G8kVoPSDwBx-NPmPj58Z8SNJGE_3ldd?usp=sharing)

## Preprocessing

Combine the raw audio segments and labels into NumPy files used by the
training script:

```bash
python code/preprocessing.py
```

This creates:

```text
dataset/
├── train_data.npy
├── train_labels.npy
├── validation_data.npy
├── validation_labels.npy
├── test_data.npy
└── test_labels.npy
```

## Training

Train the classifier with the default settings: 50 epochs, batch size 16,
learning rate `3e-5`, fine-tuning enabled, class balancing enabled, and early
stopping after 30 epochs without validation improvement:

```bash
python code/train.py
```

The best validation-loss checkpoint is saved to
`./checkpoint/model_best_fine_tune_balance.pth`.

Additional options can be changed from the command line, for example:

```bash
python code/train.py \
  --n_epochs 50 \
  --batch_size 16 \
  --lr 3e-5 \
  --fine_tune 1 \
  --balance_data 1 \
  --early_stop 30 \
  --folder_path ./dataset
```

Set `--fine_tune 0` to freeze the MERT feature extractor and train only the
classifier. Set `--balance_data 0` to use an unweighted
`BCEWithLogitsLoss`. The checkpoint name is determined by these two options:

| `fine_tune` | `balance_data` | checkpoint |
|---:|---:|---|
| 0 | 0 | `model_best.pth` |
| 0 | 1 | `model_best_balance.pth` |
| 1 | 0 | `model_best_fine_tune.pth` |
| 1 | 1 | `model_best_fine_tune_balance.pth` |

To save training and validation loss curves, leave `--tracking` enabled. The
script saves the curve as `./loss_tracking*.pdf`.

## Inference

Run inference on the test split with the checkpoint produced by training:

```bash
python code/inference.py \
  --folder_path ./slakh \
  --threshold 0.5 \
  --fine_tune 1 \
  --balance_data 1
```

The script loads
`./checkpoint/model_best_fine_tune_balance.pth`, applies a sigmoid threshold
of `0.5`, prints a classification report, and saves the predictions to:

```text
result/
├── preds_fine_tune_balance.npy
├── prob_preds_fine_tune_balance.npy
├── test_labels_fine_tune_balance.npy
└── pred_tracks_fine_tune_balance.json
```

For a checkpoint trained without fine-tuning or class balancing, use matching
options during inference:

```bash
python code/inference.py \
  --folder_path ./slakh \
  --threshold 0.5 \
  --fine_tune 0 \
  --balance_data 0
```

The `--threshold` option controls the multi-label decision boundary. Lower
values generally increase recall, while higher values generally increase
precision.
