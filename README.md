# CIFAR-100 Classification with Wide ResNet

This repository implements an image classification model on the CIFAR-100 dataset using a Wide ResNet architecture. The main training and testing procedures are implemented in `main.py`.

## Repository Structure

```plaintext
.
├── data/             # CIFAR-100 dataset will be downloaded here.
├── main.py           # Main training and testing script.
├── networks/         # Contains model definitions (Wide ResNet).
│   ├── wide_resnet.py      
│   └── __init__.py
└── results/          # Folder where result plots (loss and accuracy curves) are saved.
```

## Installation

### Prerequisites

- Python 3.x
- pip

### Dependencies

Install the required Python packages with:

```sh
pip install torch torchvision matplotlib
```

Make sure your [PyTorch](https://pytorch.org/) installation matches your system's CUDA configuration if GPU acceleration is desired.

## Usage

To train and evaluate the model, run:

```sh
python main.py
```

After training, the loss and accuracy convergence plots will be saved in the `results` folder.
