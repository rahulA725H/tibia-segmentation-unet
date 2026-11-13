# Tibia Segmentation from Knee X-Ray Images Using U-Net

A deep-learning project for automatic tibia segmentation from knee X-ray images using a U-Net convolutional neural network implemented in PyTorch.

## Project Overview

The objective of this project is to identify and segment the tibia bone in grayscale knee radiographs. The model generates a binary segmentation mask indicating the predicted tibia region.

## Dataset

The dataset contains 150 knee X-ray images, including 100 manually annotated images and 50 unlabeled images.

| Dataset | Images | Ground-truth masks |
|---|---:|---|
| Training | 80 | Yes |
| Validation | 20 | Yes |
| Testing | 50 | No |
| **Total** | **150** | **100** |

The 100 labeled images were split into 80 training and 20 validation images. The 50 unlabeled images were reserved for prediction.

Images and masks are resized to 256 × 256 pixels.

## U-Net Architecture

The U-Net architecture includes an encoder, bottleneck, decoder, and skip connections.

- Four downsampling stages
- Convolutional blocks with batch normalization and ReLU
- Bilinear upsampling in the decoder
- Skip connections to preserve spatial information
- Single-channel output for binary segmentation

The model contains **1,963,809 trainable parameters**.

## Training

The model was trained using PyTorch with:

- AdamW optimizer
- Combined binary cross-entropy and soft Dice loss
- Learning-rate scheduling
- Data augmentation for training images
- Best-checkpoint saving based on validation Dice

The final training run completed **60 epochs**, with the best model selected at **epoch 46**.

To train:

```bash
python main.py -epochs 60 -bs 10 -out_dir sessions/final
```

## Evaluation Results

The best checkpoint was evaluated using the 20 manually annotated validation images.

| Metric | Result |
|---|---:|
| Mean Dice coefficient | **97.11%** |
| Mean IoU | **94.44%** |
| Mean precision | **97.09%** |

These are validation metrics. The 50 unlabeled test images do not have ground-truth masks, so test Dice cannot be calculated.

To evaluate:

```bash
python evaluate.py
```

## Prediction Results

The trained model generated binary tibia masks for all 50 unlabeled test images.

The project also produces validation comparisons showing the original X-ray, ground-truth mask, model prediction, and segmentation overlay.

Outputs are stored in `demo_images/final/`.

## Project Structure

```text
Knee_Segmentation/
├── data/
│   ├── xrays/
│   ├── masks/
│   └── CSVs/
├── sessions/
│   └── final/
├── demo_images/
│   └── final/
├── args.py
├── data_preparation.py
├── dataset.py
├── evaluate.py
├── main.py
├── model.py
├── trainer.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Install the required Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Limitations

The dataset is relatively small, and quantitative evaluation was performed on only 20 validation images. Further evaluation using independently annotated test images would be necessary to assess generalization.

## Disclaimer

This project is intended for academic and educational purposes and has not been validated for clinical use.