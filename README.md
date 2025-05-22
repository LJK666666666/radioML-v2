# RadioML

This repository contains machine learning models for radio modulation classification.

## Important Note

**This repository only contains the source code for the models. The dataset and pre-trained models are not included due to their large size.**

## Project Structure

- `src/`: Source code for all models
  - Models implementations (CNN 1D, CNN 2D, Complex NN, ResNet, Transformer)
  - Utility functions and callbacks

## Missing Files

The following files are not included in this repository due to GitHub's file size limitations:

- `RML2016.10a_dict.pkl`: The RadioML dataset (> 600MB)
- Model weights in `model_weight_saved/` and `output/models/` directories
- Output plots and evaluation results

## Getting Started

To use this code, you will need to:

1. Download the RadioML dataset (RML2016.10a) from the official website
2. Place the dataset file in the repository root
3. Run the training scripts to train the models or use pre-trained weights

## Models

The repository includes implementations of various models for radio modulation classification:

- CNN 1D: One-dimensional convolutional neural network
- CNN 2D: Two-dimensional convolutional neural network 
- Complex NN: Neural network with complex-valued operations
- ResNet: Residual network architecture
- Transformer: Attention-based transformer model