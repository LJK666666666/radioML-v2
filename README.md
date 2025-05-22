# RadioML

This repository contains machine learning models for radio modulation classification.

## Important Note

**This repository includes all source code, models, and data files. Large files are managed using Git LFS.**

## Project Structure

- `src/`: Source code for all models
  - Models implementations (CNN 1D, CNN 2D, Complex NN, ResNet, Transformer)
  - Utility functions and callbacks
- `model_weight_saved/`: Saved model weights (managed with Git LFS)
- `output/models/`: Output model files (managed with Git LFS)
- `RML2016.10a_dict.pkl`: The RadioML dataset (managed with Git LFS)
- `projects/`: Contains submodules of related projects
  - `AMC-Net`: Implementation of the AMC-Net architecture
  - `ULCNN`: Implementation of the ULCNN architecture

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