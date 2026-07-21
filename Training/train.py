"""
========================================
EMBRYO ANALYSIS SYSTEM - TRAINING SCRIPT
========================================

This is the main training script for the embryo development analysis system.
It orchestrates the complete training pipeline including model initialization,
data loading, training, validation, and testing.

Features:
- Configuration-driven training setup
- Automatic device detection (CUDA/CPU)
- Model architecture selection (ResNet, TimeSformer)
- Data augmentation and preprocessing
- Training with validation monitoring
- Best model checkpointing and testing

Training Pipeline:
1. Load configuration parameters
2. Initialize model and optimizer
3. Setup data loaders with transforms
4. Train model with validation monitoring
5. Load best model and evaluate on test set

Usage:
    python train.py [--model_name resnet18] [--epochs 100] [--learning_rate 0.001]

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import os

import torch
import torch.optim as optim
from config_args import ConfigArgs
from Load_data import get_dataloaders
from ModelBuilder import get_model
from torchvision import transforms
from train_val_test_pipline import evaluate, train_model

if __name__ == "__main__":
    """
    Main training execution block.
    
    This script performs the complete training pipeline for embryo development
    analysis models, including configuration loading, model setup, training,
    and evaluation.
    """
    
    # Load configuration from config file and command line arguments
    config = ConfigArgs()

    # Extract training parameters from configuration
    num_epochs = config.get("epochs")
    learning_rate = config.get("learning_rate")
    image_size = config.get("image_size")
    model_name = config.get("model_name")
    window_size = config.get("window_size")
    pretrained = config.get("pretrained")
    data_loader = config.get("data_loader")

    # Detect available device (CUDA GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training using {model_name} for {window_size} frames with {num_epochs} epochs, on {device}"
    )

    # Initialize model and move to device
    model = get_model(model_name, preTrained=pretrained)
    model = model.to(device)

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Define data augmentation and preprocessing transforms
    transform = transforms.Compose(
        [
            transforms.Resize((256)),  # Resize to 256px
            transforms.CenterCrop((image_size, image_size)),  # Center crop to target size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ]
    )

    # Create data loaders for training, validation, and testing
    train_loader, val_loader, test_loader = get_dataloaders(Transform=transform)

    # Train the model with validation monitoring
    train_model(
        model, train_loader, val_loader, criterion, optimizer, model_name, num_epochs
    )

    print("Training completed.")
    print(f"Now testing the model {model_name} with the highest val accuracy")
    
    # Load the best model checkpoint and evaluate on test set
    best_model_path = os.path.join("..", "Results", f"{model_name}", "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    evaluate(model, test_loader, model_name)
