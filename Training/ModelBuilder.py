"""
========================================
EMBRYO ANALYSIS SYSTEM - MODEL BUILDER
========================================

This module provides a unified interface for creating and configuring deep learning
models for embryo development analysis. It supports both traditional CNN architectures
and state-of-the-art transformer-based models for spatio-temporal analysis.

Features:
- Support for multiple model architectures (ResNet, TimeSformer)
- Pretrained model loading and fine-tuning
- Custom wrapper for video classification models
- Configurable input channels and output classes
- Integration with configuration system

Supported Models:
- ResNet18: Traditional CNN for image sequence classification
- TimeSformer: Transformer-based model for video classification

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import warnings

import timm
import torch.nn as nn
from config_args import ConfigArgs
from transformers import TimesformerForVideoClassification

# Suppress deprecated weight warnings for cleaner output
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*weights.*deprecated.*"
)

# Load configuration for model parameters
config = ConfigArgs()
WINDOW_SIZE = config.get("window_size")


class TimeSformerWrapper(nn.Module):
    """
    Wrapper class for TimeSformer model adapted for embryo development classification.
    
    This class provides a custom wrapper around the Hugging Face TimeSformer model,
    specifically configured for binary classification of embryo development stages.
    
    Attributes:
        model: The underlying TimeSformer model from Hugging Face
        
    Example:
        >>> wrapper = TimeSformerWrapper(pretrained=True)
        >>> output = wrapper(input_tensor)
    """
    
    def __init__(
        self, pretrained=True, model_name="facebook/timesformer-base-finetuned-k600"
    ):
        """
        Initialize the TimeSformer wrapper.
        
        Args:
            pretrained (bool): Whether to load pretrained weights (default: True)
            model_name (str): Hugging Face model identifier (default: facebook/timesformer-base-finetuned-k600)
        
        The model is automatically configured for binary classification (2 classes)
        representing different embryo development stages.
        """
        super().__init__()
        
        if pretrained:
            # Load pretrained model from Hugging Face
            self.model = TimesformerForVideoClassification.from_pretrained(model_name)
        else:
            # Initialize model from scratch with default configuration
            self.model = TimesformerForVideoClassification.from_config(
                TimesformerForVideoClassification.config_class()
            )

        # Modify classifier for binary classification (Transition vs No Transition)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)

    def forward(self, x, **kwargs):
        """
        Forward pass through the TimeSformer model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, frames, height, width)
            **kwargs: Additional arguments passed to the model
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, 2)
        """
        return self.model(x, **kwargs).logits


def get_model(model_name, preTrained=True, **kwargs):
    """
    Factory function to create model instances based on the specified architecture.
    
    This function provides a unified interface for creating different model architectures
    with consistent configuration and parameter handling.
    
    Args:
        model_name (str): Name of the model architecture to create
        preTrained (bool): Whether to use pretrained weights (default: True)
        **kwargs: Additional keyword arguments passed to model constructors
        
    Returns:
        torch.nn.Module: Configured model instance
        
    Raises:
        ValueError: If the specified model_name is not supported
        
    Supported Models:
        - "resnet18": ResNet-18 CNN for image sequence classification
        - "timesformer": TimeSformer transformer for video classification
        
    Example:
        >>> model = get_model("resnet18", preTrained=True)
        >>> model = get_model("timesformer", preTrained=False)
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        # Create ResNet-18 model with custom input channels for image sequences
        model = timm.create_model(
            "resnet18", 
            pretrained=preTrained, 
            num_classes=2,  # Binary classification for embryo stages
            in_chans=WINDOW_SIZE  # Custom input channels based on window size
        )

    elif model_name == "timesformer":
        # Create TimeSformer wrapper for video classification
        model = TimeSformerWrapper(pretrained=preTrained)

    else:
        raise ValueError(f"Model {model_name} not supported. Available models: resnet18, timesformer")

    return model
