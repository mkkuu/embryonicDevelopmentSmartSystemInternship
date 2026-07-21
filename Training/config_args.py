"""
========================================
EMBRYO ANALYSIS SYSTEM - CONFIGURATION ARGUMENTS
========================================

This module provides a comprehensive configuration management system for the
Embryo Analysis System training pipeline. It handles both configuration file
and command-line argument parsing with a hierarchical override system.

Features:
- Configuration file parsing (config.ini)
- Command-line argument parsing
- Hierarchical configuration (CLI overrides config file)
- Type-safe parameter handling
- Default value management
- Easy parameter retrieval

Configuration Hierarchy:
1. Default values (hardcoded fallbacks)
2. Configuration file values (config.ini)
3. Command-line arguments (highest priority)

Supported Parameters:
- Data parameters: multi_cpu, data_loader, window_size, stride
- Model parameters: model_name, pretrained, num_classes
- Training parameters: batch_size, epochs, learning_rate, num_workers, image_size

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import argparse
import configparser


class ConfigArgs:
    """
    Configuration management class that handles both config file and CLI arguments.
    
    This class provides a unified interface for accessing configuration parameters
    with proper type handling and hierarchical override support.
    
    Attributes:
        config (configparser.ConfigParser): Configuration file parser
        defaults (dict): Final configuration values after all overrides
        args (dict): Parsed command-line arguments
    
    Example:
        >>> config = ConfigArgs("path/to/config.ini")
        >>> batch_size = config.get("batch_size")
        >>> print(config)  # Display all configuration values
    """
    
    def __init__(self, config_path="Configs\config.ini"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the configuration file (default: "Configs\config.ini")
        
        The initialization process:
        1. Reads configuration file
        2. Sets up default values from config file
        3. Parses command-line arguments
        4. Applies CLI overrides to configuration values
        """
        # Initialize the config parser and read the config file
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # Load default values from the config file
        self.defaults = {
            # Data parameters
            "multi_cpu": self.config.getboolean("data", "multi_cpu", fallback=True),
            "data_loader": self.config.get("data", "data_loader", fallback="image_seq"),
            "window_size": self.config.getint("data", "window_size", fallback=8),
            "stride": self.config.getint("data", "stride", fallback=1),
            # Model parameters
            "model_name": self.config.get("model", "name", fallback="resnet18"),
            "pretrained": self.config.getboolean("model", "pretrained", fallback=True),
            # Training parameters
            "batch_size": self.config.getint("training", "batch_size", fallback=16),
            "epochs": self.config.getint("training", "epochs", fallback=100),
            "learning_rate": self.config.getfloat(
                "training", "learning_rate", fallback=0.001
            ),
            "num_workers": self.config.getint("training", "num_workers", fallback=8),
            "image_size": self.config.getint("training", "image_size", fallback=224),
        }

        # Parse command-line arguments (CLI args overwrite config.ini)
        self.args = self.parse_args()

        # Use the parsed CLI args to overwrite defaults if present
        for key, value in self.args.items():
            if value is not None:
                self.defaults[key] = value

    def parse_args(self):
        """
        Parse command-line arguments using argparse.
        
        Returns:
            dict: Dictionary of parsed command-line arguments
            
        This method sets up argument parsing for all supported parameters
        with appropriate types and help descriptions.
        """
        # Set up command-line argument parser
        parser = argparse.ArgumentParser(description="Model and Training Configuration")

        # Data parameters
        parser.add_argument(
            "--multi_cpu",
            type=lambda x: x.lower() == "true",
            help="Use Multi Processing (True/False)",
        )
        parser.add_argument("--data_loader", type=str, help="Use DataLoader")
        parser.add_argument(
            "--window_size", type=int, help="Window size for image sequences"
        )
        parser.add_argument("--stride", type=int, help="Stride for image sequences")
        # Model parameters
        parser.add_argument(
            "--model_name",
            type=str,
            help="Model architecture (resnet18, resnet50, etc.)",
        )
        parser.add_argument("--num_classes", type=int, help="Number of output classes")
        parser.add_argument(
            "--pretrained",
            type=lambda x: x.lower() == "true",
            help="Use pretrained weights (True/False)",
        )

        # Training parameters
        parser.add_argument("--batch_size", type=int, help="Batch size")
        parser.add_argument("--epochs", type=int, help="Number of training epochs")
        parser.add_argument("--learning_rate", type=float, help="Learning rate")
        parser.add_argument(
            "--multi_gpu",
            type=lambda x: x.lower() == "false",
            help="Use multiple gpu (True/False)",
        )
        parser.add_argument(
            "--num_workers", type=int, help="Number of workers for data loading"
        )
        parser.add_argument(
            "--image_size", type=int, help="Input image size for the model"
        )

        # Parse arguments
        args = parser.parse_args()

        # Convert args to a dictionary and return it
        return vars(args)

    def get(self, key):
        """
        Retrieve a specific configuration value.
        
        Args:
            key (str): Configuration parameter name
            
        Returns:
            Any: Configuration value or None if not found
            
        Example:
            >>> config = ConfigArgs()
            >>> batch_size = config.get("batch_size")
        """
        return self.defaults.get(key, None)

    def __repr__(self):
        """
        String representation of the configuration.
        
        Returns:
            str: String representation of all configuration values
            
        This method provides a convenient way to display all configuration
        parameters for debugging and verification purposes.
        """
        return str(self.defaults)
