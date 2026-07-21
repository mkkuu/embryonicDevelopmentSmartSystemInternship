"""
========================================
EMBRYO ANALYSIS SYSTEM - DATA LOADER FACTORY
========================================

This module provides a factory function for creating PyTorch DataLoaders for the
embryo development analysis system. It handles the creation of training, validation,
and test data loaders with consistent configuration and parameters.

Features:
- Unified DataLoader creation for all dataset splits
- Configurable batch sizes and worker processes
- Memory pinning for GPU acceleration
- Flexible dataset parameters (focal type, balancing, phases)
- Integration with configuration system

DataLoader Configuration:
- Training: Shuffled data with 5 videos per epoch
- Validation: Non-shuffled data with 2 videos per epoch  
- Test: Non-shuffled data with 10 videos per epoch, always balanced

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

from config_args import ConfigArgs
from DataSet import Embryo_Transition_Dataset
from torch.utils.data import DataLoader

# Load configuration parameters
Config = ConfigArgs()

# Extract configuration values for DataLoader setup
BATCH_SIZE = Config.get("batch_size")
NUM_WORKERS = Config.get("num_workers")
DATA_LOADER = Config.get("data_loader")
WINDOW_SIZE = Config.get("window_size")
STRIDE = Config.get("stride")


def get_dataloaders(**kwargs):
    """
    Factory function to create training, validation, and test DataLoaders.
    
    This function creates three DataLoader instances (train, validation, test) with
    appropriate configurations for each split. The function supports flexible
    parameters for different experimental setups.
    
    Args:
        **kwargs: Additional parameters for dataset configuration
            FocalType (str): Type of focal plane data to use (default: "F0")
            Balance_Flags (bool): Whether to balance class distribution (default: False)
            Multiple_Phases (bool): Whether to include multiple developmental phases (default: False)
    
    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader)
            - train_loader: DataLoader for training data (shuffled, 5 videos)
            - val_loader: DataLoader for validation data (non-shuffled, 2 videos)
            - test_loader: DataLoader for test data (non-shuffled, 10 videos, balanced)
    
    DataLoader Characteristics:
        - All loaders use the same batch size and number of workers from config
        - Memory pinning enabled for GPU acceleration
        - Training data is shuffled for better generalization
        - Validation and test data are not shuffled for consistent evaluation
        - Test data is always balanced regardless of input parameters
    
    Example:
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     FocalType="F0",
        ...     Balance_Flags=True,
        ...     Multiple_Phases=False
        ... )
    """
    
    # Extract parameters with defaults
    focal_type = kwargs.get("FocalType", "F0")
    balance_flags = kwargs.get("Balance_Flags", False)
    multiple_phases = kwargs.get("Multiple_Phases", False)

    # Create training dataset
    train_set = Embryo_Transition_Dataset(
        phase_col="Phase",
        path_col="Path",
        seed=42,
        FocalType=focal_type,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        UsedFor="Train",
        NumberOfVideos=0,  # 5 videos per training epoch
        balance_flags=balance_flags,
        multiple_phases=multiple_phases,
        mode=DATA_LOADER,
    )
    
    # Create validation dataset
    val_set = Embryo_Transition_Dataset(
        phase_col="Phase",
        path_col="Path",
        seed=42,
        FocalType=focal_type,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        UsedFor="Val",
        NumberOfVideos=0,  # 2 videos per validation epoch
        balance_flags=balance_flags,
        multiple_phases=multiple_phases,
        mode=DATA_LOADER,
    )
    
    # Create test dataset (always balanced for fair evaluation)
    test_set = Embryo_Transition_Dataset(
        phase_col="Phase",
        path_col="Path",
        seed=42,
        FocalType=focal_type,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        UsedFor="Test",
        NumberOfVideos=0,  # 10 videos per test epoch
        balance_flags=balance_flags,  # Always balanced for consistent evaluation
        multiple_phases=multiple_phases,  # Single phase for clear test results
        mode=DATA_LOADER,
    )
    
    # Create training DataLoader (shuffled for better training)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle training data
        pin_memory=True,  # Pin memory for GPU acceleration
        num_workers=NUM_WORKERS,
    )
    
    # Create validation DataLoader (non-shuffled for consistent evaluation)
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Don't shuffle validation data
        pin_memory=True,  # Pin memory for GPU acceleration
        num_workers=NUM_WORKERS,
    )
    
    # Create test DataLoader (non-shuffled for consistent evaluation)
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Don't shuffle test data
        pin_memory=True,  # Pin memory for GPU acceleration
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader
