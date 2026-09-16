"""
EMBRYO ANALYSIS SYSTEM - CLASS-BALANCED TRAINING ENTRY POINT
==============================================================

Composition-only variant of train.py: fixes the ~6:1 class imbalance between
"No Transition" and "Transition" windows (see docs/SESSION_STATE.md diagnostic,
2026-07-24) by turning on `Embryo_Transition_Dataset`'s existing `balance_flags`
option (DataSet.py, unused by any default call path) instead of touching
train.py / Load_data.py / DataSet.py / train_val_test_pipline.py, all of which
stay untouched per this project's composition-over-modification rule.

Train/val loaders are built balanced (1:1 undersampling of the majority class)
so both gradient updates and best-checkpoint selection (val accuracy) are not
dominated by the majority class. The *test* loader is deliberately built
UNBALANCED (real-world ~6:1 distribution) so its Classification_Report is
directly comparable to Results/resnet18/Classification_Report_resnet18.txt
from the original run.

Usage (same CLI flags as train.py):
    python train_balanced.py --model_name resnet18 --batch_size 16 --epochs 50 --learning_rate 0.0001

Results are written to Results/{model_name}_balanced/ — the original
Results/{model_name}/ checkpoint and metrics are left untouched.
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
    config = ConfigArgs()

    num_epochs = config.get("epochs")
    learning_rate = config.get("learning_rate")
    image_size = config.get("image_size")
    model_name = config.get("model_name")
    window_size = config.get("window_size")
    pretrained = config.get("pretrained")

    results_name = f"{model_name}_balanced"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training using {model_name} for {window_size} frames with {num_epochs} epochs, "
        f"on {device} (class-balanced train/val, results -> Results/{results_name})"
    )

    model = get_model(model_name, preTrained=pretrained)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    transform = transforms.Compose(
        [
            transforms.Resize((256)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Balanced train/val: gradients and checkpoint selection see a 1:1 class ratio.
    train_loader, val_loader, _ = get_dataloaders(Transform=transform, Balance_Flags=True)
    # Unbalanced test: real-world ~6:1 ratio, for a like-for-like comparison
    # against the original run's Classification_Report_resnet18.txt.
    _, _, test_loader = get_dataloaders(Transform=transform, Balance_Flags=False)

    train_model(
        model, train_loader, val_loader, criterion, optimizer, results_name, num_epochs
    )

    print("Training completed.")
    print(f"Now testing the model {model_name} (balanced run) with the highest val accuracy")

    best_model_path = os.path.join("..", "Results", results_name, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    evaluate(model, test_loader, results_name)
