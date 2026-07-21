"""
========================================
EMBRYO ANALYSIS SYSTEM - TRAINING PIPELINE
========================================

This module provides the core training, validation, and testing pipeline for the
embryo development analysis system. It implements the complete machine learning
workflow including model training, evaluation, and result visualization.

Features:
- Complete training loop with validation monitoring
- Model evaluation with comprehensive metrics
- Automatic checkpointing and best model saving
- Training progress visualization and logging
- Mixed precision training for efficiency
- Comprehensive performance metrics (accuracy, F1-score, loss)

Pipeline Components:
- validate_model(): Validation loop with metrics calculation
- train_model(): Complete training loop with checkpointing
- evaluate(): Final model evaluation on test set
- Visualization and logging utilities

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


def validate_model(model, val_loader, criterion, epoch, num_epochs):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_targets, all_predictions = [], []
    loop = tqdm(
        val_loader,
        desc=f"Validating [{epoch+1}/{num_epochs}]",
        total=len(val_loader),
        leave=False,
    )
    with torch.no_grad():
        for batch in loop:
            stacks = batch[0]
            targets = batch[1]

            if stacks is None or targets is None:
                continue

            stacks, targets = stacks.to("cuda"), targets.to("cuda")

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(stacks)
                loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            loop.set_postfix(
                {"Val Loss": val_loss / len(val_loader), "Accuracy": correct / total}
            )

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    f1 = f1_score(all_targets, all_predictions, average="weighted", zero_division=0)

    return avg_val_loss, accuracy, f1


def train_model(
    model, train_loader, val_loader, criterion, optimizer, name_model, num_epochs
):
    save_path = os.path.join("..", "Results", f"{name_model}")
    os.makedirs(save_path, exist_ok=True)

    epoch_metrics = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "train_time": [],
    }

    best_acc = 0
    total_training_time = 0.0

    scaler = torch.amp.GradScaler("cuda")
    batch = next(iter(train_loader))
    stacks = batch[0]
    targets = batch[1]
    print(f"Confirming shapes - Stacks: {stacks.shape}, Targets: {targets.shape}")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        loop = tqdm(
            train_loader,
            desc=f"Training [{epoch+1}/{num_epochs}]",
            total=len(train_loader),
            leave=False,
        )
        for batch_idx, batch in enumerate(loop):

            stacks = batch[0]
            targets = batch[1]

            if stacks is None or targets is None:
                continue

            stacks, targets = stacks.to("cuda"), targets.to("cuda")

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(stacks)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            loop.set_postfix(
                {
                    "Train Loss": running_loss / (batch_idx + 1),
                    "Accuracy": correct / total,
                }
            )
        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(epoch_time))

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        val_loss, val_accuracy, val_f1 = validate_model(
            model, val_loader, criterion, epoch, num_epochs
        )

        epoch_metrics["epoch"].append(epoch + 1)
        epoch_metrics["train_loss"].append(avg_train_loss)
        epoch_metrics["train_accuracy"].append(train_accuracy)
        epoch_metrics["val_loss"].append(val_loss)
        epoch_metrics["val_accuracy"].append(val_accuracy)
        epoch_metrics["val_f1"].append(val_f1)
        epoch_metrics["train_time"].append(formatted_time)

        if val_accuracy > best_acc:
            save_best = os.path.join(save_path, f"best_model.pth")
            best_acc = val_accuracy
            torch.save(model.state_dict(), save_best)

    epoch_df = pd.DataFrame(epoch_metrics)
    csv_path = os.path.join(save_path, "epoch_metrics.csv")
    epoch_df.to_csv(csv_path, index=False)

    formatted_total_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
    print(f"Total Training Time : {formatted_total_time}")

    plot_metrics(
        train_losses=epoch_metrics["train_loss"],
        val_losses=epoch_metrics["val_loss"],
        train_accuracies=epoch_metrics["train_accuracy"],
        val_accuracies=epoch_metrics["val_accuracy"],
        model_name=name_model,
        path=save_path,
    )


def evaluate(model, test_loader, model_name):
    import os

    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    from tqdm import tqdm

    path = os.path.join("..", "Results", f"{model_name}")
    os.makedirs(path, exist_ok=True)

    model.eval()
    phase_to_index = {
        "tPB2": 0,
        "tPNa": 1,
        "tPNf": 2,
        "t2": 3,
        "t3": 4,
        "t4": 5,
        "t5": 6,
        "t6": 7,
        "t7": 8,
        "t8": 9,
        "t9+": 10,
        "tM": 11,
        "tSB": 12,
        "tB": 13,
        "tEB": 14,
    }
    phase_names = list(phase_to_index.keys())

    all_preds, all_labels = [], []
    correct_transitions, misclassified_transitions = [], []
    total = 0

    with torch.no_grad():
        for inputs, labels, first_phases, last_phases in tqdm(
            test_loader, desc="Evaluating", leave=False
        ):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            first_phases, last_phases = first_phases.to("cuda"), last_phases.to("cuda")

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)

            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)

            for i in range(len(labels)):
                start_phase = first_phases[i].item()
                end_phase = last_phases[i].item()

                if preds[i] == labels[i].item():
                    correct_transitions.append((start_phase, end_phase))
                else:
                    misclassified_transitions.append((start_phase, end_phase))

    correct = len(correct_transitions)
    accuracy = 100 * correct / total
    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Save classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["No Transition", "Transition"],
        zero_division=0,
    )
    report_path = os.path.join(path, f"Classification_Report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write("Test Set Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Test Set Accuracy: {accuracy:.4f}%\n")
        f.write("Test Set Metrics:\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1-Score: {f1:.4f}\n")

    # 2-class confusion matrix
    cm_2classes = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm_2classes,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Transition", "Transition"],
        yticklabels=["No Transition", "Transition"],
    )
    plt.title("Confusion Matrix for Transition Predictions (2 Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(path, f"confusion_matrix_2classes_{model_name}.png"))
    plt.close()

    # Correct transitions confusion matrix
    if correct_transitions:
        correct_cm = confusion_matrix(
            [x[0] for x in correct_transitions], [x[1] for x in correct_transitions]
        )
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            correct_cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=phase_names,
            yticklabels=phase_names,
        )
        plt.title("Confusion Matrix for Correct Transitions")
        plt.xlabel("End Phase")
        plt.ylabel("Start Phase")
        plt.savefig(
            os.path.join(path, f"confusion_matrix_correct_transitions_{model_name}.png")
        )
        plt.close()

    # Misclassified transitions confusion matrix
    if misclassified_transitions:
        misclassified_cm = confusion_matrix(
            [x[0] for x in misclassified_transitions],
            [x[1] for x in misclassified_transitions],
        )
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            misclassified_cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            xticklabels=phase_names,
            yticklabels=phase_names,
        )
        plt.title("Confusion Matrix for Misclassified Transitions")
        plt.xlabel("End Phase")
        plt.ylabel("Start Phase")
        plt.savefig(
            os.path.join(
                path, f"confusion_matrix_misclassified_transitions_{model_name}.png"
            )
        )
        plt.close()

    print(
        f"Evaluation finished. Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )


def plot_metrics(
    train_losses, val_losses, train_accuracies, val_accuracies, model_name, path
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accuracies, label="Train Accuracy")
    ax2.plot(val_accuracies, label="Validation Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    img_path = os.path.join(path, f"{model_name}.png")
    fig.suptitle(f"Model {model_name}", fontsize=16)
    fig.savefig(img_path)
    plt.close()
