"""
========================================
EMBRYO ANALYSIS SYSTEM - DATA PREPROCESSING
========================================

This module provides comprehensive data preprocessing utilities for the embryo
development analysis system. It handles dataset cleaning, organization, and
preparation for machine learning training.

Features:
- Duplicate folder detection and removal
- Systematic folder and image renaming
- Dataset splitting (train/validation/test)
- CSV annotation file generation
- Data quality validation and cleaning
- Progress tracking with tqdm

Preprocessing Pipeline:
1. Remove duplicate subfolders
2. Rename folders and images systematically
3. Generate annotation CSV files
4. Split dataset into train/validation/test sets
5. Validate data integrity and completeness

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import filecmp
import os
import re
import shutil
import warnings

import pandas as pd
from config_args import ConfigArgs
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*DataFrameGroupBy.apply operated on the grouping columns.*",
)


def delete_duplicate_subfolders(main_folder):
    """
    Remove duplicate subfolders within the main folder structure.
    
    This function identifies and removes subfolders that contain identical
    files to their parent folders, helping to clean up redundant data
    structures in the embryo dataset.
    
    Args:
        main_folder (str): Path to the main folder containing subfolders
        
    The function uses filecmp.dircmp to compare folder contents and removes
    subfolders that have common files with their parent directories.
    """
    all_folders = [
        folder
        for folder in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, folder))
    ]
    dir_name = os.path.basename(main_folder)
    for folder in tqdm(all_folders, desc=f"Deleting duplicate subfolders {dir_name}"):
        folder_path = os.path.join(main_folder, folder)
        subfolders = [
            f
            for f in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, f))
        ]
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            try:
                if filecmp.dircmp(folder_path, subfolder_path).common_files:
                    # print(f"Deleting duplicate subfolder: {subfolder_path}")
                    shutil.rmtree(subfolder_path)
            except Exception as e:
                print(f"Skipping comparison due to error: {e}")


def rename_folders_and_images(base_dir, image_pattern):
    folders = sorted(
        [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    )
    name_map = {}
    dir_name = os.path.basename(base_dir)
    for idx, folder in tqdm(
        enumerate(folders), desc=f"Renaming folders and images {dir_name}"
    ):
        old_path = os.path.join(base_dir, folder)
        new_name = f"Patient_{idx}"
        new_path = os.path.join(base_dir, new_name)
        os.rename(old_path, new_path)
        name_map[folder] = new_name
        for image_name in os.listdir(new_path):
            match = image_pattern.search(image_name)
            if match:
                image_id = match.group(1)
                new_image_name = f"{new_name}_Image_{image_id}.jpeg"
                os.rename(
                    os.path.join(new_path, image_name),
                    os.path.join(new_path, new_image_name),
                )
    return name_map


# 3. Rename shared CSV files once
def rename_shared_csvs(phases_dir, time_dir, grades_path, name_map):
    for old_name, new_name in name_map.items():
        phases_file = os.path.join(phases_dir, f"{old_name}_phases.csv")
        if os.path.exists(phases_file):
            os.rename(phases_file, os.path.join(phases_dir, f"{new_name}_phases.csv"))
        time_file = os.path.join(time_dir, f"{old_name}_timeElapsed.csv")
        if os.path.exists(time_file):
            os.rename(time_file, os.path.join(time_dir, f"{new_name}_timeElapsed.csv"))
    if os.path.exists(grades_path):
        df = pd.read_csv(grades_path)
        df.replace(name_map, inplace=True)
        df.fillna("NA", inplace=True)
        df["Patient_Number"] = (
            df["video_name"].str.extract(r"Patient_(\d+)").astype(int)
        )
        df_sorted = df.sort_values(by=["Patient_Number"], ascending=[True])
        df_sorted = df_sorted.drop(columns=["Patient_Number"])
        df_sorted.to_csv(grades_path, index=False)


# 4. Process annotations to build Path Full CSV
def process_annotations(annotation_dir, images_dir, dataset_name, output_dir, focal):
    annotation_dir = os.path.abspath(annotation_dir)
    images_dir = os.path.abspath(images_dir)
    output_dir = os.path.abspath(output_dir)
    data_csv = []
    for file in tqdm(os.listdir(annotation_dir), desc=f"Processing {dataset_name}"):
        if not file.endswith("_phases.csv"):
            continue
        df = pd.read_csv(
            os.path.join(annotation_dir, file),
            header=None,
            names=["phase", "frame_start", "frame_end"],
        )
        patient = file.replace("_phases.csv", "")
        folder = os.path.join(images_dir, patient)
        for _, row in df.iterrows():
            for i in range(row["frame_start"], row["frame_end"] + 1):
                filename = f"{patient}_Image_{i}.jpeg"
                full_path = os.path.join(folder, filename)
                if os.path.exists(full_path):
                    data_csv.append(
                        {
                            "Video_name": patient,
                            "Path": full_path,
                            "Phase": row["phase"],
                            "Identifier": filename,
                            "Focal": focal,
                        }
                    )
    df_out = pd.DataFrame(data_csv)
    df_out["Patient_Number"] = (
        df_out["Video_name"].str.extract(r"Patient_(\d+)").astype(int)
    )
    df_out["Image_Number"] = (
        df_out["Identifier"].str.extract(r"Image_(\d+)").astype(int)
    )
    df_sorted = df_out.sort_values(
        by=["Patient_Number", "Image_Number"], ascending=[True, True]
    )
    df_sorted = df_sorted.drop(columns=["Patient_Number", "Image_Number"])
    df_sorted.to_csv(os.path.join(output_dir, f"{dataset_name}.csv"), index=False)


# 7. remove unreferenced images
def remove_images_not_in_csv(folder_path, csv_path):
    df = pd.read_csv(csv_path)
    valid_paths = set(df["Identifier"])
    print(f"Valid images listed in CSV: {len(valid_paths)}")
    deleted_count = 0
    for root, _, files in tqdm(os.walk(folder_path), desc="Removing images not in CSV"):
        for file in files:
            if file not in valid_paths:
                full_path = os.path.join(root, file)
                os.remove(full_path)
                deleted_count += 1
                # print(f"Removed: {full_path}")

    print(f"Total images removed: {deleted_count}")


# 8. Create name map for renaming
def create_name_map(name_map):
    df = pd.DataFrame(list(name_map.items()), columns=["OldName", "NewName"])
    df["Patient_Number"] = df["NewName"].str.extract(r"Patient_(\d+)").astype(int)
    df_sorted = df.sort_values(by=["Patient_Number"], ascending=[True])
    df_sorted = df_sorted.drop(columns=["Patient_Number"])
    df_sorted.to_csv(os.path.join("..", "Data", "folder_map_naming.csv"), index=False)
    return df


# 9. Create train, val, test splits
def create_splits(focal, csv_path):
    df = pd.read_csv(csv_path)
    video_names = df["Video_name"].unique()
    print(f"Total unique videos: {len(video_names)}")
    train_videos, temp_videos = train_test_split(
        video_names, test_size=0.30, random_state=42
    )
    val_videos, test_videos = train_test_split(
        temp_videos, test_size=0.5, random_state=42
    )

    train_df = df[df["Video_name"].isin(train_videos)].copy()
    val_df = df[df["Video_name"].isin(val_videos)].copy()
    test_df = df[df["Video_name"].isin(test_videos)].copy()

    train_df["UsedFor"] = "Train"
    val_df["UsedFor"] = "Val"
    test_df["UsedFor"] = "Test"

    print(f"Train videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    print(f"Test videos: {len(test_videos)}")

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    df_merged = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_merged["Patient_Number"] = (
        df_merged["Video_name"].str.extract(r"Patient_(\d+)").astype(int)
    )
    df_merged["Image_Number"] = (
        df_merged["Identifier"].str.extract(r"Image_(\d+)").astype(int)
    )
    df_sorted = df_merged.sort_values(
        by=["Patient_Number", "Image_Number"], ascending=[True, True]
    )
    df_sorted = df_sorted.drop(columns=["Patient_Number", "Image_Number"])
    df_sorted.to_csv(os.path.join("..", "Data", "Splits", f"{focal}.csv"), index=False)


# 10. remove problematic images
def remove_problematic_images(folder_path, csv_missing_path, csv_path):
    if not os.path.exists(csv_missing_path):
        print(
            f"CSV file {csv_missing_path} does not exist. Skipping removal of unreferenced images."
        )
        return
    df = pd.read_csv(csv_missing_path)
    df_path = pd.read_csv(csv_path)

    problem_images = set(df["Identifier"])
    print(f"Problematic images listed in CSV: {len(problem_images)}")
    deleted_count = 0
    for root, _, files in tqdm(os.walk(folder_path), desc="Removing images in CSV"):
        for file in files:
            if file in problem_images:
                full_path = os.path.join(root, file)
                os.remove(full_path)
                deleted_count += 1
                # print(f"Removed: {full_path}")
    df_path = df_path[~df_path["Identifier"].isin(problem_images)]
    df_path.to_csv(csv_path, index=False)
    print(f"Total images removed: {deleted_count}")


# Main
if __name__ == "__main__":

    config = ConfigArgs()

    if os.path.exists(os.path.join("..", "Data", "embryo_dataset")):
        os.rename(
            os.path.join("..", "Data", "embryo_dataset"),
            os.path.join("..", "Data", "embryo_dataset_F0"),
        )

    focal_dirs = ["F0"]
    base_data_dir = os.path.join("..", "Data")
    annotation_dir = os.path.join("..", "Data", "embryo_dataset_annotations")
    time_csv_dir = os.path.join("..", "Data", "embryo_dataset_time_elapsed")
    grades_csv_path = os.path.join("..", "Data", "embryo_dataset_grades.csv")
    image_pattern = re.compile(r"RUN(\d+)\.jpeg", re.IGNORECASE)

    os.makedirs(os.path.join("..", "Data", "Paths"), exist_ok=True)
    os.makedirs(os.path.join("..", "Data", "Splits"), exist_ok=True)
    os.makedirs(os.path.join("..", "Results"), exist_ok=True)

    for focal in focal_dirs:
        dir_path = os.path.join(base_data_dir, f"embryo_dataset_{focal}")
        delete_duplicate_subfolders(dir_path)
        name_map = rename_folders_and_images(dir_path, image_pattern)

        if focal == "F0":
            rename_shared_csvs(annotation_dir, time_csv_dir, grades_csv_path, name_map)
            create_name_map(name_map)

        process_annotations(
            annotation_dir,
            dir_path,
            f"embryo_dataset_{focal}",
            os.path.join("..", "Data", "Paths"),
            focal,
        )

        remove_images_not_in_csv(
            dir_path, os.path.join("..", "Data", "Paths", f"embryo_dataset_{focal}.csv")
        )
        remove_problematic_images(
            dir_path,
            os.path.join("..", "Data", "missing_paths_F0.csv"),
            os.path.join("..", "Data", "Paths", f"embryo_dataset_{focal}.csv"),
        )
        create_splits(
            focal, os.path.join("..", "Data", "Paths", f"embryo_dataset_{focal}.csv")
        )
