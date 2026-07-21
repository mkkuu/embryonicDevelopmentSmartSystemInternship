"""
========================================
DOCTOR AND EMBRYO MANAGEMENT CLASS
========================================

This module contains the Doctor class which handles all doctor-related operations,
embryo data management, image processing, and AI-powered predictions for the
Embryo Analysis Web Application.

Features:
- Doctor account CRUD operations
- Embryo record management with images and annotations
- AI model integration for developmental transition prediction
- File management and organized storage system
- CSV annotation handling
- Database transaction management
- Role-based access control

AI Capabilities:
- ResNet18 model for 8-frame sequences
- TimeSformer model for 32-frame sequences
- Sliding window analysis for transition prediction
- GPU acceleration support
- Fallback to random predictions if models unavailable

File Management:
- Organized folder structure: embryo_{ID}_{date}/
- Original filename preservation
- CSV annotation files
- Automatic cleanup on deletion

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import csv
import os
import re

import psycopg2
from dotenv import load_dotenv

# Load environment variables for database configuration
load_dotenv()

class Doctor:
    """
    Doctor and Embryo Management Class
    
    This class handles all operations related to doctors and embryo data management,
    including CRUD operations, image processing, AI predictions, and file management.
    It provides a comprehensive interface for managing embryo records with their
    associated images and annotations.
    
    Key Responsibilities:
    - Doctor account management (CRUD operations)
    - Embryo record management with images and annotations
    - AI-powered developmental transition prediction
    - File management and organized storage
    - CSV annotation handling
    - Database transaction management
    - Role-based access control
    
    Attributes:
        session: Flask session object for database connections
        db_config: Database configuration dictionary from environment variables
    
    Methods:
        Core Database Operations:
        - execute_query: Execute SQL queries with error handling
        - get_list: Retrieve all doctor users
        - add: Add new doctor user
        - update: Update existing doctor information
        - delete: Delete doctor user
        
        Embryo Management:
        - addEmbryo: Add new embryo record with images and annotations
        - fetchAllEmbryo: Retrieve embryo records based on access level
        - deleteEmbryo: Delete embryo record and associated files
        - getEmbryoImagesAndAnnotations: Get images and annotations for embryo
        - updateEmbryo: Update embryo record (excluding images)
        
        AI Prediction:
        - predictTransitions: Run AI prediction on embryo image sequences
    """
    
    def __init__(self, session):
        """
        Initialize the Doctor class with session management
        
        Args:
            session: Flask session object for database connections
        """
        self.session = session
        # Database configuration from environment variables
        # This ensures secure credential management and environment-specific settings
        self.db_config = {
            'host': os.getenv('host'),           # Database server hostname
            'database': os.getenv('database'),   # Database name
            'user': os.getenv('user'),           # Database username
            'password': os.getenv('password'),   # Database password
            'port': int(os.getenv('port')),      # Database port number
        }

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def execute_query(self, query, values=None):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if values:
                        cursor.execute(query, values)
                    else:
                        cursor.execute(query)
                conn.commit()
            return True
        except Exception as e:
            return False

    def get_list(self):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                            SELECT 
                                u.user_id,
                                u.first_name,
                                u.last_name,
                                u.gender,
                                u.birthday,
                                u.contact,
                                u.address,
                                ua.password,
                                u.global_access
                            FROM 
                                users u
                            JOIN 
                                user_auth ua ON u.user_id = ua.user_id
                            WHERE 
                                u.role = 'Doctor';
                    """
                    cursor.execute(query)
                    results = cursor.fetchall()
                    return list(results), True
        except Exception as e:
            return [], False


    def add(self, values):
        try:
            placeholders = ', '.join(['%s'] * len(values))
            columns = ', '.join(values.keys())
            query = f"INSERT INTO users ({columns}) VALUES ({placeholders})"
            return self.execute_query(query, tuple(values.values()))
        except Exception as e:
            return False

    def update(self, table, ID, values):
        try:
            columns = ', '.join([f"{key} = %s" for key in values.keys()])
            query = f"UPDATE {table} SET {columns} WHERE user_id = %s"
            return self.execute_query(query, tuple(values.values()) + (ID,))
        except Exception as e:
            return False

    def delete(self, table, ID):
        try:
            query = f"DELETE FROM {table} WHERE user_id = %s"
            return self.execute_query(query, (ID,))
        except Exception as e:
            return False

    def addEmbryo(self, embryo_data, images, annotations_csv, current_user_id):
        """Add a new embryo record with images and annotations"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Create folder path
                    folder_name = f"embryo_{embryo_data['embryo_id']}_{embryo_data['date']}"
                    base_path = os.getenv('EMBRYO_IMAGES_PATH', r"C:/Embryo_images")
                    folder_path = os.path.join(base_path, folder_name)
                    
                    # Normalize path separators
                    folder_path = os.path.normpath(folder_path)
                    
                    # Create folder
                    os.makedirs(folder_path, exist_ok=True)
                    
                    # Save images
                    for image in images:
                        if image.filename:  # Check if image has a filename
                            # Extract just the filename from the full path
                            # Handle cases where filename includes folder path like "TEST/Patient_0_Image_5.jpeg"
                            filename = os.path.basename(image.filename)
                            
                            image_path = os.path.join(folder_path, filename)
                            image_path = os.path.normpath(image_path)  # Normalize path separators
                            
                            # Check if the image file exists and is readable
                            if hasattr(image, 'save'):
                                # This is a Flask FileStorage object
                                image.save(image_path)
                            else:
                                # This might be a file path, try to copy it
                                import shutil
                                shutil.copy2(image, image_path)
                        else:
                            pass  # Skip images with no filename
                    
                    # Create and save CSV file
                    csv_filename = f"embryo_{embryo_data['embryo_id']}_{embryo_data['date']}_annotations.csv"
                    csv_path = os.path.join(folder_path, csv_filename)
                    csv_path = os.path.normpath(csv_path)  # Normalize path separators
                    
                    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                        csvfile.write("image_name,phase\n")
                        for annotation in annotations_csv:
                            # Extract just the filename from the full path for CSV
                            csv_filename = os.path.basename(annotation['image_name'])
                            csvfile.write(f"{csv_filename},{annotation['phase']}\n")
                    
                    # Insert into database
                    query = """
                        INSERT INTO embryo (date, contact, blastocyst_grade, pgt_a_grade, live_birth, path, doctor_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(query, (
                        embryo_data['date'],
                        embryo_data['contact'],
                        embryo_data['blastocyst_grade'],
                        embryo_data['pgt_a_grade'],
                        embryo_data['live_birth'],
                        folder_path,
                        current_user_id
                    ))
                    
                    conn.commit()
                    return True, folder_path
                    
        except Exception as e:
            # Clean up folder if database insert failed
            try:
                if 'folder_path' in locals():
                    import shutil
                    shutil.rmtree(folder_path)
            except Exception as cleanup_error:
                pass
            
            return False, str(e)

    def fetchAllEmbryo(self, current_user_id):
        """Fetch embryo records based on doctor's global access level"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # First, check if the current doctor has global access
                    access_query = """
                        SELECT global_access 
                        FROM users 
                        WHERE user_id = %s
                    """
                    cursor.execute(access_query, (current_user_id,))
                    access_result = cursor.fetchone()
                    
                    if not access_result:
                        return [], False
                    
                    global_access = access_result[0]
                    
                    # Build query based on global access
                    if global_access:
                        # Global access = 1: Fetch ALL embryos
                        query = """
                            SELECT 
                                embryo_id,
                                date,
                                contact,
                                blastocyst_grade,
                                pgt_a_grade,
                                live_birth,
                                path,
                                doctor_id
                            FROM embryo
                            ORDER BY embryo_id DESC
                        """
                        cursor.execute(query)
                    else:
                        # Global access = 0: Fetch only embryos added by this doctor
                        query = """
                            SELECT 
                                embryo_id,
                                date,
                                contact,
                                blastocyst_grade,
                                pgt_a_grade,
                                live_birth,
                                path,
                                doctor_id
                            FROM embryo
                            WHERE doctor_id = %s
                            ORDER BY embryo_id DESC
                        """
                        cursor.execute(query, (current_user_id,))
                    
                    results = cursor.fetchall()
                    
                    # Convert to list of dictionaries
                    embryo_list = []
                    for row in results:
                        embryo_dict = {
                            'embryo_id': row[0],
                            'date': row[1].isoformat() if row[1] else None,
                            'contact': row[2],
                            'blastocyst_grade': row[3],
                            'pgt_a_grade': row[4],
                            'live_birth': row[5],
                            'path': row[6],
                            'doctor_id': row[7]
                        }
                        embryo_list.append(embryo_dict)
                    
                    return embryo_list, True
        except Exception as e:
            return [], False

    def deleteEmbryo(self, embryo_id, current_user_id):
        """Delete embryo record from database and remove associated directory"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # First, get the path before deleting the record
                    path_query = """
                        SELECT path 
                        FROM embryo 
                        WHERE embryo_id = %s AND doctor_id = %s
                    """
                    cursor.execute(path_query, (embryo_id, current_user_id))
                    path_result = cursor.fetchone()
                    
                    if not path_result:
                        return False, "Embryo not found or access denied"
                    
                    folder_path = path_result[0]
                    
                    # Delete the database record
                    delete_query = """
                        DELETE FROM embryo 
                        WHERE embryo_id = %s AND doctor_id = %s
                    """
                    cursor.execute(delete_query, (embryo_id, current_user_id))
                    
                    if cursor.rowcount == 0:
                        return False, "No record deleted"
                    
                    conn.commit()
                    
                    # Delete the associated directory
                    if folder_path and os.path.exists(folder_path):
                        try:
                            import shutil
                            shutil.rmtree(folder_path)
                        except Exception as dir_error:
                            # Don't fail the operation if directory deletion fails
                            pass
                    
                    return True, "Embryo deleted successfully"
                    
        except Exception as e:
            return False, str(e)

    def getEmbryoImagesAndAnnotations(self, embryo_id, current_user_id):
        """Get images and annotations for a specific embryo"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # First, check if user has access to this embryo
                    access_query = """
                        SELECT path, doctor_id 
                        FROM embryo 
                        WHERE embryo_id = %s
                    """
                    cursor.execute(access_query, (embryo_id,))
                    result = cursor.fetchone()
                    
                    if not result:
                        return None, "Embryo not found"
                    
                    folder_path, doctor_id = result
                    
                    # Check if user has access (either owns it or has global access)
                    if doctor_id != current_user_id:
                        # Check if user has global access
                        global_access_query = """
                            SELECT global_access 
                            FROM users 
                            WHERE user_id = %s
                        """
                        cursor.execute(global_access_query, (current_user_id,))
                        access_result = cursor.fetchone()
                        
                        if not access_result or not access_result[0]:
                            return None, "Access denied"
                    
                    if not folder_path or not os.path.exists(folder_path):
                        return None, "Image folder not found"
                    
                    # Get list of image files
                    image_files = []
                    csv_file = None
                    
                    for filename in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, filename)
                        if os.path.isfile(file_path):
                            if filename.endswith('.csv'):
                                csv_file = filename
                            elif filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                                image_files.append(filename)
                    
                    # Sort image files naturally
                    image_files.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', x)])
                    
                    # Read annotations from CSV if it exists
                    annotations = []
                    if csv_file:
                        csv_path = os.path.join(folder_path, csv_file)
                        try:
                            with open(csv_path, 'r', encoding='utf-8') as f:
                                reader = csv.DictReader(f)
                                annotations = list(reader)
                        except Exception as e:
                            pass
                    
                    return {
                        'folder_path': folder_path,
                        'image_files': image_files,
                        'annotations': annotations,
                        'csv_file': csv_file
                    }, None
                    
        except Exception as e:
            return None, str(e)

    def updateEmbryo(self, embryo_id, embryo_data, annotations_csv, current_user_id):
        """Update embryo record (excluding images)"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # First, check if user has access to this embryo
                    access_query = """
                        SELECT path, doctor_id 
                        FROM embryo 
                        WHERE embryo_id = %s
                    """
                    cursor.execute(access_query, (embryo_id,))
                    result = cursor.fetchone()
                    
                    if not result:
                        return False, "Embryo not found"
                    
                    folder_path, doctor_id = result
                    
                    # Check if user has access (either owns it or has global access)
                    if doctor_id != current_user_id:
                        # Check if user has global access
                        global_access_query = """
                            SELECT global_access 
                            FROM users 
                            WHERE user_id = %s
                        """
                        cursor.execute(global_access_query, (current_user_id,))
                        access_result = cursor.fetchone()
                        
                        if not access_result or not access_result[0]:
                            return False, "Access denied"
                    
                    # Update the database record
                    update_query = """
                        UPDATE embryo 
                        SET date = %s, contact = %s, blastocyst_grade = %s, 
                            pgt_a_grade = %s, live_birth = %s
                        WHERE embryo_id = %s AND doctor_id = %s
                    """
                    cursor.execute(update_query, (
                        embryo_data['date'],
                        embryo_data['contact'],
                        embryo_data['blastocyst_grade'],
                        embryo_data['pgt_a_grade'],
                        embryo_data['live_birth'],
                        embryo_id,
                        current_user_id
                    ))
                    
                    if cursor.rowcount == 0:
                        return False, "No record updated"
                    
                    # Update the CSV file with new annotations
                    if folder_path and os.path.exists(folder_path):
                        # Find the CSV file
                        csv_file = None
                        for filename in os.listdir(folder_path):
                            if filename.endswith('.csv'):
                                csv_file = filename
                                break
                        
                        if csv_file:
                            csv_path = os.path.join(folder_path, csv_file)
                            try:
                                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                                    csvfile.write("image_name,phase\n")
                                    for annotation in annotations_csv:
                                        csvfile.write(f"{annotation['image_name']},{annotation['phase']}\n")
                            except Exception as e:
                                pass
                    
                    conn.commit()
                    return True, "Embryo updated successfully"
                    
        except Exception as e:
            return False, str(e)

    def predictTransitions(self, images, frame_count, current_user_id):
        """
        Predict developmental transitions using AI models
        
        This method processes embryo image sequences through AI models to predict
        developmental transitions. It supports two different models based on
        the number of frames: ResNet18 for 8-frame sequences and TimeSformer
        for 32-frame sequences.
        
        Args:
            images (list): List of base64-encoded image data
            frame_count (int): Number of frames in the sequence (8 or 32)
            current_user_id (int): ID of the requesting doctor for access control
            
        Returns:
            tuple: (success, predictions, is_random) where:
                - success (bool): True if prediction completed successfully
                - predictions (list): List of binary predictions (0=no transition, 1=transition)
                - is_random (bool): True if random predictions were used (model unavailable)
        
        Model Selection:
            - 8 frames: Uses ResNet18 model (Results/resnet18/best_model.pth)
            - 32 frames: Uses TimeSformer model (Results/timesformer/best_model.pth)
        
        Image Processing Pipeline:
            1. Decode base64 images to PIL Image objects
            2. Resize images to 224x224 pixels
            3. Convert to appropriate tensor format:
               - 8 frames: Grayscale sequences (T, H, W)
               - 32 frames: RGB video format (T, 3, H, W)
            4. Apply sliding window analysis
            5. Run model inference on each window
            6. Return binary predictions
        
        Sliding Window Analysis:
            - For N images and frame_count F, creates (N-F+1) windows
            - Each window contains F consecutive frames
            - Predictions are made for each window independently
            - Results indicate transitions in the sequence
        
        Fallback Behavior:
            - If model files don't exist, returns random predictions
            - Sets is_random flag to True for frontend notification
            - Maintains same prediction format for consistency
        
        GPU Support:
            - Automatically detects CUDA availability
            - Falls back to CPU if GPU unavailable
            - Optimizes performance based on available hardware
        
        Security:
            - Validates frame_count parameter
            - Handles model loading errors gracefully
            - Provides consistent error responses
        
        Example:
            >>> doctor = Doctor(session)
            >>> images = ["base64_image1", "base64_image2", ...]
            >>> success, predictions, is_random = doctor.predictTransitions(images, 8, user_id)
            >>> print(f"Predictions: {predictions}, Random: {is_random}")
        """
        try:
            import base64
            import io
            import random

            import torch
            from PIL import Image
            from torchvision import transforms

            # Determine model path based on frame count (relative to project root)
            if frame_count == 8:
                model_path = "../Results/resnet18/best_model.pth"
                mode = "image_seq"  # Grayscale sequences
            elif frame_count == 32:
                model_path = "../Results/timesformer/best_model.pth" 
                mode = "video"  # RGB video format
            else:
                return False, f"Unsupported frame count: {frame_count}"
            
            # Check if model exists
            if not os.path.exists(model_path):
                # Fallback to random predictions if model doesn't exist
                predictions = []
                for i in range(len(images) - frame_count + 1):
                    prediction = random.randint(0, 1)
                    predictions.append(prediction)
                    if i == 0:
                        pass  # First window colors all frames
                    else:
                        pass  # Subsequent windows color only new frame
                return True, predictions, True  # Return with random flag
            
            # Set device (GPU if available)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Decode and preprocess images
            processed_images = []
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            
            for i, img_base64 in enumerate(images):
                try:
                    # Decode base64 image
                    img_data = base64.b64decode(img_base64)
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    
                    # Apply transforms
                    img_tensor = transform(img)
                    processed_images.append(img_tensor)
                    
                except Exception as e:
                    return False, f"Error processing image {i}: {str(e)}"
            
            # Convert to appropriate format based on mode
            if mode == "video":
                # Stack as (T, 3, H, W) for TimeSformer
                images_tensor = torch.stack(processed_images, dim=0)
            elif mode == "image_seq":
                # Convert to grayscale and squeeze to (T, H, W) for ResNet
                grayscale_images = [im.mean(dim=0, keepdim=True) for im in processed_images]
                images_tensor = torch.stack(grayscale_images, dim=0).squeeze(1)
            
            # Load model
            model = torch.load(model_path, map_location=device)
            model.eval()
            model.to(device)
            
            # Run inference
            predictions = []
            with torch.no_grad():
                # Sliding window approach
                for i in range(len(processed_images) - frame_count + 1):
                    window = images_tensor[i:i+frame_count]
                    if mode == "video":
                        window = window.unsqueeze(0)  # Add batch dimension
                    
                    window = window.to(device)
                    output = model(window)
                    prediction = torch.round(torch.sigmoid(output)).item()
                    predictions.append(int(prediction))
                    
                    if i == 0:
                        pass  # First window colors all frames
                    else:
                        pass  # Subsequent windows color only new frame
            
            return True, predictions, False  # Return with random flag (False = real model)
            
        except Exception as e:
            return False, f"Error predicting transitions: {str(e)}"