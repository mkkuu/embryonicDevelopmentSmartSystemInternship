"""
========================================
DOCTOR ROUTES - FLASK BLUEPRINT
========================================

This module contains Flask routes for all doctor-specific operations in the
Embryo Analysis Web Application. It provides RESTful API endpoints for
embryo management, image processing, and AI-powered predictions.

Features:
- Embryo record CRUD operations
- Image upload and management
- AI prediction endpoints
- File serving for embryo images
- Role-based access control
- Session-based authentication

API Endpoints:
- GET /Doctor/Embryo/LIST: Retrieve embryo records
- POST /Doctor/Embryo/ADD: Add new embryo with images
- POST /Doctor/Embryo/UPDATE: Update embryo record
- POST /Doctor/Embryo/DELETE: Delete embryo record
- POST /Doctor/Embryo/GET_IMAGES: Get embryo images and annotations
- GET /Doctor/Embryo/IMAGE/<id>/<filename>: Serve image files
- POST /Doctor/Embryo/PREDICT: Run AI prediction on sequences

Security Features:
- Session-based authentication
- Doctor ownership verification
- Global access level checking
- Input validation and sanitization
- SQL injection prevention

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import datetime
import json
import os

from Classes.Doctor import Doctor
from flask import (Blueprint, jsonify, render_template, request, send_file,
                   session)

# Initialize Doctor class instance for business logic
doctor = Doctor(session)

# Create Flask blueprint for doctor-specific routes
# All routes will be prefixed with '/Doctor'
doctor_bp = Blueprint('doctor_bp', __name__, url_prefix='/Doctor')


@doctor_bp.route('/Register', endpoint='Register')
def Register():
    # name = session.get('Name')
    pass

# Doctor
@doctor_bp.route('/Embryo', endpoint='Embryo')
def Doctor():
    name = session.get('first_name')
    lastName = session.get('last_name')
    return render_template('Doctor/Embryo.html',
                           name=name,
                           lastName=lastName)

# Add-Doctor
@doctor_bp.route('/Doctor/ADD', methods=['POST'])
def Doctor_add():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    
    NAME = data.get('Name')
    LAST = data.get('Last')
    GENDER = data.get('Gender')
    BIRTHDAY = data.get('BirthDay') 
    CONTACT = data.get('Contact')
    ADDRESS = data.get('Address')
    status = admin.add({'first_name': NAME, 'last_name': LAST, 'gender': GENDER, 'birthday': BIRTHDAY,
                                  'contact': CONTACT, 'address': ADDRESS, 'Role':'Doctor'})
    
    if status:
        return jsonify({"message": "Row added successfully"}), 200
    else:
        return jsonify({"error": "Failed to add row or user does not exist"}), 500



# Update-Doctor
@doctor_bp.route('/Doctor/UPDATE', methods=['POST', 'PUT'])
def Doctor_update():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON in request"}), 400
    ID = data.get('ID')
    NAME = data.get('Name')
    LAST = data.get('Last')
    GENDER = data.get('Gender')
    BIRTHDAY = data.get('BirthDay') 
    CONTACT = data.get('Contact')
    ADDRESS = data.get('Address')
    PASSWORD = data.get('Password')
    
    status_users = admin.update('users', ID, {
        'first_name': NAME, 
        'last_name': LAST,
        'gender': GENDER, 
        'birthday': BIRTHDAY,
        'contact': CONTACT, 
        'address': ADDRESS
    })
    status_uers_auth = admin.update('user_auth', ID, {
        'username': CONTACT, 
        'password': PASSWORD
    })

    if status_users and status_uers_auth:
        return jsonify({"message": "Row updated successfully"}), 200
    else:
        return jsonify({"error": "Failed to update row in the database"}), 500


# Update-Doctor-Acess
@doctor_bp.route('/Doctor/ACCESS', methods=['POST'])
def Doctor_update_Access():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON in request"}), 400
    ID = data.get('ID')
    global_access = str(data.get('GlobalAccess')).upper()
    status_users = admin.update('users', ID, {
        'global_access': global_access
    })

    if status_users:
        return jsonify({"message": "Row updated successfully"}), 200
    else:
        return jsonify({"error": "Failed to update row in the database"}), 500
    
# Delete-Doctor
@doctor_bp.route('/Doctor/DELETE/<int:row_id>', methods=['DELETE'])
def Doctor_delete(row_id):
    status = admin.delete("users", row_id)
    if status:
        return jsonify({"message": "Row delete successfully"}), 200
    else:
        return jsonify({"error": "Failed to delete row"}), 500

# Doctor-List
@doctor_bp.route('/Doctor/LIST', methods=['GET'])
def Doctor_List():
    doctor_tuples, status = admin.get_list()
    if not status:
        return jsonify({"error": "Doctor List Problem"}), 500

    keys = ["ID", "Name", "Last", "Gender", "BirthDay", "Contact", "Address", "Password", "GlobalAccess"]
    
    doctor_list = []
    for row_tuple in doctor_tuples:
        row_dict = dict(zip(keys, row_tuple))
        if isinstance(row_dict.get("BirthDay"), datetime.date):
            row_dict["BirthDay"] = row_dict["BirthDay"].isoformat()
            
        doctor_list.append(row_dict)

    return jsonify(doctor_list), 200

# Fetch All Embryo Records
@doctor_bp.route('/Embryo/LIST', methods=['GET'])
def Embryo_List():
    # Get current user ID from session
    current_user_id = session.get('user_id')
    if not current_user_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    embryo_list, status = doctor.fetchAllEmbryo(current_user_id)
    if not status:
        return jsonify({"error": "Failed to fetch embryo records"}), 500
    
    return jsonify(embryo_list), 200

# Add New Embryo Record
@doctor_bp.route('/Embryo/ADD', methods=['POST'])
def Embryo_Add():
    # Get current user ID from session
    current_user_id = session.get('user_id')
    
    if not current_user_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        # Get form data
        embryo_data = {
            'embryo_id': request.form.get('embryo_id'),
            'date': request.form.get('date'),
            'contact': request.form.get('contact', ''),
            'blastocyst_grade': request.form.get('blastocyst_grade'),
            'pgt_a_grade': request.form.get('pgt_a_grade', '') or None,  # Convert empty string to None
            'live_birth': request.form.get('live_birth', '') or None,  # Convert empty string to None
        }
        
        # Get images
        images = request.files.getlist('images')
        
        if not images or len(images) == 0:
            return jsonify({"error": "No images provided"}), 400
        
        # Get annotations
        annotations_json = request.form.get('annotations')
        if not annotations_json:
            return jsonify({"error": "No annotations provided"}), 400
        
        annotations_csv = json.loads(annotations_json)
        
        # Add embryo record
        success, result = doctor.addEmbryo(embryo_data, images, annotations_csv, current_user_id)
        
        if success:
            return jsonify({"message": "Embryo record added successfully", "path": result}), 200
        else:
            return jsonify({"error": f"Failed to add embryo record: {result}"}), 500
            
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

# Delete Embryo Record
@doctor_bp.route('/Embryo/DELETE', methods=['POST'])
def Embryo_Delete():
    # Get current user ID from session
    current_user_id = session.get('user_id')
    
    if not current_user_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        # Get embryo ID from request
        data = request.get_json()
        embryo_id = data.get('embryo_id')
        
        if not embryo_id:
            return jsonify({"error": "Embryo ID is required"}), 400
        
        # Call delete method
        success, message = doctor.deleteEmbryo(embryo_id, current_user_id)
        
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Get Embryo Images and Annotations
@doctor_bp.route('/Embryo/GET_IMAGES', methods=['POST'])
def Embryo_Get_Images():
    # Get current user ID from session
    current_user_id = session.get('user_id')
    
    if not current_user_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        # Get embryo ID from request
        data = request.get_json()
        embryo_id = data.get('embryo_id')
        
        if not embryo_id:
            return jsonify({"error": "Embryo ID is required"}), 400
        
        # Get images and annotations
        result, error = doctor.getEmbryoImagesAndAnnotations(embryo_id, current_user_id)
        
        if error:
            return jsonify({"error": error}), 400
        
        return jsonify(result), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve Embryo Image Files
@doctor_bp.route('/Embryo/IMAGE/<int:embryo_id>/<path:filename>')
def Embryo_Serve_Image(embryo_id, filename):
    # Get current user ID from session
    current_user_id = session.get('user_id')
    
    if not current_user_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        # Get embryo path and verify access
        result, error = doctor.getEmbryoImagesAndAnnotations(embryo_id, current_user_id)
        
        if error:
            return jsonify({"error": error}), 404
        
        folder_path = result['folder_path']
        image_path = os.path.join(folder_path, filename)
        
        # Security check - ensure the file is within the embryo folder
        if not os.path.exists(image_path) or not image_path.startswith(folder_path):
            return jsonify({"error": "Image not found"}), 404
        
        # Determine content type based on file extension
        content_type = 'image/jpeg'  # default
        if filename.lower().endswith('.png'):
            content_type = 'image/png'
        elif filename.lower().endswith('.gif'):
            content_type = 'image/gif'
        elif filename.lower().endswith('.bmp'):
            content_type = 'image/bmp'
        elif filename.lower().endswith('.tiff'):
            content_type = 'image/tiff'
        
        return send_file(image_path, mimetype=content_type)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Update Embryo Record
@doctor_bp.route('/Embryo/UPDATE', methods=['POST'])
def Embryo_Update():
    # Get current user ID from session
    current_user_id = session.get('user_id')
    
    if not current_user_id:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        # Get form data
        embryo_data = {
            'embryo_id': request.form.get('embryo_id'),
            'date': request.form.get('date'),
            'contact': request.form.get('contact', ''),
            'blastocyst_grade': request.form.get('blastocyst_grade'),
            'pgt_a_grade': request.form.get('pgt_a_grade', '') or None,  # Convert empty string to None
            'live_birth': request.form.get('live_birth', '') or None,  # Convert empty string to None
        }
        
        # Get annotations
        annotations_json = request.form.get('annotations')
        if not annotations_json:
            return jsonify({"error": "No annotations provided"}), 400
        
        annotations_csv = json.loads(annotations_json)
        
        # Update embryo record
        success, message = doctor.updateEmbryo(embryo_data['embryo_id'], embryo_data, annotations_csv, current_user_id)
        
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict Transitions
@doctor_bp.route('/Embryo/PREDICT', methods=['POST'])
def Embryo_Predict():
    """
    AI Prediction Endpoint for Developmental Transitions
    
    This endpoint processes embryo image sequences through AI models to predict
    developmental transitions. It supports two different models based on the
    number of frames: ResNet18 for 8-frame sequences and TimeSformer for
    32-frame sequences.
    
    Request Format (JSON):
        {
            "frame_count": 8,  # or 32
            "images": ["base64_image1", "base64_image2", ...]
        }
    
    Response Format (JSON):
        {
            "success": true,
            "predictions": [0, 1, 0, 1],  # Binary predictions
            "is_random": false  # True if random predictions used
        }
    
    Authentication:
        - Requires valid doctor session
        - Validates user_id from session
    
    Model Selection:
        - 8 frames: ResNet18 model (Results/resnet18/best_model.pth)
        - 32 frames: TimeSformer model (Results/timesformer/best_model.pth)
    
    Image Processing:
        - Decodes base64 images
        - Resizes to 224x224 pixels
        - Applies sliding window analysis
        - Returns binary predictions (0=no transition, 1=transition)
    
    Fallback Behavior:
        - Returns random predictions if model unavailable
        - Sets is_random flag for frontend notification
    
    Error Handling:
        - 401: User not authenticated
        - 400: Insufficient images or validation errors
        - 500: Server errors during processing
    
    Returns:
        JSON response with prediction results or error message
    """
        
    try:
        current_user_id = session.get('user_id')
        if not current_user_id:
            return jsonify({'success': False, 'message': 'User not authenticated'}), 401
        
        # Get prediction parameters
        data = request.get_json()
        frame_count = int(data.get('frame_count', 8))  # 8 or 32
        images = data.get('images', [])  # Base64 encoded images
        
        if len(images) < frame_count:
            return jsonify({'success': False, 'message': f'Need at least {frame_count} images for prediction, got {len(images)}'}), 400
        
        # Call prediction function
        result = doctor.predictTransitions(images, frame_count, current_user_id)
        
        if len(result) == 3:  # New format with random flag
            success, predictions, is_random = result
            if success:
                return jsonify({'success': True, 'predictions': predictions, 'is_random': is_random})
            else:
                return jsonify({'success': False, 'message': predictions}), 400
        else:  # Old format for backward compatibility
            success, predictions = result
            if success:
                return jsonify({'success': True, 'predictions': predictions, 'is_random': False})
            else:
                return jsonify({'success': False, 'message': predictions}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error predicting transitions: {str(e)}'}), 500