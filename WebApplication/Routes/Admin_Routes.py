"""
========================================
EMBRYO ANALYSIS SYSTEM - ADMIN ROUTES
========================================

This module provides Flask routes for the Admin functionality in the Embryo
Analysis System. It handles all HTTP endpoints for doctor management including
CRUD operations, access control, and data validation.

Features:
- RESTful API endpoints for doctor management
- JSON request/response handling
- Input validation and sanitization
- Error handling with appropriate HTTP status codes
- Session management and user context
- Integration with Admin database class

API Endpoints:
- GET /Admin/Doctor: Render doctor management page
- POST /Admin/Doctor/ADD: Add new doctor
- POST/PUT /Admin/Doctor/UPDATE: Update existing doctor
- POST /Admin/Doctor/ACCESS: Update doctor access level
- DELETE /Admin/Doctor/DELETE/<id>: Delete doctor
- GET /Admin/Doctor/LIST: Get all doctors

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import datetime
import logging
from typing import Any, Dict, Tuple

from Classes.Admin import Admin
from flask import Blueprint, jsonify, render_template, request, session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Admin instance and create blueprint
admin = Admin(session)
admin_bp = Blueprint('admin_bp', __name__, url_prefix='/Admin')


def validate_doctor_data(data: Dict[str, Any], required_fields: list) -> Tuple[bool, str]:
    """
    Validate doctor data for required fields and data types.
    
    Args:
        data (dict): Doctor data to validate
        required_fields (list): List of required field names
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not data:
        return False, "No data provided"
    
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    
    return True, ""


def format_doctor_response(doctor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format doctor data for JSON response.
    
    Args:
        doctor_data (dict): Raw doctor data from database
        
    Returns:
        dict: Formatted doctor data for frontend
    """
    formatted_data = {
        "ID": doctor_data.get('user_id'),
        "Name": doctor_data.get('first_name', ''),
        "Last": doctor_data.get('last_name', ''),
        "Gender": doctor_data.get('gender', ''),
        "BirthDay": doctor_data.get('birthday'),
        "Contact": doctor_data.get('contact', ''),
        "Address": doctor_data.get('address', ''),
        "Password": doctor_data.get('password', ''),
        "GlobalAccess": doctor_data.get('global_access', False)
    }
    
    # Format date if present
    if isinstance(formatted_data["BirthDay"], datetime.date):
        formatted_data["BirthDay"] = formatted_data["BirthDay"].isoformat()
    
    return formatted_data


@admin_bp.route('/Register', endpoint='Register')
def register():
    """
    Register route placeholder.
    
    This route is reserved for future registration functionality.
    Currently returns a pass statement as a placeholder.
    """
    # TODO: Implement registration functionality
    pass


@admin_bp.route('/Doctor', endpoint='Doctor')
def doctor_page():
    """
    Render the doctor management page.
    
    Returns:
        str: Rendered HTML template for doctor management interface
        
    This route renders the main doctor management page with user context
    information from the session.
    """
    try:
        name = session.get('first_name', '')
        last_name = session.get('last_name', '')
        
        logger.info(f"Rendering doctor page for user: {name} {last_name}")
        
        return render_template('Admin/Doctor.html',
                             name=name,
                             lastName=last_name)
                             
    except Exception as e:
        logger.error(f"Error rendering doctor page: {e}")
        return render_template('Admin/Doctor.html',
                             name='',
                             lastName='')


@admin_bp.route('/Doctor/ADD', methods=['POST'])
def doctor_add():
    """
    Add a new doctor to the system.
    
    Request Body:
        JSON object containing doctor information:
        - Name (str): First name
        - Last (str): Last name  
        - Gender (str): Gender
        - BirthDay (str): Birth date
        - Contact (str): Email or phone
        - Address (str): Address
        
    Returns:
        JSON response with success/error message and HTTP status code
        
    This endpoint validates input data and creates a new doctor record
    in the database with role 'Doctor'.
    """
    try:
        data = request.get_json()
        
        # Validate JSON data
        if not data:
            logger.warning("Invalid JSON received in doctor_add")
            return jsonify({"error": "Invalid JSON data"}), 400
        
        # Validate required fields
        required_fields = ['Name', 'Last', 'Gender', 'BirthDay', 'Contact', 'Address']
        is_valid, error_message = validate_doctor_data(data, required_fields)
        
        if not is_valid:
            logger.warning(f"Validation failed in doctor_add: {error_message}")
            return jsonify({"error": error_message}), 400
        
        # Prepare data for database insertion
        doctor_data = {
            'first_name': data.get('Name'),
            'last_name': data.get('Last'),
            'gender': data.get('Gender'),
            'birthday': data.get('BirthDay'),
            'contact': data.get('Contact'),
            'address': data.get('Address'),
            'role': 'Doctor'
        }
        
        # Add doctor to database
        success = admin.add(doctor_data)
        
        if success:
            logger.info(f"Doctor added successfully: {doctor_data['first_name']} {doctor_data['last_name']}")
            return jsonify({"message": "Doctor added successfully"}), 200
        else:
            logger.error("Failed to add doctor to database")
            return jsonify({"error": "Failed to add doctor to database"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in doctor_add: {e}")
        return jsonify({"error": "Internal server error"}), 500


@admin_bp.route('/Doctor/UPDATE', methods=['POST', 'PUT'])
def doctor_update():
    """
    Update an existing doctor's information.
    
    Request Body:
        JSON object containing doctor information to update:
        - ID (int): Doctor ID
        - Name (str): First name
        - Last (str): Last name
        - Gender (str): Gender
        - BirthDay (str): Birth date
        - Contact (str): Email or phone
        - Address (str): Address
        - Password (str): Password
        
    Returns:
        JSON response with success/error message and HTTP status code
        
    This endpoint updates both the users table and user_auth table
    with the provided information.
    """
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("Invalid JSON received in doctor_update")
            return jsonify({"error": "Invalid or missing JSON data"}), 400
        
        # Validate required fields
        required_fields = ['ID', 'Name', 'Last', 'Gender', 'BirthDay', 'Contact', 'Address', 'Password']
        is_valid, error_message = validate_doctor_data(data, required_fields)
        
        if not is_valid:
            logger.warning(f"Validation failed in doctor_update: {error_message}")
            return jsonify({"error": error_message}), 400
        
        doctor_id = data.get('ID')
        
        # Update users table
        users_data = {
            'first_name': data.get('Name'),
            'last_name': data.get('Last'),
            'gender': data.get('Gender'),
            'birthday': data.get('BirthDay'),
            'contact': data.get('Contact'),
            'address': data.get('Address')
        }
        
        users_success = admin.update('users', doctor_id, users_data)
        
        # Update user_auth table
        auth_data = {
            'username': data.get('Contact'),
            'password': data.get('Password')
        }
        
        auth_success = admin.update('user_auth', doctor_id, auth_data)
        
        if users_success and auth_success:
            logger.info(f"Doctor updated successfully: ID {doctor_id}")
            return jsonify({"message": "Doctor updated successfully"}), 200
        else:
            logger.error(f"Failed to update doctor: ID {doctor_id}")
            return jsonify({"error": "Failed to update doctor information"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in doctor_update: {e}")
        return jsonify({"error": "Internal server error"}), 500


@admin_bp.route('/Doctor/ACCESS', methods=['POST'])
def doctor_update_access():
    """
    Update a doctor's global access level.
    
    Request Body:
        JSON object containing:
        - ID (int): Doctor ID
        - GlobalAccess (bool): New access level
        
    Returns:
        JSON response with success/error message and HTTP status code
        
    This endpoint updates the global_access field for the specified doctor.
    """
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("Invalid JSON received in doctor_update_access")
            return jsonify({"error": "Invalid or missing JSON data"}), 400
        
        doctor_id = data.get('ID')
        global_access = str(data.get('GlobalAccess')).upper() == 'TRUE'
        
        if not doctor_id:
            return jsonify({"error": "Doctor ID is required"}), 400
        
        # Update global access
        access_data = {'global_access': global_access}
        success = admin.update('users', doctor_id, access_data)
        
        if success:
            logger.info(f"Doctor access updated: ID {doctor_id}, Access: {global_access}")
            return jsonify({"message": "Doctor access updated successfully"}), 200
        else:
            logger.error(f"Failed to update doctor access: ID {doctor_id}")
            return jsonify({"error": "Failed to update doctor access"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in doctor_update_access: {e}")
        return jsonify({"error": "Internal server error"}), 500


@admin_bp.route('/Doctor/DELETE/<int:row_id>', methods=['DELETE'])
def doctor_delete(row_id):
    """
    Delete a doctor from the system.
    
    Args:
        row_id (int): ID of the doctor to delete
        
    Returns:
        JSON response with success/error message and HTTP status code
        
    This endpoint removes the doctor record from the users table,
    which will cascade to the user_auth table due to foreign key constraints.
    """
    try:
        if not row_id:
            return jsonify({"error": "Doctor ID is required"}), 400
        
        success = admin.delete("users", row_id)
        
        if success:
            logger.info(f"Doctor deleted successfully: ID {row_id}")
            return jsonify({"message": "Doctor deleted successfully"}), 200
        else:
            logger.error(f"Failed to delete doctor: ID {row_id}")
            return jsonify({"error": "Failed to delete doctor"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in doctor_delete: {e}")
        return jsonify({"error": "Internal server error"}), 500


@admin_bp.route('/Doctor/LIST', methods=['GET'])
def doctor_list():
    """
    Retrieve all doctors from the system.
    
    Returns:
        JSON response containing list of all doctors with HTTP status code
        
    This endpoint returns a formatted list of all doctors with their
    complete information including authentication data.
    """
    try:
        doctors_data, success = admin.get_list()
        
        if not success:
            logger.error("Failed to retrieve doctor list from database")
            return jsonify({"error": "Failed to retrieve doctor list"}), 500
        
        # Format doctor data for frontend
        formatted_doctors = []
        for doctor in doctors_data:
            formatted_doctor = format_doctor_response(doctor)
            formatted_doctors.append(formatted_doctor)
        
        logger.info(f"Retrieved {len(formatted_doctors)} doctors")
        return jsonify(formatted_doctors), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in doctor_list: {e}")
        return jsonify({"error": "Internal server error"}), 500