"""
========================================
EMBRYO ANALYSIS WEB APPLICATION
========================================

Main Flask application entry point for the Embryo Analysis System.
This module initializes the Flask application, configures settings,
and sets up the routing structure for the web interface.

Features:
- Session-based authentication system
- Role-based access control (Admin/Doctor)
- File upload management for embryo images
- AI-powered developmental transition prediction
- RESTful API endpoints for frontend integration
- Database integration with PostgreSQL
- Organized file storage system

Architecture:
- Flask-based MVC architecture
- Blueprint-based modular routing
- Session management for user authentication
- Environment-based configuration
- Error handling and logging

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import os

from Classes.Admin import Admin
from Classes.Doctor import Doctor
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from HandleAccess import GlobalDataBase
from Routes.Admin_Routes import admin_bp
from Routes.Doctor_Routes import doctor_bp

# Load environment variables from .env file
load_dotenv()

def ensure_directories_exist():
    """
    Ensure the embryo images directory exists, create it if it doesn't
    
    This function validates and creates the required directory structure
    for storing embryo images. It checks for the EMBRYO_IMAGES_PATH
    environment variable and creates the directory if it doesn't exist.
    
    Returns:
        bool: True if directory exists or was created successfully,
              False if an error occurred during creation
    
    Environment Variables:
        EMBRYO_IMAGES_PATH (str): Path to embryo images directory
                                 Default: "C:/Embryo_images"
    
    Raises:
        OSError: If directory creation fails due to permissions
        Exception: For other unexpected errors during directory operations
    """
    # Get directory path from environment or use default
    directory = os.getenv('EMBRYO_IMAGES_PATH', r"C:/Embryo_images")
    
    try:
        if os.path.exists(directory):
            pass  # Directory exists
        else:
            os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        return False

# Ensure directories exist when app is created
if ensure_directories_exist():
    pass  # All directories ready
else:
    exit(1)

# Initialize Flask application with configuration
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.urandom(24)

# Configure session management for security and persistence
# SESSION_COOKIE_SECURE: Set to True in production with HTTPS
# SESSION_COOKIE_HTTPONLY: Prevents XSS attacks by making cookies inaccessible to JavaScript
# SESSION_COOKIE_SAMESITE: Provides CSRF protection
# PERMANENT_SESSION_LIFETIME: 24 hours session duration
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours

# Configure file upload settings
# MAX_CONTENT_LENGTH: Maximum file size limit (100MB) to prevent DoS attacks
# UPLOAD_FOLDER: Directory for storing uploaded embryo images
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = os.getenv('EMBRYO_IMAGES_PATH', r"C:/Embryo_images")


# Initialize business logic classes with session management
global_instance = GlobalDataBase(session)  # Authentication and user management
admin = Admin(session)                      # Admin operations
doctor = Doctor(session)                    # Doctor and embryo operations

# Register Flask blueprints for modular routing
app.register_blueprint(admin_bp)   # Admin-specific endpoints (/Admin/*)
app.register_blueprint(doctor_bp)  # Doctor-specific endpoints (/Doctor/*)

@app.route('/')
def Log_In():
    """
    Serve the main login page
    
    This route renders the login template which provides the authentication
    interface for both Admin and Doctor users.
    
    Returns:
        str: Rendered HTML template for the login page
    """
    return render_template('Login.html')


@app.route('/Submit-Info-Sign-In', methods=['POST'])
def submit_basic_IN():
    """
    Handle user authentication
    
    This endpoint processes login credentials submitted from the login form.
    It validates the username and password against the database and sets
    up the user session if authentication is successful.
    
    Request Parameters:
        Username (str): User's username/email
        Password (str): User's password
    
    Returns:
        dict: JSON response containing authentication status
        {
            "exists": bool,      # True if credentials are valid
            "user_type": str     # "Admin" or "Doctor" if valid, null otherwise
        }
    
    Session Data Set:
        - user_id: Database ID of the authenticated user
        - first_name: User's first name
        - last_name: User's last name
        - role: User's role (Admin/Doctor)
        - global_access: Doctor's global access level (if applicable)
    """
    Username = request.form.get('Username')
    Password = request.form.get('Password')
    exists, user_type = global_instance.retrieve_data_from_database(Username, Password)
    
    return jsonify({'exists': exists, 'user_type': user_type})


@app.route('/Log_Out')
def Log_out():
    """
    Handle user logout
    
    This endpoint clears all session data and redirects the user
    back to the login page, effectively logging them out of the system.
    
    Returns:
        str: Rendered HTML template for the login page
    
    Session Actions:
        - Clears all session variables
        - Invalidates user session
        - Removes authentication state
    """
    session.clear()
    return render_template('Login.html')


if __name__ == '__main__':
    app.run(debug=True)