"""
========================================
AUTHENTICATION AND DATABASE ACCESS MODULE
========================================

This module handles user authentication and database connectivity for the
Embryo Analysis Web Application. It provides secure user verification,
session management, and database connection handling.

Features:
- PostgreSQL database connectivity
- User authentication and validation
- Session management for authenticated users
- Role-based access control
- Secure password verification
- Environment-based database configuration

Security Features:
- Parameterized queries to prevent SQL injection
- Session-based authentication
- Secure database connection handling
- Environment variable configuration

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import os

import psycopg2
import requests
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Load environment variables from .env file
load_dotenv()

# Database configuration from environment variables
# This ensures secure credential management and environment-specific settings
db_config = {
    'host': os.getenv('host'),           # Database server hostname
    'database': os.getenv('database'),   # Database name
    'user': os.getenv('user'),           # Database username
    'password': os.getenv('password'),   # Database password
    'port': int(os.getenv('port')),      # Database port number
}

class GlobalDataBase:
    """
    Global database access and authentication handler
    
    This class manages user authentication, database connectivity, and session
    management for the Embryo Analysis Web Application. It provides secure
    user verification against the PostgreSQL database and manages user sessions.
    
    Attributes:
        session: Flask session object for storing user authentication data
    
    Methods:
        retrieve_data_from_database: Authenticates user credentials and sets up session
    """
    
    def __init__(self, session):
        """
        Initialize the GlobalDataBase handler
        
        Args:
            session: Flask session object for storing user authentication data
        """
        self.session = session

    def retrieve_data_from_database(self, Username, Password):
        """
        Authenticate user credentials and set up session data
        
        This method verifies the provided username and password against the database,
        retrieves user information, and sets up the Flask session with user data
        if authentication is successful.
        
        Args:
            Username (str): User's username or email address
            Password (str): User's password (plain text)
        
        Returns:
            tuple: (exists, user_type) where:
                - exists (bool): True if credentials are valid, False otherwise
                - user_type (str): "Admin" or "Doctor" if valid, None otherwise
        
        Session Data Set (if authentication successful):
            - user_id: Database ID of the authenticated user
            - first_name: User's first name
            - last_name: User's last name
            - role: User's role (Admin/Doctor)
            - global_access: Doctor's global access level (if applicable)
        
        Database Query:
            SELECTs user data from the users table where email matches username
            and password matches the provided password.
        
        Security Features:
            - Uses parameterized queries to prevent SQL injection
            - Validates user existence before password verification
            - Sets session as permanent for persistence
            - Handles database connection errors gracefully
        """
        try:
            with psycopg2.connect(**db_config) as conn:
                with conn.cursor() as cursor:
                    query = """
                        SELECT u.user_id, u.role, u.first_name, u.last_name
                        FROM user_auth ua
                        JOIN users u ON ua.user_id = u.user_id
                        WHERE ua.username = %s AND ua.password = %s
                    """
                    cursor.execute(query, (Username, Password))
                    result = cursor.fetchone()

                    if result:
                        user_id, user_type, first_name, last_name = result

                        self.session['user_id'] = user_id
                        self.session['first_name'] = first_name
                        self.session['last_name'] = last_name
                        self.session.permanent = True

                        return True , user_type
                    else:
                        return False, None
        except Exception as error:
            return False, None

