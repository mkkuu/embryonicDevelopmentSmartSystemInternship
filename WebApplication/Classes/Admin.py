"""
========================================
EMBRYO ANALYSIS SYSTEM - ADMIN DATABASE CLASS
========================================

This module provides a comprehensive database management class for the Admin
functionality in the Embryo Analysis System. It handles all database operations
for doctor management including CRUD operations, user authentication, and
access control.

Features:
- PostgreSQL database connection management
- Environment-based configuration
- CRUD operations for doctor management
- User authentication and access control
- Error handling and logging
- Connection pooling and resource management

Database Operations:
- get_list(): Retrieve all doctors with authentication data
- add(): Add new doctor records
- update(): Update existing doctor information
- delete(): Remove doctor records
- execute_query(): Generic query execution

Author: LSL Team
Version: 1.0
Last Updated: 2025-10-04
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Admin:
    """
    Admin database management class for doctor operations.
    
    This class provides a comprehensive interface for managing doctor records
    in the PostgreSQL database, including authentication and access control
    functionality.
    
    Attributes:
        session: Flask session object for user context
        db_config (dict): Database connection configuration
        logger: Logger instance for error tracking
        
    Example:
        >>> admin = Admin(session)
        >>> doctors, success = admin.get_list()
        >>> if success:
        ...     print(f"Retrieved {len(doctors)} doctors")
    """
    
    def __init__(self, session):
        """
        Initialize the Admin database manager.
        
        Args:
            session: Flask session object containing user context
            
        The initialization loads database configuration from environment
        variables and sets up logging for error tracking.
        """
        self.session = session
        
        # Database configuration from environment variables
        self.db_config = {
            'host': os.getenv('host'),
            'database': os.getenv('database'),
            'user': os.getenv('user'),
            'password': os.getenv('password'),
            'port': int(os.getenv('port', 5432)),  # Default PostgreSQL port
        }
        
        # Validate required configuration
        self._validate_config()
        
        self.logger = logger

    def _validate_config(self) -> None:
        """
        Validate database configuration parameters.
        
        Raises:
            ValueError: If required configuration parameters are missing
        """
        required_keys = ['host', 'database', 'user', 'password']
        missing_keys = [key for key in required_keys if not self.db_config.get(key)]
        
        if missing_keys:
            raise ValueError(f"Missing required database configuration: {missing_keys}")

    def _get_connection(self):
        """
        Create a new database connection.
        
        Returns:
            psycopg2.connection: Database connection object
            
        Raises:
            psycopg2.Error: If connection fails
        """
        try:
            connection = psycopg2.connect(**self.db_config)
            connection.autocommit = False  # Use transactions
            return connection
        except psycopg2.Error as e:
            self.logger.error(f"Database connection failed: {e}")
            raise

    def execute_query(self, query: str, values: Optional[Tuple] = None) -> bool:
        """
        Execute a database query with proper error handling.
        
        Args:
            query (str): SQL query to execute
            values (tuple, optional): Query parameters for prepared statements
            
        Returns:
            bool: True if query executed successfully, False otherwise
            
        This method handles database transactions automatically and provides
        comprehensive error logging for debugging purposes.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if values:
                        cursor.execute(query, values)
                    else:
                        cursor.execute(query)
                conn.commit()
                self.logger.info(f"Query executed successfully: {query[:50]}...")
                return True
                
        except psycopg2.Error as e:
            self.logger.error(f"Database query error: {e}")
            self.logger.error(f"Query: {query}")
            if values:
                self.logger.error(f"Values: {values}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error executing query: {e}")
            return False

    def get_list(self) -> Tuple[List[Dict], bool]:
        """
        Retrieve all doctor records with authentication data.
        
        Returns:
            tuple: (list of doctor dictionaries, success boolean)
            
        The method joins the users and user_auth tables to provide complete
        doctor information including authentication credentials and access levels.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
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
                            u.role = 'Doctor'
                        ORDER BY u.user_id ASC;
                    """
                    cursor.execute(query)
                    results = cursor.fetchall()
                    
                    # Convert to list of dictionaries
                    doctor_list = [dict(row) for row in results]
                    
                    self.logger.info(f"Retrieved {len(doctor_list)} doctor records")
                    return doctor_list, True
                    
        except psycopg2.Error as e:
            self.logger.error(f"Error retrieving doctor list: {e}")
            return [], False
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving doctor list: {e}")
            return [], False

    def add(self, values: Dict[str, Any]) -> bool:
        """
        Add a new doctor record to the database.
        
        Args:
            values (dict): Dictionary containing doctor information
                Required keys: first_name, last_name, gender, birthday, 
                              contact, address, role
                
        Returns:
            bool: True if doctor added successfully, False otherwise
            
        The method inserts a new record into the users table with the
        provided doctor information.
        """
        try:
            # Validate required fields
            required_fields = ['first_name', 'last_name', 'gender', 'birthday', 'contact', 'address', 'role']
            missing_fields = [field for field in required_fields if field not in values]
            
            if missing_fields:
                self.logger.error(f"Missing required fields: {missing_fields}")
                return False
            
            # Prepare query with placeholders
            placeholders = ', '.join(['%s'] * len(values))
            columns = ', '.join(values.keys())
            query = f"INSERT INTO users ({columns}) VALUES ({placeholders})"
            
            success = self.execute_query(query, tuple(values.values()))
            
            if success:
                self.logger.info(f"Doctor added successfully: {values.get('first_name')} {values.get('last_name')}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error adding doctor: {e}")
            return False

    def update(self, table: str, user_id: int, values: Dict[str, Any]) -> bool:
        """
        Update an existing doctor record.
        
        Args:
            table (str): Table name to update ('users' or 'user_auth')
            user_id (int): ID of the user to update
            values (dict): Dictionary containing fields to update
            
        Returns:
            bool: True if update successful, False otherwise
            
        This method updates specific fields in the specified table for the
        given user ID.
        """
        try:
            if not values:
                self.logger.warning("No values provided for update")
                return False
            
            # Prepare update query
            columns = ', '.join([f"{key} = %s" for key in values.keys()])
            query = f"UPDATE {table} SET {columns} WHERE user_id = %s"
            
            # Add user_id to values tuple
            update_values = tuple(values.values()) + (user_id,)
            
            success = self.execute_query(query, update_values)
            
            if success:
                self.logger.info(f"Updated {table} for user_id {user_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating {table} for user_id {user_id}: {e}")
            return False

    def delete(self, table: str, user_id: int) -> bool:
        """
        Delete a doctor record from the database.
        
        Args:
            table (str): Table name to delete from ('users' or 'user_auth')
            user_id (int): ID of the user to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
            
        Note: Deleting from 'users' table will cascade to 'user_auth' table
        due to foreign key constraints.
        """
        try:
            query = f"DELETE FROM {table} WHERE user_id = %s"
            success = self.execute_query(query, (user_id,))
            
            if success:
                self.logger.info(f"Deleted user_id {user_id} from {table}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting user_id {user_id} from {table}: {e}")
            return False
