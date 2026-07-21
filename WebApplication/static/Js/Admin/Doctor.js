/**
 * ========================================
 * EMBRYO ANALYSIS SYSTEM - DOCTOR MANAGEMENT
 * ========================================
 * 
 * This JavaScript module provides comprehensive frontend functionality for the
 * Doctor Management system in the Embryo Analysis System. It handles all
 * client-side operations including CRUD operations, data validation, UI
 * interactions, and API communication.
 * 
 * Features:
 * - Complete CRUD operations for doctor management
 * - Real-time data validation and error handling
 * - Dynamic table with sorting, filtering, and pagination
 * - Modal dialogs for user feedback and confirmations
 * - Password visibility toggle functionality
 * - Responsive UI with loading states
 * - API integration with Flask backend
 * 
 * Architecture:
 * - Modular design with separated concerns
 * - State management for UI and data
 * - Event-driven programming
 * - Error handling with user feedback
 * - Performance optimization with DOM caching
 * 
 * Author: LSL Team
 * Version: 1.0
 * Last Updated: 2025-10-04
 */

document.addEventListener("DOMContentLoaded", function () {
    /**
     * ========================================
     * CONFIGURATION AND CONSTANTS
     * ========================================
     */
    
    const CONFIG = {
        API_ENDPOINTS: {
            LIST: '/Admin/Doctor/LIST',
            ADD: '/Admin/Doctor/ADD',
            UPDATE: '/Admin/Doctor/UPDATE',
            DELETE: '/Admin/Doctor/DELETE',
            ACCESS: '/Admin/Doctor/ACCESS'
        },
        VALIDATION: {
            REQUIRED_FIELDS: ['Name', 'Last', 'Gender', 'BirthDay', 'Contact', 'Address']
        },
        UI: {
            MODAL_DELAY: 3000,
            DEBOUNCE_DELAY: 300,
            MAX_RETRY_ATTEMPTS: 3
        }
    };

    /**
     * ========================================
     * DOM ELEMENT REFERENCES
     * ========================================
     */
    
    // Form elements
    const doctorForm = document.querySelector('.doctor-form');
    const idField = document.getElementById('ID');
    const nameField = document.getElementById('Name');
    const lastField = document.getElementById('Last');
    const genderField = document.getElementById('Gender');
    const birthDayField = document.getElementById('BirthDay');
    const contactField = document.getElementById('Contact');
    const addressField = document.getElementById('Address');
    const passwordField = document.getElementById('Password');
    const togglePassword = document.getElementById('togglePassword');
    const inputsToVerify = [nameField, lastField, genderField, birthDayField, contactField, addressField];

    // Action buttons
    const clearButton = document.getElementById('Clear');
    const addButton = document.getElementById('Add');
    const updateButton = document.getElementById('Update');
    const deleteButton = document.getElementById('Delete');
    
    // Table elements
    const tableHeader = document.querySelector(".doctors-table thead");
    const tableBody = document.querySelector(".doctors-table tbody");
    const searchInput = document.getElementById('search');
    const noDataRow = document.querySelector('.no-data-row');
    const entriesPerPageSelect = document.getElementById('entries-per-page');
    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');
    const entriesInfo = document.getElementById('entries-info');
    
    // Modal elements
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalMessage = document.getElementById('modal-message');
    const modalButtons = document.getElementById('modal-buttons');

    /**
     * ========================================
     * STATE MANAGEMENT
     * ========================================
     */
    
    let originalAllDoctors = [];
    let allDoctors = []; 
    let currentDoctorId = null;
    let currentPage = 1;
    let rowsPerPage = parseInt(entriesPerPageSelect.value, 10);
    let sortColumn = 'ID';
    let sortDirection = 'asc';
    
    /**
     * ========================================
     * UTILITY FUNCTIONS
     * ========================================
     */
    
    /**
     * Debounce function to limit API calls
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} Debounced function
     */
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }


    /**
     * Format error message for display
     * @param {string} field - Field name
     * @param {string} error - Error type
     * @returns {string} Formatted error message
     */
    function formatErrorMessage(field, error) {
        const messages = {
            'required': `${field} is required`,
            'invalid': `Invalid ${field} format`,
            'duplicate': `${field} already exists`
        };
        return messages[error] || `Invalid ${field}`;
    }

    /**
     * ========================================
     * DYNAMIC ERROR CLEARING
     * ========================================
     */
    
    // Remove red error border on input
    [...inputsToVerify, passwordField].forEach(input => {
        input.addEventListener('input', () => {
            if (input.classList.contains('input-error')) {
                input.classList.remove('input-error');
            }
        });
    });

    /**
     * ========================================
     * MODAL DIALOG FUNCTIONS
     * ========================================
     */
    
    /**
     * Display modal dialog with customizable buttons
     * @param {string} title - Modal title
     * @param {string} message - Modal message
     * @param {Array} buttons - Array of button configurations
     */
    function showModal(title, message, buttons = [{text: 'OK', class: 'btn-add'}]) {
        modalTitle.textContent = title;
        modalMessage.textContent = message;
        modalButtons.innerHTML = '';

        buttons.forEach(btnInfo => {
            const button = document.createElement('button');
            button.textContent = btnInfo.text;
            button.className = `btn ${btnInfo.class}`;
            button.onclick = () => {
                modal.classList.remove('show');
                if (btnInfo.onClick) {
                    btnInfo.onClick();
                }
            };
            modalButtons.appendChild(button);
        });
        
        modal.classList.add('show');
        
        // Add default OK button behavior if no custom buttons
        if (buttons.length === 1 && buttons[0].text === 'OK') {
            const okButton = modalButtons.querySelector('button');
            okButton.onclick = () => modal.classList.remove('show');
        }
    }
    
    /**
     * ========================================
     * API COMMUNICATION FUNCTIONS
     * ========================================
     */
    
    /**
     * Fetch all doctors from the server
     * @returns {Promise<void>} Promise that resolves when data is loaded
     */
    async function fetchAllDoctors() {
        try {
            const response = await fetch(CONFIG.API_ENDPOINTS.LIST);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            originalAllDoctors = [...data];
            allDoctors = [...data];
            sortData();
            updateSortIcons();
            displayPage();
            
        } catch (error) {
            showModal(
                "Error", 
                "Could not load doctor list from the server. Make sure the backend is running and the /Admin/Doctor/LIST route is available."
            );
            noDataRow.querySelector('.no-data').textContent = 'Error loading data.';
        }
    }

    /**
     * Update doctor's global access status
     * @param {number} doctorId - ID of the doctor to update
     * @param {boolean} newStatus - New access status
     * @returns {Promise<void>} Promise that resolves when update is complete
     */
    async function updateDoctorAccess(doctorId, newStatus) {
        try {
            const response = await fetch(CONFIG.API_ENDPOINTS.ACCESS, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ID: doctorId, GlobalAccess: newStatus })
            });

            if (!response.ok) {
                throw new Error('Failed to update access status.');
            }
            
            // Update local data
            const doctor = originalAllDoctors.find(d => d.ID == doctorId);
            if (doctor) {
                doctor.GlobalAccess = newStatus;
            }

        } catch (error) {
            showModal(
                "Error", 
                "Could not update the doctor's access status. Please try again."
            );
            
            // Revert the checkbox on failure
            const checkbox = tableBody.querySelector(`input[data-doctor-id="${doctorId}"]`);
            if (checkbox) {
                checkbox.checked = !newStatus;
            }
        }
    }


    
    /**
     * ========================================
     * CRUD OPERATIONS
     * ========================================
     */
    
    /**
     * Add a new doctor to the system
     * @returns {Promise<void>} Promise that resolves when doctor is added
     */
    async function addDoctor() {
        // Check if we're in update mode
        if (currentDoctorId) {
            showModal("Duplicate Entry", "This Doctor already exists. Please use Update instead.");
            return;
        }

        // Validate form inputs
        if (!validateInputs()) {
            return;
        }
        
        // Prepare form data
        const formData = new FormData(doctorForm);
        const doctorData = Object.fromEntries(formData.entries());

        try {
            const response = await fetch(CONFIG.API_ENDPOINTS.ADD, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(doctorData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Server error');
            }
            
            showModal("Success", "Doctor added successfully.");
            clearFields();
            await fetchAllDoctors(); // Refresh table

        } catch (error) {
            showModal("Error", `Could not add doctor: ${error.message}`);
        }
    }


    /**
     * Update an existing doctor's information
     * @returns {Promise<void>} Promise that resolves when doctor is updated
     */
    async function updateDoctor() {
        // Check if a doctor is selected
        if (!currentDoctorId) {
            showModal("No Selection", "Please select a doctor from the list to update.");
            return;
        }
        
        // Validate form inputs
        if (!validateInputs()) {
            return;
        }

        // Validate password field for updates
        if (!passwordField.value.trim()) {
            showModal("Validation Error", "Password field cannot be empty when updating.");
            passwordField.classList.add('input-error');
            passwordField.focus();
            return;
        }

        // Prepare form data
        const formData = new FormData(doctorForm);
        const doctorData = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch(CONFIG.API_ENDPOINTS.UPDATE, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(doctorData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Server error');
            }
            
            showModal("Success", "Doctor information updated.");
            clearFields();
            await fetchAllDoctors(); // Refresh table
            
        } catch (error) {
            showModal("Error", `Could not update doctor: ${error.message}`);
        }
    }

    /**
     * Show confirmation dialog for doctor deletion
     */
    function confirmDelete() {
        if (!currentDoctorId) {
            showModal("No Selection", "Please select a doctor from the list to delete.");
            return;
        }
        
        showModal(
            "Confirm Deletion", 
            `Are you sure you want to delete Dr. ${nameField.value} ${lastField.value}?`, 
            [
                { text: 'Cancel', class: 'btn-clear' },
                { text: 'Delete', class: 'btn-delete', onClick: deleteDoctor }
            ]
        );
    }
    
    /**
     * Delete the selected doctor from the system
     * @returns {Promise<void>} Promise that resolves when doctor is deleted
     */
    async function deleteDoctor() {
        try {
            const response = await fetch(`${CONFIG.API_ENDPOINTS.DELETE}/${currentDoctorId}`, { 
                method: 'DELETE' 
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Server error');
            }
            
            showModal("Success", "Doctor deleted.");
            clearFields();
            await fetchAllDoctors(); // Refresh table
            
        } catch (error) {
            showModal("Error", `Could not delete doctor: ${error.message}`);
        }
    }

    
    /**
     * ========================================
     * FORM HANDLING & VALIDATION
     * ========================================
     */
    
    /**
     * Clear all form fields and reset UI state
     */
    function clearFields() {
        doctorForm.reset();
        birthDayField.type = 'text'; // Reset date field placeholder behavior
        passwordField.readOnly = true;
        passwordField.type = 'password';
        togglePassword.classList.remove('fa-eye-slash');
        togglePassword.classList.add('fa-eye');
        removeRowStyling();
        resetInputErrors();
        currentDoctorId = null;
    }

    /**
     * Validate all required form inputs
     * @returns {boolean} True if all inputs are valid, false otherwise
     */
    function validateInputs() {
        resetInputErrors();
        let isValid = true;
        
        for (const input of inputsToVerify) {
            const isInvalid = !input.value || input.value.trim() === '';
            if (isInvalid) {
                input.classList.add('input-error');
                if (isValid) {
                    input.focus();
                }
                isValid = false;
            }
        }
        
        
        if (!isValid) {
            showModal("Validation Error", "Please fill in all required fields with valid data.");
        }
        
        return isValid;
    }
    
    /**
     * Remove error styling from all input fields
     */
    function resetInputErrors() {
        [...inputsToVerify, addressField, passwordField].forEach(input => {
            input.classList.remove('input-error');
        });
    }

    /**
     * Populate form fields with doctor data
     * @param {Object} doctorData - Doctor data object
     */
    function populateFields(doctorData) {
        resetInputErrors();
        idField.value = doctorData.ID;
        nameField.value = doctorData.Name;
        lastField.value = doctorData.Last;
        genderField.value = doctorData.Gender;
        birthDayField.type = 'date';
        birthDayField.value = doctorData.BirthDay;
        contactField.value = doctorData.Contact;
        addressField.value = doctorData.Address;
        passwordField.value = doctorData.Password;
        passwordField.readOnly = false;
        currentDoctorId = doctorData.ID;
        
    }

    
    /**
     * ========================================
     * TABLE DISPLAY & PAGINATION
     * ========================================
     */
    
    /**
     * Display the current page of doctors in the table
     */
    function displayPage() {
        tableBody.innerHTML = ''; 
        const filteredDoctors = getFilteredDoctors();
        const start = (currentPage - 1) * rowsPerPage;
        const end = start + rowsPerPage;
        const paginatedDoctors = filteredDoctors.slice(start, end);

        if (paginatedDoctors.length > 0) {
            paginatedDoctors.forEach(doctor => {
                const row = document.createElement('tr');
                const isChecked = doctor.GlobalAccess ? 'checked' : '';
                row.innerHTML = `
                    <td>${doctor.ID}</td>
                    <td>${doctor.Name}</td>
                    <td>${doctor.Last}</td>
                    <td>${doctor.Gender}</td>
                    <td>${doctor.BirthDay}</td>
                    <td>${doctor.Contact}</td>
                    <td>${doctor.Address || 'N/A'}</td>
                    <td>********</td>
                    <td class="access-column">
                        <label class="switch">
                            <input type="checkbox" data-doctor-id="${doctor.ID}" ${isChecked}>
                            <span class="slider"></span>
                        </label>
                    </td>
                `;
                
                // Add row click handler
                row.addEventListener('click', (e) => {
                    // Prevent row selection when clicking the toggle switch
                    if (e.target.closest('.access-toggle') || e.target.closest('.access-column')) {
                        return;
                    }
                    
                    // Toggle selection
                    if (row.classList.contains('selected')) {
                        clearFields();
                    } else {
                        populateFields(doctor);
                        highlightRow(row);
                    }
                });

                // Add toggle switch handler
                row.querySelector('.access-column input').addEventListener('change', (e) => {
                    updateDoctorAccess(e.target.dataset.doctorId, e.target.checked);
                });
                
                row.doctorData = doctor;
                tableBody.appendChild(row);
            });
        }

        updatePaginationControls(filteredDoctors.length);
        updateEntriesInfo(filteredDoctors.length);
        toggleNoDataRow(filteredDoctors.length);
    }

    /**
     * Get filtered doctors based on search term
     * @returns {Array} Filtered array of doctors
     */
    function getFilteredDoctors() {
        const searchTerm = searchInput.value.toLowerCase().trim();
        if (!searchTerm) return allDoctors;
        
        return allDoctors.filter(doctor => {
            const doctorCopy = {...doctor};
            delete doctorCopy.Password; // Exclude password from search
            delete doctorCopy.GlobalAccess; // Exclude access status from search
            return Object.values(doctorCopy).join(' ').toLowerCase().includes(searchTerm);
        });
    }
    
    /**
     * Sort doctors data based on current sort column and direction
     */
    function sortData() {
        if (!sortColumn) {
            allDoctors = [...originalAllDoctors];
            return;
        }
        
        allDoctors.sort((a, b) => {
            let valA = a[sortColumn];
            let valB = b[sortColumn];

            // Handle numeric sorting for ID and boolean for Access
            if (sortColumn === 'ID' || sortColumn === 'GlobalAccess') {
                valA = parseInt(valA, 10) || 0;
                valB = parseInt(valB, 10) || 0;
            } else if (typeof valA === 'string') {
                valA = valA.toLowerCase();
                valB = valB.toLowerCase();
            }

            if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
            if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
    }
    
    /**
     * Update sort icons to reflect current sort state
     */
    function updateSortIcons() {
        tableHeader.querySelectorAll('.sort-icons i').forEach(icon => {
            icon.classList.remove('active');
        });
        
        if (sortColumn) {
            const currentTh = tableHeader.querySelector(`th[data-column="${sortColumn}"]`);
            if (currentTh) { 
                if (sortDirection === 'asc') {
                    currentTh.querySelector('.fa-sort-up').classList.add('active');
                } else if (sortDirection === 'desc') {
                    currentTh.querySelector('.fa-sort-down').classList.add('active');
                }
            }
        }
    }

    /**
     * Highlight the selected table row
     * @param {HTMLElement} row - Row element to highlight
     */
    function highlightRow(row) {
        removeRowStyling();
        row.classList.add('selected');
    }

    /**
     * Remove highlighting from all table rows
     */
    function removeRowStyling() {
        const previous = document.querySelector('.doctors-table tbody tr.selected');
        if (previous) {
            previous.classList.remove('selected');
        }
    }
    
    /**
     * Toggle visibility of no-data row
     * @param {number} visibleRowCount - Number of visible rows
     */
    function toggleNoDataRow(visibleRowCount) {
        noDataRow.style.display = visibleRowCount === 0 ? 'table-row' : 'none';
        if (visibleRowCount === 0) {
            noDataRow.querySelector('.no-data').textContent = 'No Data Available in Table';
        }
    }
    
    /**
     * Update entries information display
     * @param {number} totalEntries - Total number of entries
     */
    function updateEntriesInfo(totalEntries) {
        const startEntry = totalEntries > 0 ? (currentPage - 1) * rowsPerPage + 1 : 0;
        const endEntry = Math.min(startEntry + rowsPerPage - 1, totalEntries);
        entriesInfo.textContent = `Showing ${startEntry} to ${endEntry} of ${totalEntries} entries`;
    }
    
    /**
     * Update pagination control states
     * @param {number} totalEntries - Total number of entries
     */
    function updatePaginationControls(totalEntries) {
        const totalPages = Math.ceil(totalEntries / rowsPerPage);
        prevButton.disabled = currentPage === 1;
        nextButton.disabled = currentPage === totalPages || totalPages === 0;
    }
    
    /**
     * ========================================
     * EVENT LISTENERS
     * ========================================
     */
    
    // Form action buttons
    clearButton.addEventListener('click', clearFields);
    addButton.addEventListener('click', addDoctor);
    updateButton.addEventListener('click', updateDoctor);
    deleteButton.addEventListener('click', confirmDelete);

    // Password visibility toggle
    togglePassword.addEventListener('click', function() {
        const isPassword = passwordField.type === 'password';
        if (isPassword) {
            passwordField.type = 'text';
            this.classList.remove('fa-eye');
            this.classList.add('fa-eye-slash');
        } else {
            passwordField.type = 'password';
            this.classList.remove('fa-eye-slash');
            this.classList.add('fa-eye');
        }
    });

    // Search functionality with debouncing
    const debouncedSearch = debounce(() => {
        currentPage = 1;
        displayPage();
    }, CONFIG.UI.DEBOUNCE_DELAY);
    
    searchInput.addEventListener('input', debouncedSearch);

    // Pagination controls
    entriesPerPageSelect.addEventListener('change', (e) => {
        rowsPerPage = parseInt(e.target.value, 10);
        currentPage = 1;
        displayPage();
    });
    
    prevButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            displayPage();
        }
    });
    
    nextButton.addEventListener('click', () => {
        const totalPages = Math.ceil(getFilteredDoctors().length / rowsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            displayPage();
        }
    });
    
    // Table sorting
    tableHeader.querySelectorAll('th[data-column]').forEach(th => {
        th.addEventListener('click', () => {
            const newSortColumn = th.dataset.column;
            
            if (sortColumn === newSortColumn) {
                if (sortDirection === 'asc') {
                    sortDirection = 'desc';
                } else { 
                    sortColumn = null; 
                    sortDirection = 'asc';
                }
            } else {
                sortColumn = newSortColumn;
                sortDirection = 'asc';
            }

            updateSortIcons();
            sortData();
            displayPage();
        });
    });

    /**
     * ========================================
     * LOGOUT FUNCTIONALITY
     * ========================================
     */
    
    function handleLogout() {
        showModal(
            "Confirm Logout", 
            "Are you sure you want to logout?", 
            [
                { text: 'Cancel', class: 'btn-clear' },
                { text: 'Logout', class: 'btn-delete', onClick: performLogout }
            ]
        );
    }

    function performLogout() {
        // Clear any local data
        clearFields();
        allDoctors = [];
        originalAllDoctors = [];
        
        // Redirect to logout endpoint
        window.location.href = '/Log_Out';
    }

    // Logout button event listener
    const logoutBtn = document.getElementById("logout-btn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", (e) => {
            e.preventDefault(); // Prevent default link behavior
            handleLogout();
        });
    }

    /**
     * ========================================
     * INITIALIZATION
     * ========================================
     */
    
    // Initialize the application by loading doctor data
    fetchAllDoctors();
    
});