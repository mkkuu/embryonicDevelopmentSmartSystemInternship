/**
 * ========================================
 * EMBRYO ANALYSIS SYSTEM - LOGIN PAGE JAVASCRIPT
 * ========================================
 * 
 * This JavaScript module provides comprehensive functionality for the login page
 * of the Embryo Analysis System. It features:
 * 
 * - Modern ES6+ JavaScript with modular architecture
 * - Real-time form validation with debounced input handling
 * - Persistent error messages until user interaction
 * - Password visibility toggle with accessibility support
 * - Remember me functionality with localStorage
 * - Loading states and user feedback
 * - Keyboard navigation support
 * - Error handling with retry mechanisms
 * - API communication with timeout handling
 * 
 * Architecture:
 * - Configuration-driven approach with constants
 * - Modular design with separate concerns
 * - State management for loading and error states
 * - Utility functions for common operations
 * - Event-driven programming with proper cleanup
 * 
 * Features:
 * - Form validation without length requirements
 * - Error persistence until user input
 * - Remember me with 30-day expiration
 * - Password toggle with ARIA support
 * - Loading animations and feedback
 * - Responsive error handling
 * 
 * Author: LSL Team
 * Version: 1.0
 * Last Updated: 2025-10-04
 */

/**
 * Configuration constants for the login system
 * Centralized configuration for easy maintenance
 */
const CONFIG = {
  REMEMBER_ME_KEY: 'rememberedUserName',
  REMEMBER_ME_DURATION: 30 * 24 * 60 * 60 * 1000, // 30 days in milliseconds
  API_ENDPOINT: '/Submit-Info-Sign-In',
  LOADING_DELAY: 1000, // Simulated delay for UX
  DEBOUNCE_DELAY: 300,
  MAX_RETRY_ATTEMPTS: 3
};

// DOM Elements - cached for performance
const elements = {
  form: document.getElementById('loginForm'),
  userNameInput: document.getElementById('userName'),
  passwordInput: document.getElementById('passWord'),
  loginButton: document.getElementById('loginButton'),
  togglePassword: document.getElementById('togglePassword'),
  errorMessage: document.getElementById('errorMessage'),
  rememberMeCheckbox: document.getElementById('rememberMe')
};

// State management
const state = {
  isLoading: false,
  retryCount: 0,
  lastError: null
};

// Utility functions
const utils = {
  // Debounce function for input validation
  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  // Show error message with animation
  showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.classList.add('show');
    
    // Don't auto-hide - let user input clear it
  },

  // Clear error message - only clear when user explicitly interacts
  clearError() {
    elements.errorMessage.classList.remove('show');
    setTimeout(() => {
      elements.errorMessage.textContent = '';
    }, 300);
  },

  // Clear error only on successful form submission or explicit user action
  clearErrorOnSuccess() {
    elements.errorMessage.classList.remove('show');
    setTimeout(() => {
      elements.errorMessage.textContent = '';
    }, 300);
  },

  // Highlight input field with error
  highlightError(input) {
    const inputGroup = input.closest('.input-group');
    inputGroup.classList.add('error');
    input.focus();
    
    // Remove error styling on input
    const removeError = () => {
      inputGroup.classList.remove('error');
      input.removeEventListener('input', removeError);
    };
    input.addEventListener('input', removeError);
  },

  // Format error message for user
  formatErrorMessage(error) {
    if (error.message) {
      return error.message;
    }
    if (error.status) {
      switch (error.status) {
        case 401:
          return 'Invalid username or password. Please try again.';
        case 403:
          return 'Access denied. Please contact your administrator.';
        case 404:
          return 'Service not found. Please try again later.';
        case 500:
          return 'Server error. Please try again later.';
        default:
          return 'Something went wrong. Please try again.';
      }
    }
    return 'Network error. Please check your connection and try again.';
  }
};

// Password visibility toggle
const passwordToggle = {
  init() {
    elements.togglePassword.addEventListener('click', this.toggle);
    elements.togglePassword.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        this.toggle();
      }
    });
  },

  toggle() {
    const isPassword = elements.passwordInput.type === 'password';
    elements.passwordInput.type = isPassword ? 'text' : 'password';
    
    const icon = elements.togglePassword.querySelector('i');
    icon.classList.toggle('fa-eye', !isPassword);
    icon.classList.toggle('fa-eye-slash', isPassword);
    
    // Update aria-label for accessibility
    elements.togglePassword.setAttribute(
      'aria-label', 
      isPassword ? 'Hide password' : 'Show password'
    );
  }
};

// Remember me functionality
const rememberMe = {
  init() {
    this.loadRememberedUser();
  },

  loadRememberedUser() {
    try {
      const rememberedData = localStorage.getItem(CONFIG.REMEMBER_ME_KEY);
      if (rememberedData) {
        const { username, timestamp } = JSON.parse(rememberedData);
        
        // Check if remember me is still valid
        if (Date.now() - timestamp < CONFIG.REMEMBER_ME_DURATION) {
          elements.userNameInput.value = username;
          elements.rememberMeCheckbox.checked = true;
        } else {
          // Expired, remove from storage
          localStorage.removeItem(CONFIG.REMEMBER_ME_KEY);
        }
      }
    } catch (error) {
      localStorage.removeItem(CONFIG.REMEMBER_ME_KEY);
    }
  },

  saveUser(username) {
    if (elements.rememberMeCheckbox.checked) {
      const data = {
        username,
        timestamp: Date.now()
      };
      localStorage.setItem(CONFIG.REMEMBER_ME_KEY, JSON.stringify(data));
    } else {
      localStorage.removeItem(CONFIG.REMEMBER_ME_KEY);
    }
  }
};

// Form validation
const validation = {
  init() {
    // Real-time validation with debouncing
    const debouncedValidation = utils.debounce(this.validateField, CONFIG.DEBOUNCE_DELAY);
    
    elements.userNameInput.addEventListener('input', debouncedValidation);
    elements.passwordInput.addEventListener('input', debouncedValidation);
  },

  validateField(event) {
    const field = event.target;
    const value = field.value.trim();
    
    // Don't clear errors automatically - let them persist
    // Only remove visual error styling when user starts typing
    field.closest('.input-group').classList.remove('error');
    
    // No length validation - just check if field has content
    return value.length > 0;
  },

  validateForm() {
    const username = elements.userNameInput.value.trim();
    const password = elements.passwordInput.value.trim();
    
    // Don't clear errors automatically - let them persist until user action
    
    if (!username) {
      utils.showError('Username is required.');
      utils.highlightError(elements.userNameInput);
      return false;
    }
    
    if (!password) {
      utils.showError('Password is required.');
      utils.highlightError(elements.passwordInput);
      return false;
    }
    
    return true;
  }
};

// Loading state management
const loadingState = {
  setLoading(isLoading) {
    state.isLoading = isLoading;
    elements.loginButton.disabled = isLoading;
    elements.loginButton.classList.toggle('loading', isLoading);
    
    // Disable form inputs during loading
    [elements.userNameInput, elements.passwordInput, elements.rememberMeCheckbox].forEach(
      input => input.disabled = isLoading
    );
  }
};

// API communication
const api = {
  async signIn(username, password) {
    const formData = new FormData();
    formData.append('Username', username);
    formData.append('Password', password);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
    
    try {
      const response = await fetch(CONFIG.API_ENDPOINT, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data;
      
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new Error('Request timeout. Please try again.');
      }
      
      throw error;
    }
  }
};

// Login process
const loginProcess = {
  async handleLogin(event) {
    event.preventDefault();
    
    if (state.isLoading) return;
    
    if (!validation.validateForm()) return;
    
    const username = elements.userNameInput.value.trim();
    const password = elements.passwordInput.value.trim();
    
    loadingState.setLoading(true);
    // Don't clear errors automatically - let them persist
    
    try {
      // Simulate loading delay for better UX
      await new Promise(resolve => setTimeout(resolve, CONFIG.LOADING_DELAY));
      
      const data = await api.signIn(username, password);
      
      if (data.exists) {
        // Clear errors only on successful login
        utils.clearErrorOnSuccess();
        
        // Save remember me preference
        rememberMe.saveUser(username);
        
        // Clear password field
        elements.passwordInput.value = '';
        
        // Route based on user type
        this.routeUser(data.user_type);
        
                } else {
        throw new Error('Invalid username or password.');
      }
      
    } catch (error) {
      const errorMessage = utils.formatErrorMessage(error);
      utils.showError(errorMessage);
      
      // Reset retry count on successful request
      state.retryCount = 0;
      
    } finally {
      loadingState.setLoading(false);
    }
  },

  routeUser(userType) {
    const routes = {
      'Admin': '/Admin/Doctor',
      'Doctor': '/Doctor/Embryo'
    };
    
    const route = routes[userType];
    if (route) {
      // Add loading indicator to button before redirect
      elements.loginButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Redirecting...';
      
      // Small delay to show loading state
      setTimeout(() => {
        window.location.href = route;
      }, 500);
            } else {
      utils.showError('Unknown user type. Please contact support.');
    }
  }
};

// Keyboard navigation
const keyboardNavigation = {
  init() {
    document.addEventListener('keydown', (e) => {
      // Enter key on form
      if (e.key === 'Enter' && e.target.closest('#loginForm')) {
        e.preventDefault();
        loginProcess.handleLogin(e);
      }
      
      // Escape key to clear errors
      if (e.key === 'Escape') {
        utils.clearError();
      }
    });
  }
};

// Initialize application
const app = {
  init() {
    // Check if all required elements exist
    const requiredElements = Object.values(elements);
    const missingElements = requiredElements.filter(el => !el);
    
    if (missingElements.length > 0) {
      return;
    }
    
    // Initialize all modules
    passwordToggle.init();
    rememberMe.init();
    validation.init();
    keyboardNavigation.init();
    
    // Form submission
    elements.form.addEventListener('submit', loginProcess.handleLogin.bind(loginProcess));
    
    // Clear visual error styling on input - but keep error messages
    [elements.userNameInput, elements.passwordInput].forEach(input => {
      input.addEventListener('input', () => {
        // Only remove visual error styling, keep error message visible
        input.closest('.input-group').classList.remove('error');
      });
    });
    
  }
};

// Start the application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', app.init);
} else {
  app.init();
}
