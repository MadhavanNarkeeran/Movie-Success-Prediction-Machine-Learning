// Main JavaScript for Movie Success Predictor

// Global variables
let currentChart = null;

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Add smooth scrolling to all anchor links
    addSmoothScrolling();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Add loading states to buttons
    addLoadingStates();
    
    // Initialize form validation
    initializeFormValidation();
}

function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips if available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

function addLoadingStates() {
    const buttons = document.querySelectorAll('button[type="submit"]');
    
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.form && this.form.checkValidity()) {
                showLoadingState(this);
            }
        });
    });
}

function showLoadingState(button) {
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Loading...';
    button.disabled = true;
    
    // Reset button after 5 seconds (fallback)
    setTimeout(() => {
        button.innerHTML = originalText;
        button.disabled = false;
    }, 5000);
}

function hideLoadingState(button, originalText) {
    button.innerHTML = originalText;
    button.disabled = false;
}

function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });
}

// Utility Functions
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

function formatNumber(number, decimals = 2) {
    return Number(number).toFixed(decimals);
}

function formatPercentage(number, decimals = 1) {
    return (Number(number) * 100).toFixed(decimals) + '%';
}

function capitalizeFirst(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

// Chart Utilities
function createChart(ctx, config) {
    // Destroy existing chart if it exists
    if (currentChart) {
        currentChart.destroy();
    }
    
    // Create new chart
    currentChart = new Chart(ctx, config);
    return currentChart;
}

function getChartColors(count) {
    const colors = [
        'rgba(255, 99, 132, 0.8)',
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 205, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)',
        'rgba(153, 102, 255, 0.8)',
        'rgba(255, 159, 64, 0.8)',
        'rgba(199, 199, 199, 0.8)',
        'rgba(83, 102, 255, 0.8)'
    ];
    
    return colors.slice(0, count);
}

// API Utilities
async function makeAPIRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API Request Error:', error);
        throw error;
    }
}

// Form Utilities
function serializeForm(form) {
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    return data;
}

function resetForm(formId) {
    const form = document.getElementById(formId);
    if (form) {
        form.reset();
        form.classList.remove('was-validated');
    }
}

// Animation Utilities
function animateValue(element, start, end, duration = 1000) {
    const startTimestamp = performance.now();
    
    function step(timestamp) {
        const elapsed = timestamp - startTimestamp;
        const progress = Math.min(elapsed / duration, 1);
        
        const current = start + (end - start) * progress;
        element.textContent = Math.round(current);
        
        if (progress < 1) {
            requestAnimationFrame(step);
        }
    }
    
    requestAnimationFrame(step);
}

function fadeIn(element, duration = 300) {
    element.style.opacity = '0';
    element.style.display = 'block';
    
    const start = performance.now();
    
    function fade(timestamp) {
        const elapsed = timestamp - start;
        const progress = Math.min(elapsed / duration, 1);
        
        element.style.opacity = progress;
        
        if (progress < 1) {
            requestAnimationFrame(fade);
        }
    }
    
    requestAnimationFrame(fade);
}

function fadeOut(element, duration = 300) {
    const start = performance.now();
    const startOpacity = parseFloat(window.getComputedStyle(element).opacity);
    
    function fade(timestamp) {
        const elapsed = timestamp - start;
        const progress = Math.min(elapsed / duration, 1);
        
        element.style.opacity = startOpacity * (1 - progress);
        
        if (progress >= 1) {
            element.style.display = 'none';
        } else {
            requestAnimationFrame(fade);
        }
    }
    
    requestAnimationFrame(fade);
}

// Local Storage Utilities
function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
        console.error('Error saving to localStorage:', error);
    }
}

function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : defaultValue;
    } catch (error) {
        console.error('Error loading from localStorage:', error);
        return defaultValue;
    }
}

// Error Handling
function handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    
    let message = 'An unexpected error occurred. Please try again.';
    
    if (error.message) {
        message = error.message;
    }
    
    showNotification(message, 'danger');
}

// Export functions for use in other scripts
window.MoviePredictorUtils = {
    showNotification,
    formatNumber,
    formatPercentage,
    capitalizeFirst,
    createChart,
    getChartColors,
    makeAPIRequest,
    serializeForm,
    resetForm,
    animateValue,
    fadeIn,
    fadeOut,
    saveToLocalStorage,
    loadFromLocalStorage,
    handleError
};
