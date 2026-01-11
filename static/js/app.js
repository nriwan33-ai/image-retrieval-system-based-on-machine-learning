// Global state
let currentUploadedFile = null;
let isSearching = false;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const previewFilename = document.getElementById('previewFilename');
const searchBtn = document.getElementById('searchBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');

// Upload Area Events
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// File Selection Handler
function handleFileSelect(file) {
    if (!isValidImage(file)) {
        showError('Please select a valid image file (JPG, PNG, GIF, WebP)');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showError('File size must be less than 50MB');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewFilename.textContent = file.name;
        uploadArea.style.display = 'none';
        preview.style.display = 'block';
        resultsSection.style.display = 'none';
        clearError();
        currentUploadedFile = file;
        
        // Upload the image
        uploadImage(file);
    };
    reader.readAsDataURL(file);
}

// Validate Image File
function isValidImage(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    return validTypes.includes(file.type);
}

// Upload Image to Server
function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Image uploaded successfully');
            clearError();
        } else {
            showError('Error uploading image: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        showError('Error uploading image: ' + error.message);
    });
}

// Search Similar Images
searchBtn.addEventListener('click', () => {
    if (!currentUploadedFile || isSearching) {
        return;
    }

    isSearching = true;
    searchBtn.disabled = true;
    clearBtn.disabled = true;
    showLoading(true);
    clearError();

    const filename = currentUploadedFile.name;
    const query = extractQueryFromFilename(filename);

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: filename,
            query: query
        })
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        
        if (data.success && data.results && data.results.length > 0) {
            displayResults(data.results);
            resultsSection.style.display = 'block';
            clearError();
        } else if (data.error) {
            showError('Error: ' + data.error);
            resultsSection.style.display = 'none';
        } else {
            showError('No results found. Please try again.');
            resultsSection.style.display = 'none';
        }
    })
    .catch(error => {
        console.error('Search error:', error);
        showError('Error during search: ' + error.message);
        showLoading(false);
        resultsSection.style.display = 'none';
    })
    .finally(() => {
        isSearching = false;
        searchBtn.disabled = false;
        clearBtn.disabled = false;
    });
});

// Extract Query from Filename
function extractQueryFromFilename(filename) {
    // Remove file extension
    let name = filename.split('.')[0];
    // Replace hyphens, underscores with spaces
    name = name.replace(/[-_]/g, ' ');
    // Remove numbers and special characters, keep only letters and spaces
    name = name.replace(/[^a-zA-Z\s]/g, '');
    // Clean up multiple spaces
    name = name.trim().replace(/\s+/g, ' ');
    
    return name || 'similar images';
}

// Display Results
function displayResults(results) {
    resultsContainer.innerHTML = '';
    
    results.forEach((result, index) => {
        const resultItem = createResultItem(result, index + 1);
        resultsContainer.appendChild(resultItem);
    });
}

// Create Result Item
function createResultItem(result, rank) {
    const item = document.createElement('div');
    item.className = 'result-item';
    
    const similarity = result.similarity;
    const scoreColor = getSimilarityColor(similarity);
    
    item.innerHTML = `
        <img src="${escapeHtml(result.url)}" alt="Similar image ${rank}" class="result-image" 
             onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22200%22 height=%22200%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 fill=%22%23999%22 font-family=%22Arial%22%3EImage not found%3C/text%3E%3C/svg%3E'">
        <div class="result-info">
            <div class="result-score">
                <span class="result-rank">Result #${rank}</span>
                <span class="similarity-score">${similarity.toFixed(3)}</span>
            </div>
            <p class="result-url" title="${escapeHtml(result.url)}">${truncateUrl(result.url)}</p>
            <a href="${escapeHtml(result.url)}" target="_blank" class="result-link">View Image</a>
        </div>
    `;
    
    return item;
}

// Get Color Based on Similarity Score
function getSimilarityColor(score) {
    if (score >= 0.9) return '#10b981'; // Green
    if (score >= 0.7) return '#f59e0b'; // Orange
    return '#ef4444'; // Red
}

// Truncate URL for Display
function truncateUrl(url) {
    if (url.length > 50) {
        return url.substring(0, 47) + '...';
    }
    return url;
}

// HTML Escape
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Clear Button
clearBtn.addEventListener('click', () => {
    currentUploadedFile = null;
    fileInput.value = '';
    uploadArea.style.display = 'block';
    preview.style.display = 'none';
    resultsSection.style.display = 'none';
    resultsContainer.innerHTML = '';
    clearError();
    isSearching = false;
    searchBtn.disabled = false;
    clearBtn.disabled = false;
});

// Show Loading Spinner
function showLoading(show) {
    if (show) {
        loadingSpinner.style.display = 'flex';
    } else {
        loadingSpinner.style.display = 'none';
    }
}

// Show Error Message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

// Clear Error Message
function clearError() {
    errorMessage.textContent = '';
    errorMessage.style.display = 'none';
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Image Retrieval System loaded');
    
    // Test backend connection
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            console.log('Backend status:', data);
        })
        .catch(error => {
            console.error('Backend connection error:', error);
        });
});
