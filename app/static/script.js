// 獲取 DOM 元素
const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const resultsSection = document.getElementById('resultsSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorBox = document.getElementById('errorBox');
const errorText = document.getElementById('errorText');

let selectedFile = null;

// 拖放事件
dropZone.addEventListener('click', () => imageInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#764ba2';
    dropZone.style.background = '#f0f2ff';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#667eea';
    dropZone.style.background = '#f8f9ff';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
    dropZone.style.borderColor = '#667eea';
    dropZone.style.background = '#f8f9ff';
});

// document select
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    selectedFile = file;
    
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
    
    uploadBtn.disabled = false;
}

// Upload and predict
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    loadingSpinner.style.display = 'block';
    errorBox.style.display = 'none';
    resultsSection.style.display = 'none';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // show results
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        loadingSpinner.style.display = 'none';
    }
});

function displayResults(data) {
    // Prediction label
    document.getElementById('predLabel').textContent = data.label;
    
    // Probability chart
    const probsChart = document.getElementById('probsChart');
    probsChart.innerHTML = '';
    Object.entries(data.probs).forEach(([label, prob]) => {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        probItem.innerHTML = `
            <div class="prob-label">${label}</div>
            <div class="prob-bar">
                <div class="prob-fill" style="width: ${prob * 100}%"></div>
            </div>
            <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
        `;
        probsChart.appendChild(probItem);
    });
    
    // Medical report
    document.getElementById('reportText').textContent = data.report;
    
    // Data drift
    if (data.drift && data.drift.alert) {
        document.getElementById('driftBox').style.display = 'block';
        document.getElementById('driftAlert').textContent = `${data.drift.alert} (分數: ${data.drift.score?.toFixed(2) || 'N/A'})`;
    } else {
        document.getElementById('driftBox').style.display = 'none';
    }
    
    // show performance metrics
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    errorText.textContent = `錯誤: ${message}`;
    errorBox.style.display = 'block';
}