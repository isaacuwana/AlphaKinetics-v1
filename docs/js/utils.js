// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function generateId() {
    return Math.random().toString(36).substr(2, 9);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

function logToSession(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    if (window.appState) {
        window.appState.sessionHistory.push({
            timestamp,
            message,
            type
        });
    }
    
    // Update session history display if it exists
    const sessionHistory = document.getElementById('session-history');
    if (sessionHistory) {
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> <span class="log-${type}">${message}</span>`;
        sessionHistory.appendChild(logEntry);
        sessionHistory.scrollTop = sessionHistory.scrollHeight;
    }
}

function startSessionTimer() {
    setInterval(() => {
        if (!window.appState) return;
        const elapsed = Date.now() - window.appState.startTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        
        const timerElement = document.getElementById('session-timer');
        if (timerElement) {
            timerElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }, 1000);
}

function getExperimentalData() {
    if (window.appState && window.appState.processedData && window.appState.processedData.experimentalData) {
        return window.appState.processedData.experimentalData;
    }
    return null;
}

function updateFileList() {
    const fileList = document.getElementById('file-list');
    if (!fileList || !window.appState) return;

    if (window.appState.uploadedFiles.length === 0) {
        fileList.innerHTML = `
            <div style="text-align: center; color: var(--gray); padding: 3rem;">
                <i class="fas fa-inbox" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                <p>No files uploaded yet</p>
                <small>Upload experimental data to get started</small>
            </div>
        `;
        return;
    }

    fileList.innerHTML = window.appState.uploadedFiles.map(file => `
        <div class="file-item" style="padding: 1rem; margin-bottom: 1rem; background: white; border-radius: 8px; border: 1px solid var(--border);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>${file.name}</strong>
                    <div style="font-size: 0.9rem; color: var(--gray);">
                        ${formatFileSize(file.size)} • Quality: ${file.quality ? file.quality.score.toFixed(0) : 'N/A'}%
                    </div>
                </div>
                <div>
                    <span class="status-indicator ${file.processed ? 'status-success' : 'status-warning'}">
                        ${file.processed ? 'Processed' : 'Processing...'}
                    </span>
                </div>
            </div>
        </div>
    `).join('');
}

function showDataPreview(data) {
    const previewElement = document.getElementById('data-preview');
    if (!previewElement) return;

    previewElement.style.display = 'block';
    
    // Show basic statistics
    const statsElement = document.getElementById('data-stats');
    if (statsElement && data.concentrations) {
        const nTimePoints = data.concentrations.length;
        const nSpecies = data.concentrations[0]?.length || 0;
        const timeRange = data.time_points ? `${data.time_points[0].toFixed(2)} - ${data.time_points[data.time_points.length - 1].toFixed(2)}` : 'N/A';
        
        statsElement.innerHTML = `
            <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                <div class="stat-item">
                    <div class="stat-value">${nTimePoints}</div>
                    <div class="stat-label">Time Points</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${nSpecies}</div>
                    <div class="stat-label">Species</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${timeRange}</div>
                    <div class="stat-label">Time Range</div>
                </div>
            </div>
        `;
    }
}

function updateSimulationMetrics(results) {
    const compTimeElement = document.getElementById('comp-time');
    const intStepsElement = document.getElementById('int-steps');
    
    if (compTimeElement) compTimeElement.textContent = `${results.computationTime}ms`;
    if (intStepsElement) intStepsElement.textContent = results.integrationSteps;
}

function showSearchLoading(show) {
    const statusElement = document.getElementById('search-status');
    if (statusElement) {
        statusElement.style.display = show ? 'block' : 'none';
    }
}

function updateResultsDashboard() {
    const dashboardElement = document.getElementById('results-dashboard');
    if (!dashboardElement || !window.appState) return;

    dashboardElement.style.display = 'block';
    
    // Update analysis count
    const totalAnalysesElement = document.getElementById('total-analyses');
    if (totalAnalysesElement) {
        totalAnalysesElement.textContent = window.appState.analysisCount;
    }
    
    // Update computation time
    const totalTimeElement = document.getElementById('total-computation-time');
    if (totalTimeElement) {
        const elapsed = (Date.now() - window.appState.startTime) / 1000;
        totalTimeElement.textContent = `${elapsed.toFixed(1)}s`;
    }
}

function initializeComprehensiveChart() {
    const ctx = document.getElementById('comprehensive-chart');
    if (!ctx || typeof Chart === 'undefined') return;

    // Initialize with placeholder chart
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Start', 'Analysis', 'Complete'],
            datasets: [{
                label: 'Analysis Progress',
                data: [0, 50, 100],
                borderColor: window.CONFIG ? window.CONFIG.COLORS.PRIMARY : '#3b82f6',
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Comprehensive Analysis Dashboard'
                }
            }
        }
    });
}

function initializeDesignSpace() {
    // Initialize the design space UI
    const designSpaceContent = document.getElementById('design-space-content');
    if (designSpaceContent) {
        designSpaceContent.innerHTML = `
            <div class="design-empty-state">
                <i class="fas fa-flask-vial" style="font-size: 3rem; margin-bottom: 1rem; color: var(--primary); opacity: 0.7;"></i>
                <h4>Interactive Experimental Design</h4>
                <p style="color: var(--gray);">Analyze the design space, generate candidates, and run optimization.</p>
                <div class="design-quick-actions" style="margin-top: 1.5rem;">
                    <button class="btn btn-primary" onclick="previewDesignSpace()" style="margin-right: 0.5rem;">
                        <i class="fas fa-eye"></i> Preview Space
                    </button>
                    <button class="btn btn-success" onclick="generateInitialDesigns()">
                        <i class="fas fa-magic"></i> Generate Candidates
                    </button>
                </div>
            </div>
        `;
    }
}

function exportExperiment(rank) {
    if (typeof showAlert === 'function') {
        showAlert(`Exporting experiment ${rank}...`, 'info');
    }
    // Implementation for exporting experiment data
}

function exportResults(format) {
    if (typeof showAlert === 'function') {
        showAlert(`Exporting results in ${format} format...`, 'info');
    }
    // Implementation for exporting results
}

function shareResults() {
    if (typeof showAlert === 'function') {
        showAlert('Generating shareable link...', 'info');
    }
    // Implementation for sharing results
}

function generateDOI() {
    if (typeof showAlert === 'function') {
        showAlert('Generating DOI for results...', 'info');
    }
    // Implementation for DOI generation
}

function saveSession() {
    if (typeof showAlert === 'function') {
        showAlert('Saving session...', 'info');
    }
    // Implementation for saving session
}

function loadSession() {
    if (typeof showAlert === 'function') {
        showAlert('Loading session...', 'info');
    }
    // Implementation for loading session
}

function showHelp() {
    if (typeof showAlert === 'function') {
        showAlert('Opening help documentation...', 'info');
    }
    // Implementation for showing help
}

function showDataStatistics(data) {
    // Implementation for showing data statistics
    console.log('Data statistics:', data);
}