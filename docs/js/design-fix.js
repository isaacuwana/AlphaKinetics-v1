// Fix for experimental design display issues

// Override the displayDesignResults function
function displayDesignResults(results) {
    document.getElementById('design-results').style.display = 'block';
    document.getElementById('design-progress').style.display = 'none';
    
    // Handle different property name formats (server vs local fallback)
    const totalInfo = results.total_information_gain || results.totalInformation || 0;
    const efficiency = results.design_efficiency || results.designEfficiency || 0;
    const precision = results.expected_precision_gain || results.expectedPrecision || 0;
    const cost = results.resource_cost || results.resourceCost || 1;
    
    document.getElementById('total-information').textContent = Number(totalInfo).toFixed(3);
    document.getElementById('design-efficiency').textContent = `${Number(efficiency).toFixed(1)}%`;
    document.getElementById('expected-precision').textContent = `${Number(precision).toFixed(1)}%`;
    document.getElementById('resource-cost').textContent = `${Number(cost).toFixed(2)}x`;

    const tbody = document.querySelector('#design-table tbody');
    if (tbody) {
        // Handle both server response (recommended_experiments) and local fallback (experiments)
        const experiments = results.recommended_experiments || results.experiments || [];
        if (experiments && experiments.length > 0) {
            tbody.innerHTML = experiments.map(exp => {
                let priorityClass = 'status-error';
                if (exp.priority === 'High') priorityClass = 'status-success';
                if (exp.priority === 'Medium') priorityClass = 'status-warning';
                
                // Handle different initial conditions formats safely
                let initialConditionsStr = 'N/A';
                if (exp.initial_conditions && Array.isArray(exp.initial_conditions)) {
                    initialConditionsStr = exp.initial_conditions.map(c => Number(c).toFixed(2)).join(', ');
                } else if (exp.initialConditions) {
                    // Handle string format from local fallback
                    initialConditionsStr = exp.initialConditions;
                }
                
                // Handle different information gain property names
                const infoGain = exp.information_gain || exp.informationGain || exp.expected_information_gain || 0;
                const infoGainStr = Number(infoGain).toFixed(3);
                
                return `<tr>
                    <td><strong>${exp.rank}</strong></td>
                    <td>${exp.type}</td>
                    <td>${initialConditionsStr}</td>
                    <td>${exp.duration}</td>
                    <td>${infoGainStr}</td>
                    <td>${exp.feasibility}</td>
                    <td><span class="status-indicator ${priorityClass}">${exp.priority}</span></td>
                    <td><button class="btn btn-success" style="padding:0.25rem 0.5rem;font-size:0.8rem;" onclick="exportExperiment(${exp.rank})">Export</button></td>
                </tr>`;
            }).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">No recommended experiments found. The optimization may not have yielded a better design.</td></tr>';
        }
    }
    createDesignOutcomesChart(results);
}

// Also fix the exportExperiment function to handle both data structures
function exportExperiment(rank) {
    if (!appState.designResults) {
        showAlert('No design results available to export.', 'error');
        return;
    }
    
    // Handle both property names
    const experiments = appState.designResults.recommended_experiments || appState.designResults.experiments || [];
    const experiment = experiments.find(exp => exp.rank === rank);

    if (!experiment) {
        showAlert(`Experiment with rank ${rank} not found.`, 'error');
        return;
    }

    // Handle different property formats
    const initialConditions = experiment.initial_conditions || experiment.initialConditions || 'N/A';
    const infoGain = experiment.information_gain || experiment.informationGain || experiment.expected_information_gain || 0;
    
    const exportData = {
        rank: experiment.rank,
        type: experiment.type,
        initial_conditions: Array.isArray(initialConditions) ? initialConditions : initialConditions,
        duration: experiment.duration,
        information_gain: infoGain,
        feasibility: experiment.feasibility,
        priority: experiment.priority
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `experiment_${rank}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showAlert(`Experiment ${rank} exported successfully!`, 'success');
}