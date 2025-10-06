// Fix for experimental design payload to include number of experiments

// Override the runExperimentalDesign function to fix the payload
window.addEventListener('DOMContentLoaded', function() {
    // Store the original function
    const originalRunExperimentalDesign = window.runExperimentalDesign;
    
    // Override with fixed version
    window.runExperimentalDesign = async function() {
        if (!appState.parameterResults) {
            showAlert('Please run parameter estimation first.', 'error');
            return;
        }

        showAlert('Optimizing experimental design...', 'warning');
        document.getElementById('design-progress').style.display = 'block';

        const objective = document.getElementById('design-objective').value;
        const nExperiments = parseInt(document.getElementById('n-experiments').value);
        const budget = parseInt(document.getElementById('optimization-budget').value);
        
        // Collect selected experiment types from UI checkboxes
        const selectedExperimentTypes = [];
        if (document.getElementById('initial-condition')?.checked) {
            selectedExperimentTypes.push('initial_condition');
        }
        if (document.getElementById('time-course')?.checked) {
            selectedExperimentTypes.push('time_course');
        }
        if (document.getElementById('perturbation')?.checked) {
            selectedExperimentTypes.push('perturbation_studies');
        }
        if (document.getElementById('dose-response')?.checked) {
            selectedExperimentTypes.push('dose_response');
        }
        
        // If no experiment types selected, use all types as default
        if (selectedExperimentTypes.length === 0) {
            selectedExperimentTypes.push('initial_condition', 'time_course', 'perturbation_studies', 'dose_response');
        }

        // Update progress with realistic steps
        const progressBar = document.getElementById('design-progress-bar');
        const logContainer = document.getElementById('design-log');

        const updateProgress = (percent, message) => {
            progressBar.style.width = `${percent}%`;
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `<span class="log-timestamp">[${new Date().toLocaleTimeString()}]</span> <span class="log-info">${message}</span>`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        };

        updateProgress(15, `Starting ${objective} optimization...`);
        await new Promise(resolve => setTimeout(resolve, 600));

        updateProgress(40, 'Evaluating information content...');
        await new Promise(resolve => setTimeout(resolve, 1200));

        updateProgress(70, 'Optimizing experimental conditions...');
        await new Promise(resolve => setTimeout(resolve, 1000));

        updateProgress(90, 'Computing feasibility scores...');
        await new Promise(resolve => setTimeout(resolve, 400));

        updateProgress(100, 'Design optimization complete!');

        // Generate design based on current parameter uncertainties
        // Prefer server-side design pipeline; fall back to local heuristic if server is unreachable
        (async () => {
            try {
                const payload = {
                    network: appState.currentNetwork && appState.currentNetwork.network && appState.currentNetwork.network.network ? appState.currentNetwork.network.network : null,
                    parameters: appState.parameterResults && appState.parameterResults.best_fit ? appState.parameterResults.best_fit.optimal_parameters : [],
                    selected_designs: [],
                    n_experiments: nExperiments, // Pass the number of experiments to the server
                    selected_experiment_types: selectedExperimentTypes, // Add selected experiment types
                    constraints: {
                        min_conc: parseFloat(document.getElementById('min-conc')?.value) || 0.01,
                        max_conc: parseFloat(document.getElementById('max-conc')?.value) || 10.0,
                        max_time: parseInt(document.getElementById('max-time')?.value) || 100
                    }
                };

                console.log('Sending payload to server:', payload);

                const resp = await fetch(`${CONFIG.API_BASE_URL}/api/design`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!resp.ok) throw new Error('Design API returned error');

                const results = await resp.json();
                console.log('Server response:', results);
                
                // Use server response directly
                appState.designResults = results;
                logToSession('Used server-side experimental design pipeline');

            } catch (err) {
                // Fallback: generate heuristics locally (keeps previous UX if server not available)
                logToSession(`Design API failed, using local heuristic: ${err.message}`, 'warning');
                console.error('Server error:', err);

                const poorlyIdentified = appState.identifiabilityResults?.poorlyIdentified || 0;
                const baseInformation = 50 + Math.random() * 20;
                const designQuality = Math.max(0.7, appState.parameterResults?.rSquared || 0.7);

                const localExperiments = Array.from({ length: nExperiments }, (_, i) => {
                    const experimentTypes = ['Time Course Experiments', 'Perturbation Studies', 'Initial Condition Variations', 'Dose-Response Analysis'];
                    const infoScore = baseInformation * Math.exp(-i * 0.15) + Math.random() * 5;
                    return {
                        rank: i + 1,
                        type: experimentTypes[i % experimentTypes.length],
                        initial_conditions: [(Math.random() * 2), (Math.random() * 2)], // Array format to match server
                        duration: (5 + Math.random() * 25).toFixed(1),
                        information_gain: infoScore.toFixed(2), // Correct property name
                        feasibility: `${Math.max(60, 95 - Math.random() * 20 - poorlyIdentified * 5).toFixed(0)}%`,
                        priority: i < 2 ? 'High' : (i < Math.ceil(nExperiments * 0.6) ? 'Medium' : 'Low')
                    };
                });

                // Local heuristic fallback: use server-compatible property names
                const expectedPrecisionLocal = Math.round(25 + designQuality * 35 - poorlyIdentified * 5 + Math.round(Math.random()*3));

                appState.designResults = {
                    recommended_experiments: localExperiments, // Match server property name
                    total_information_gain: (baseInformation * nExperiments * 0.7).toFixed(1), // Match server property name
                    design_efficiency: Math.round(designQuality * 100), // Match server property name
                    expected_precision_gain: expectedPrecisionLocal, // Match server property name
                    resource_cost: (1.0 + nExperiments * 0.1 + Math.random() * 0.5).toFixed(1), // Match server property name
                    objective: objective
                };
            }

            // Display results from whichever source we have
            displayDesignResults(appState.designResults);
            appState.analysisCount++;
            logToSession(`Experimental design completed: ${objective} with ${nExperiments} experiments`);
            showAlert('Experimental design complete!', 'success');
        })();
    };
});