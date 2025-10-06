# AlphaKinetics: Comprehensive Documentation

**AlphaKinetics** is an open, reproducible Python pipeline designed to propose, fit, certify, and disambiguate candidate chemical reaction mechanisms from sparse, noisy time-series data. It integrates neural-symbolic search, robust parameter estimation, identifiability diagnostics, and active experimental design into a unified, high-level toolkit.

## 1. Core Philosophy

The central challenge in systems chemistry and chemical engineering is inferring both the **topology** (which reactions occur) and the **kinetics** (rate laws and parameters) of a reaction network from limited experimental data. AlphaKinetics addresses this by providing a framework that doesn't just fit parameters to a known network but actively discovers plausible network structures and provides clear, actionable feedback on the confidence of its findings.

The key principles are:
-   **Interpretability:** Produce ranked, chemically plausible mechanism hypotheses.
-   **Certification:** Compute practical identifiability diagnostics to determine which parameters are actually knowable from the data.
-   **Guidance:** Suggest the most informative next experiments to resolve ambiguities and improve model confidence.

## 2. Key Features

-   **Reaction Network Simulation:** A robust ODE-based engine (`models/simulator.py`) to simulate complex reaction dynamics.
-   **Synthetic Data Generation:** A sophisticated generator (synthetic.py) for creating benchmark datasets with realistic noise models, partial observability, and various network motifs.
-   **Network Topology Search:** A grammar-based enumerator (`search/graph_generator.py`) to generate chemically plausible candidate networks.
-   **Robust Parameter Estimation:** Advanced fitting routines (`models/parameter_estimation.py`), including multi-start and Bayesian methods, with comprehensive uncertainty quantification.
-   **Identifiability Analysis:** Fisher Information Matrix (FIM) based analysis (`identifiability/fisher_info.py`) to diagnose practical and structural identifiability issues.
-   **Active Experimental Design:** An information-theoretic engine (acquisition.py) that suggests optimal experiments (D-optimal, A-optimal, etc.) to maximize information gain.
-   **High-Level Workflows:** End-to-end pipelines (workflow.py) that automate the entire discovery process from data to insight.
-   **Command-Line Interface (CLI):** A suite of tools for running all major components from the terminal.
-   **Interactive Web Dashboard:** A local web application (index.html) for interactive data upload, simulation, and analysis.

## 3. Installation

### User Installation

For standard use of the package:
```bash
pip install alphakinetics
```

### Developer Setup

For contributing to the project:
```bash
# 1. Fork and clone the repository
git clone https://github.com/isaacuwana/AlphaKinetics.git
cd AlphaKinetics

# 2. Create and activate a virtual environment
python -m venv alpkntenv
# On Windows:
# alpkntenv\Scripts\activate
# On macOS/Linux:
# source alpkntenv/bin/activate

# 3. Install in editable mode with all development dependencies
pip install -e ".[dev,notebooks,parallel]"

# 4. Set up pre-commit hooks for code quality
pre-commit install
```

## 4. Quick Start

### End-to-End Discovery via CLI

The CLI provides a quick way to run the core workflow.

```bash
# 1. Generate a demo dataset
alphakinetics-demo --n-species 3 --output-dir demo_data

# 2. Search for candidate networks based on the data
alphakinetics-search --n-species 3 --output-dir search_results

# 3. Fit parameters for the top candidates
alphakinetics-fit search_results/candidates.json demo_data/synthetic_dataset.pkl --output-dir fit_results

# 4. Analyze the best model for identifiability
alphakinetics-analyze identifiability fit_results/best_model.json --output-dir analysis_results

# 5. Suggest the next best experiment to run
alphakinetics-design fit_results/best_model.json --output-dir design_suggestions
```

### Interactive Web Dashboard

For a visual, interactive experience:

1.  Ensure you have completed the developer setup, including `pip install -e ".[dev,notebooks,parallel]"` and `pip install Flask Flask-Cors`.
2.  Start the backend server by running the following command in your terminal from the project's root directory:
    ```bash
    python server.py
    ```
    You should see output indicating the server is running on `http://127.0.0.1:5000`.
3.  Open the `docs/index.html` file in your web browser. The dashboard is now live and connected to the backend.

## 5. Module Breakdown

### models - The Core Engine

This module contains the fundamental classes for defining, simulating, and fitting reaction networks.

-   **`simulator.py`**:
    -   `Reaction`: Defines a single reaction with reactants, products, and a rate type.
    -   `ReactionNetwork`: The main container for a set of species and reactions. It constructs the stoichiometric matrix and rate equations.
    -   `ReactionSimulator`: The ODE solver class that simulates the time-evolution of the network using `scipy.integrate.solve_ivp`.
-   **`parameter_estimation.py`**:
    -   `RobustParameterEstimator`: Implements parameter fitting using multi-start, non-linear least-squares optimization to avoid local minima.
    -   `BayesianParameterEstimator`: (Future) Provides Bayesian inference capabilities using MCMC methods.
    -   `ParameterEstimationResult`: A data class to hold the results of a fitting run, including optimal parameters, confidence intervals, R², and other diagnostics.

### data - Data Handling

-   **synthetic.py**: A powerful and flexible synthetic data generator.
    -   `SyntheticExperimentConfig`: A dataclass to configure every aspect of data generation, from network motif and noise model to observability scenarios.
    -   `BiologicalNetworkLibrary`: Provides templates for common network structures like chains, feedback loops, and oscillators.
    -   `NoiseGenerator`: Applies various realistic noise models (Gaussian, Poisson, mixed, etc.).
    -   `create_synthetic_experiment()`: A high-level function to quickly generate a dataset with sensible defaults.

### search - Discovering Topologies

-   **`graph_generator.py`**: Responsible for proposing candidate network topologies.
    -   `NetworkSearchStrategy`: An enum for different search methods (e.g., `GRAMMAR_GUIDED`).
    -   `NetworkGenerator`: The main class that enumerates reaction graphs based on constraints like the number of species and maximum reactions. It enforces chemical plausibility rules.

### identifiability - Certifying Results

-   **`fisher_info.py`**: The core of the model certification process.
    -   `FisherInformationAnalyzer`: Computes the Fisher Information Matrix (FIM) for a given model and dataset. The FIM quantifies the maximum information the data provides about the model parameters.
    -   **Algorithm**: It uses finite-difference methods to calculate the sensitivity matrix (Jacobian `J = dY/dθ`) and then computes the FIM as `F = J.T @ Σ⁻¹ @ J`.
    -   `IdentifiabilityResult`: Holds the output, including the FIM's eigenvalues, condition number, and a per-parameter identifiability score. Small eigenvalues indicate non-identifiable parameter combinations.

### active - Guiding Experiments

-   **acquisition.py**: Implements the logic for optimal experimental design.
    -   `AcquisitionFunction`: An enum for different design criteria (`D_OPTIMAL`, `A_OPTIMAL`, `E_OPTIMAL`). These criteria correspond to different ways of optimizing the FIM of a *hypothetical* future experiment.
        -   **D-Optimal**: Maximizes `det(FIM)`, which minimizes the volume of the parameter confidence ellipsoid. Excellent for improving overall parameter certainty.
        -   **A-Optimal**: Minimizes `trace(FIM⁻¹)`, which minimizes the average parameter variance.
        -   **E-Optimal**: Maximizes the minimum eigenvalue of the FIM, which is useful for improving the identifiability of the worst-constrained parameter combination.
    -   `ActiveExperimentalDesigner`: The main class that takes a model and suggests the next best experiment to run by generating candidate experiments and scoring them with the chosen acquisition function.

### workflows - High-Level Pipelines

-   **workflow.py**: This module ties everything together into automated, end-to-end pipelines.
    -   `full_mechanism_discovery_pipeline()`: The main entry point. It takes experimental data and orchestrates the entire process:
        1.  Generates candidate networks (`search`).
        2.  Fits parameters for each candidate (`models`).
        3.  Ranks candidates by a score combining fit quality and plausibility.
        4.  Performs identifiability analysis on the top models (`identifiability`).
        5.  Suggests the next best experiments to disambiguate top models (`active`).
    -   `MechanismDiscoveryResult`: A comprehensive dataclass that organizes all outputs from the pipeline.

### cli - Command-Line Tools

This package exposes the core functionalities as terminal commands, defined in pyproject.toml under `[project.scripts]`. Each file (demo.py, `fit.py`, etc.) uses a library like `argparse` or `click` to define its command-line arguments.

-   `alphakinetics-demo`: Generate a synthetic dataset.
-   `alphakinetics-search`: Run network topology search.
-   `alphakinetics-fit`: Run parameter estimation.
-   `alphakinetics-analyze`: Run identifiability analysis.
-   `alphakinetics-design`: Run experimental design.
-   `alphakinetics-validate`: Validate input data or models.

## 6. Testing Strategy

The project uses `pytest` for robust testing, configured in pytest.ini and tox.ini.

-   **conftest.py**: Sets up shared test fixtures, such as pre-built networks (`simple_network`, `enzyme_kinetics_network`) and synthetic datasets (`synthetic_data`, `noisy_data`). This ensures tests are consistent and reproducible.
-   **Markers**: Tests are categorized with markers (`@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`) to allow for running specific test suites.
-   **Continuous Integration**: The GitHub Actions workflow in ci.yml automatically runs linters (`flake8`), type checkers (`mypy`), and the full `pytest` suite across multiple Python versions (3.8-3.11) on every push and pull request. This guarantees code quality and prevents regressions.

## 7. Configuration

The package is configurable at runtime via the `alphakinetics.configure()` function. This allows users to change global settings for solvers, plotting, and optimization without modifying the code.

```python
import alphakinetics as ak

# Set a global random seed for reproducibility and use a different plot style
ak.configure(random_seed=42, plot_style='ggplot')

# Get the current configuration
current_config = ak.get_config()
print(current_config['default_solver'])
```
