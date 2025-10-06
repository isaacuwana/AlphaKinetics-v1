# AlphaKinetics — Learn & explain reaction mechanism networks from sparse data

**One-line:** An open, reproducible pipeline to *propose, fit, certify, and disambiguate* candidate chemical reaction mechanisms from sparse, noisy time-series using constrained neural-symbolic search + robust parameter estimation + identifiability diagnostics + active experiment suggestion.

---

## Table of contents
- [Why AlphaKinetics?](#why-alphakinetics)
- [Features (what this repo contains)](#features-what-this-repo-contains)
- [Quick demo (run in <10 minutes)](#quick-demo-run-in-10-minutes)
- [Full usage & examples](#full-usage--examples)
- [Repository layout](#repository-layout)
- [Core design & algorithms](#core-design--algorithms)
- [Evaluation & benchmarks (what to report)](#evaluation--benchmarks-what-to-report)
- [Extending the project (research directions)](#extending-the-project-research-directions)
- [Reproducibility: Docker & CI](#reproducibility-docker--ci)
- [Limitations & risks](#limitations--risks)
- [License & contact](#license--contact)

---

## Why AlphaKinetics?

Inferring the **topology** (which reactions occur) and **kinetics** (rate laws and parameters) of chemical reaction networks from sparse, noisy measurements is a crucial inverse problem in chemical engineering and systems chemistry. Existing approaches often either assume the network is known (fit parameters only) or brute-force search topologies without practical identifiability guarantees.

**AlphaKinetics** sits between these extremes and provides a reproducible pipeline that:

- Produces interpretable, ranked mechanism hypotheses (graphs + parametric rate laws).
- Computes practical identifiability certificates (local Fisher Information diagnostics) that tell you *which parameters or combinations are determinable* from the available data.
- Suggests next experiments (active design) that are most likely to disambiguate top hypotheses.

This pipeline is suitable for method development, reproducible demos for applications and admissions, and early-stage collaboration with experimental labs.

---

## Features (what this repo contains)

- Synthetic data generator for toy reaction networks (measurement noise & partial observability).
- Grammar-based candidate graph enumerator (chemically plausible, stoichiometry constraints).
- Parameter fitting utilities:
  - Fully observed: nonlinear least-squares (scipy).
  - Partially observed: EM-style latent imputation loop (simple, replaceable).
- Finite-difference sensitivity and Fisher Information Matrix (FIM) tooling with eigen-analysis.
- Active experiment suggestion using FIM log-det as an information proxy.
- Runnable demo notebook that executes an end-to-end toy example.
- `paper_notes/` with a one-page PhD plan and starter lemmas.
- Minimal dependencies and quick local runtimes (designed for demos/teaching).

---

## Quick demo (run in <10 minutes)

**Prereqs:** Python 3.8+ (3.10+ recommended), `git`. Docker optional.

```bash
git clone https://github.com/<your-username>/alpha_kinetics.git alpha_kinetics
cd alpha_kinetics

# create & activate a venv (macOS/Linux)
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
# python -m venv venv
# .\venv\Scripts\Activate.ps1

# install
pip install -e .[dev,notebooks]

# generate a small synthetic demo dataset
alphakinetics-demo --output-dir demo_data

# open the demo notebook
jupyter notebook notebooks/01_synthetic_demo.ipynb
```

The notebook runs an end-to-end flow: generate data -> enumerate candidates -> fit parameters -> compute FIMs -> rank hypotheses -> suggest next experiments.

---

## Full usage & examples

Key files / modules:

- `notebooks/01_synthetic_demo.ipynb` — quick end-to-end demo.
- `src/alphakinetics/data/synthetic.py` — dataset generation CLI/options.
- `src/alphakinetics/search/graph_generator.py` — reaction grammar and enumerator.
- `src/alphakinetics/models/parameter_estimation.py` — parameter estimation wrappers.
- `src/alphakinetics/identifiability/fisher_info.py` — FD sensitivities and FIM functions.
- `src/alphakinetics/active/acquisition.py` — experiment generation & scoring.

Quick CLI examples:

```bash
# enumerate candidate networks (toy)
alphakinetics-search --n-species 4 --max-networks 50 --output candidates.json

# run parameter estimation for a candidate network
alphakinetics-fit candidates.json demo_data/experiment_0.pkl --output fit_results.pkl

# compute FIM and suggest experiments
alphakinetics-analyze identifiability fit_results.pkl demo_data/experiment_0.pkl
alphakinetics-design fit_results.pkl --output new_experiments.json
```

---

## Repository layout

```
AlphaKinetics/                  
├── alphakinetics/                
│   ├── __init__.py             
│   ├── workflows.py             
│   ├── models/
│   │   ├── __init__.py          
│   │   ├── simulator.py         
│   │   └── parameter_estimation.py  
│   ├── search/
│   │   ├── __init__.py          
│   │   └── graph_generator.py   
│   ├── identifiability/
│   │   ├── __init__.py          
│   │   └── fisher_info.py       
│   ├── active/
│   │   ├── __init__.py          
│   │   └── acquisition.py       
│   ├── data/
│   │   ├── __init__.py          
│   │   └── synthetic.py         
│   ├── utils/
│   │   ├── __init__.py          
│   │   ├── validation.py        
│   │   └── plotting.py          
│   └── cli/
│       ├── __init__.py          
│       ├── demo.py              
│       ├── fit.py               
│       ├── search.py            
│       ├── analyze.py           
│       ├── design.py            
│       └── validate.py          
├── tests/                       
│   ├── __init__.py              
│   ├── conftest.py              
│   └── test_*.py                
├── docs/                        
│   ├── conf.py                  
│   ├── index.rst               
│   └── _static/                 
├── examples/                   
├── notebooks/                   
├── .github/                     
│   └── workflows/
│       └── ci.yml             
├── pyproject.toml              
├── setup.py                    
├── requirements.txt            
├── requirements-dev.txt        
├── .pre-commit-config.yaml     
├── pytest.ini                 
├── tox.ini                     
├── .gitignore                  
├── .editorconfig               
├── MANIFEST.in                 
├── README.md                   
├── CONTRIBUTING.md             
├── CHANGELOG.md                
└── LICENSE                     
```

---

## Core design & algorithms

### Candidate generation
A compact grammar enumerates single-step reactions (stoichiometry <= bimolecular) and composes networks to a user-specified size. Enforces mass-balance and basic chemical plausibility. Replaceable by beam search, neural proposals, or domain priors.

### Parameter estimation
- **Fully observed:** nonlinear least-squares (`scipy.optimize.least_squares`) to estimate mass-action constants.
- **Partially observed:** EM-style loop (simulate hidden trajectories -> refit) — simple and extensible; recommend smoothing/MCMC for production.

### Identifiability diagnostics
- Compute finite-difference sensitivities -> Jacobian `J = dY/dθ`.
- Form FIM: `F = J.T @ Sigma^{-1} @ J`.
- Eigendecompose `F` to find near-zero eigenvalues → practically unidentifiable combinations. Report effective parameter count and condition numbers.

### Active experiment suggestion
- Generate candidate experiments (initial-condition scalings or perturbations).
- Approximate expected FIM per candidate and score by `logdet(F)` as an information proxy.
- Rank top candidates to guide next lab experiments.

---

## Evaluation & benchmarks (what to report)

When evaluating (synthetic or real), report:

- Topology recovery rate (top-1 / top-3).
- Parameter RMSE (when ground truth exists).
- Coverage: percent of parameters whose confidence intervals contain true values.
- Identifiability diagnostics: FIM spectrum, effective parameter count.
- Active learning efficiency: experiments required to reach a posterior mass threshold vs random.
- Compute cost: wall time per trial (search + fits + FIM).

Recommended plots: recovery curves, eigenvalue spectra (log scale), time-series fits per candidate, parameter CI bars.

---

## Extending the project (research directions / PhD seeds)

High-impact extensions:

- Learned proposal networks / neural priors to reduce enumeration cost.
- Symbolic regression (e.g., PySR) for non-mass-action rate forms.
- Autodiff Jacobians (JAX/PyTorch) for speed and accuracy.
- Bayesian parameter inference (MCMC/VI) for calibrated posteriors.
- Analytical structural identifiability checks to complement FIM diagnostics.
- Experiment design via direct Expected Information Gain (EIG).

A one-page PhD plan is included at `paper_notes/one_page_phd_plan.md`.

---

## Reproducibility: Docker & CI

Recommended additions (not included by default):

- `docker/Dockerfile` — pinned environment that runs demo & tests.
- GitHub Actions workflow to:
  - run `pytest`,
  - run/convert demo notebooks (`nbconvert`/`papermill`),
  - optionally build/publish a Docker image.

If desired, I can generate a ready-to-paste Dockerfile and GitHub Actions workflow.

---

## Limitations & risks

- **Combinatorial explosion:** naive enumeration doesn't scale—use priors or learned proposals.
- **Finite-difference sensitivities:** slower and less accurate for stiff systems—autodiff recommended.
- **EM imputation:** simplistic; can converge to local minima—consider smoothing or MCMC.
- **Toy vs real data:** real experiments require noise modeling, time alignment, calibration, and domain expertise.
- **Model hypotheses:** discovered rate forms are hypotheses and must be validated experimentally.

---

## License & contact

**License:** MIT — see `LICENSE`.

**Contact / demo video:**  
Name — Isaac Adeyeye
Email - isaacak88@gmail.com and iuadeyeye@gmail.com 
GitHub: https://github.com/isaacuwana/AlphaKinetics
