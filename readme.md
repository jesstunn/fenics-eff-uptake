# Sulcus Transport Modeling: Advection-Diffusion Simulation Framework

A comprehensive FEniCS-based simulation framework for studying mass transport in sulcus geometries with Robin boundary conditions, comparing sulcus domains against rectangular surrogates.

## Overview

This codebase implements finite element simulations to study sulci under various transport regimes. The framework compares transport behavior between realistic sulcus geometries and simplified rectangular domains, with hope of finding an effective boundary condition for a lower boundary robin condition applied to the sulcus domain.

## Key Features

- **Multiple Transport Modes**: Pure diffusion, advection-diffusion, and no-uptake scenarios
- **Dual Domain Types**: Sulcus (realistic curved geometry) vs rectangular (simplified surrogate)
- **Variable Uptake**: Support for spatially-varying Robin boundary conditions μ(x)
- **Comprehensive Analysis**: Flux calculations, μ_eff comparisons, and geometric parameter sweeps
- **Automated Workflows**: Geometry variations, aspect ratio studies, and validation analyses

## Transport Modes

### 1. Advection-Diffusion (`adv-diff`)

- Full Navier-Stokes flow with Stokes solver
- Advection-diffusion equation for concentration
- Peclet number analysis

### 2. Pure Diffusion (`no-adv`)

- Zero velocity field (u = 0)
- Diffusion-only transport
- Focus on uptake coefficient μ effects

### 3. No Uptake (`no-uptake`)

- Flow field present but no boundary uptake (μ = 0)
- Pure transport without sink terms

## Key Modules

### Core Simulation

- `simulation.py` - Main simulation orchestrator
- `solvers.py` - FEniCS PDE solvers (Stokes, advection-diffusion, pure diffusion)
- `mesh.py` - Gmsh-based mesh generation for sulcus and rectangular domains
- `parameters.py` - Parameter management and geometry configurations

### Analysis

- `analysis.py` - Flux computations, μ_eff calculations, profile extraction
- `plotting.py` - Comprehensive visualization suite with LaTeX formatting

### Studies

- `no_advection_analysis_A.py` - μ parameter sweeps and aspect ratio analysis
- `no_advection_analysis_B.py` - Application of mu_eff to a rectangular surrogate
- `no_uptake_analysis.py` - Peclet number studies with velocity analysis
- `adv_diff_analysis.py` - Advection-diffusion validation with step mu_eff(x) functions
- `mesh_analysis.py` - Mesh convergence studies

## Key Concepts

### Effective Uptake Coefficient (μ_eff)

The framework computes μ_eff through multiple methods:

- **Simulation**: Direct flux/concentration ratios
- **Analytical**: Arc length correction for curved boundaries
- **Enhanced**: Modified analytical model with penetration effects

### Step Functions

Spatially-varying uptake μ(x) using smoothed step functions:

```python
class StepUptakeOpen(UserExpression):
    # Smooth transitions at sulcus mouth boundaries
    # μ_base outside mouth, μ_target inside mouth
```

### Validation Framework

- Rectangular surrogates with equivalent μ_eff values
- Concentration and flux ratio comparisons
- Multi-parameter validation matrices

## Directory Structure

```

Each INDIVIDUAL study creates:
├── [Study] Analysis/             # CSV results and plots
├── [Study] Simulations/          # Individual simulation outputs
│   ├── Mesh Files/               # .geo, .msh, .xml files
│   ├── ParaView Files/           # .pvd visualization
│   ├── Analysis Plots/           # Result plots
│   └── Results Data/             # JSON simulation data
```

## Usage Examples

### Single Simulation

```python
from parameters import Parameters
from simulation import run_simulation

params = Parameters(mode='no-adv')
params.sulci_w_dim = 0.5  # mm
params.sulci_h_dim = 1.0  # mm
params.validate()
params.nondim()

results = run_simulation(
    mode='no-adv',
    study_type='Single Domain',
    config_name='test_sulcus',
    domain_type='sulcus',
    params=params
)
```

### Parameter Sweeps

```python
# Run μ parameter sweep
python no_advection_analysis_A.py  # Select option 1

# Run geometry comparison
python no_uptake_analysis.py       # Select option 1
```

## Dependencies

- **FEniCS** (dolfin) - Finite element framework
- **Gmsh** - Mesh generation
- **NumPy/SciPy** - Numerical computing
- **Matplotlib/Seaborn** - Plotting with LaTeX support
- **Pandas** - Data analysis and CSV handling

## Installation Notes

Requires FEniCS 2019.1.0 or compatible version. The codebase uses legacy XML mesh format for compatibility with older FEniCS installations.

## Development Notes

Some housekeeping remains in cleaning functions and organising folder outputs, as development time was focused on generating research results. The core simulation and analysis functionality is complete.

---

## AI Development Assistance

AI assistance was used during development to help debug errors, display nicely formatted print statements for the terminal, and to draft docstrings for various functions. The core scientific methodology, mathematical formulations, and FEniCS implementation were developed without.
