##########################################################
# No Advection Simulation Module
##########################################################
"""
This module runs the simulations
"""

# ========================================================
# region Imports
# ========================================================

# General imports
import numpy as np
import os
import time
from datetime import datetime
import json
from dolfin import *

# Default parameters and geometry configurations
from parameters import Parameters

# Mesh generation
from mesh import (
    MeshGenerator,
    setup_sulcus_measures,
    setup_rectangular_measures
)

# Solver Functions
from solvers import (
    stokes_solver,
    stokes_solver_no_adv,
    pure_diffusion_solver,
    pure_diffusion_solver_variable_mu,
    advdiff_solver,
    advdiff_solver_variable_mu
)

# Analysis Functions
from analysis import (
    compute_flux_metrics,
    compute_mass_metrics,
    compute_velocity_metrics,
    compute_mu_eff_metrics
)

# Plotting and exporting functions
from plotting import plot_single_simulation

# endregion

# ========================================================
# Simulation Function
# ========================================================

def export_boundary_flux_comparison(c, mu, D, mesh, boundary_markers, marker_id, output_dir):
    """
    Export physical flux (-D ∇c · n) and Robin flux (μ c) along a given boundary marker.
    """

    # Create measure restricted to boundary
    ds_marker = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    # Space to project flux values (DG0 = cellwise constant per facet)
    DG0 = FunctionSpace(mesh, "DG", 0)

    # Normal vector
    n = FacetNormal(mesh)

    # Compute gradient of concentration
    grad_c = grad(c)

    # Compute physical flux: -D ∇c · n
    physical_flux_expr = -D * dot(grad_c, n)
    physical_flux = project(physical_flux_expr, DG0, form_compiler_parameters={"quadrature_degree": 4})

    # Compute Robin flux: μ c
    robin_flux_expr = mu * c
    robin_flux = project(robin_flux_expr, DG0, form_compiler_parameters={"quadrature_degree": 4})

    # Restrict to boundary using marker
    class BoundaryRegion(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and boundary_markers[mesh.closest_facets(Point(*x))] == marker_id

    bottom_flux_mask = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    BoundaryRegion().mark(bottom_flux_mask, 1)

    # Save as .pvd for Paraview
    File(os.path.join(output_dir, "flux_physical_bottom.pvd")) << physical_flux
    File(os.path.join(output_dir, "flux_robin_bottom.pvd")) << robin_flux

def _simulation_generate_mesh(params, domain_type, mesh_dir, paraview_dir):
    """Generate the mesh for simulation"""

    print("\n Generating mesh...")

    mesh_params = params.get_mesh_generator_params()
    mesh_params['output_dir'] = mesh_dir
    mesh_params['domain_type'] = domain_type

    mesh_gen = MeshGenerator(**mesh_params)
    mesh_results = mesh_gen.generate_mesh()

    # Save PVD files
    mesh_gen.save_mesh_pvd_files(paraview_dir)

    if mesh_results:
        print("✓ Mesh generated successfully!")
        mesh_info = mesh_results['mesh_info']
        print(f"   📊 Mesh Statistics:")
        print(f"      • Vertices: {mesh_info['num_vertices']:,}")
        print(f"      • Elements: {mesh_info['num_cells']:,}")
        print(f"      • h_min: {mesh_info['hmin']:.6f}")
        print(f"      • h_max: {mesh_info['hmax']:.6f}")
        return mesh_results
    else:
        print("✗ Mesh generation failed!")
        return None

def _simulation_generate_vel(mode, domain_type, params, mesh_results, paraview_dir):
    """Generate the velocity field for simulation."""

    print("\n Generating velocity field...")

    mesh = mesh_results['mesh']
    V = VectorFunctionSpace(mesh, "P", 2)
    Q = FunctionSpace(mesh, "P", 1)
    W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))

    if mode == 'no-adv':
        u, p = stokes_solver_no_adv(V, Q)
    else:
        u, p = stokes_solver(mesh_results, W, params.L, params.H, domain_type)

    File(os.path.join(paraview_dir, "velocity.pvd")) << u
    File(os.path.join(paraview_dir, "pressure.pvd")) << p

    return u, p

def _simulation_generate_conc(u, mode, domain_type, params, mesh_results, paraview_dir, mu_variable=False):
    print("\n Generating concentration field...")

    mesh = mesh_results['mesh']
    C = FunctionSpace(mesh, "CG", 2)
    D_const = Constant(params.D)

    mu_val = params.mu
    if isinstance(mu_val, (int, float)):
        mu_val = Constant(mu_val)

    # Trust the flag
    if mode == 'no-adv':
        if mu_variable:
            c = pure_diffusion_solver_variable_mu(mesh_results, C, D_const, mu_val, domain_type)
        else:
            c = pure_diffusion_solver(mesh_results, C, D_const, mu_val, domain_type)
    else:
        if mu_variable:
            c = advdiff_solver_variable_mu(mesh_results, u, C, D_const, mu_val, domain_type)
        else:
            c = advdiff_solver(mesh_results, u, C, D_const, mu_val, domain_type)

    File(os.path.join(paraview_dir, "concentration.pvd")) << c
    return c

def _simulation_post_process(domain_type, params, mesh_results, c, u, p):
    """Post-process simulation results and compute metrics."""
    print("\n Analysing results...")

    # Direct access to mesh data
    mesh = mesh_results['mesh']
    bc_markers = mesh_results['bc_markers']

    # Setup measures based on domain type
    if domain_type == 'sulcus':
        # Direct access to sulcus-specific markers
        bottom_segment_markers = mesh_results['bottom_segment_markers']
        y0_markers = mesh_results['y0_markers']
        domain_markers = mesh_results['domain_markers']

        # Setup sulcus measures
        ds_bc_sulc, ds_bottom, dS_bottom, ds_y0, dS_y0, dx_domain_sulc = setup_sulcus_measures(
            mesh, bc_markers, bottom_segment_markers, y0_markers, domain_markers
        )

        # Create measures dictionary for sulcus
        measures = {
            'ds_bc': ds_bc_sulc,
            'ds_bottom': ds_bottom,
            'dS_bottom': dS_bottom,
            'ds_y0': ds_y0,
            'dS_y0': dS_y0,
            'dx_domain_sulc': dx_domain_sulc
        }

    else:  # Rectangular domain
        # Setup rectangular measures
        ds_bc_rect, dx_domain_rect = setup_rectangular_measures(mesh, bc_markers)

        # Create measures dictionary for rectangular
        measures = {
            'ds_bc': ds_bc_rect,
            'dx_domain_rect': dx_domain_rect
        }

    # Compute flux metrics
    flux_metrics = compute_flux_metrics(c, u, mesh_results, domain_type, measures, params.D, params.mu)

    # Compute mass metrics
    mass_metrics = compute_mass_metrics(c, measures, domain_type)

    # Compute velocity metrics
    vel_metrics = compute_velocity_metrics(u, mesh_results, params)

    results = {
        'c': c,
        'u': u,
        'p': p,
        'mass_metrics': mass_metrics,
        'flux_metrics': flux_metrics,
        'vel_metrics': vel_metrics,
        'params': params,
        'mesh_results': mesh_results,
        'measures': measures
    }

    # Add μ_eff analysis if it's a sulcus domain
    if domain_type == 'sulcus':
        results['mu_eff_comparison'] = compute_mu_eff_metrics(results)

    return results

def _simulation_save_results(results, filename):
    """Save simulation results to JSON file."""
    try:
        # Get mesh info from mesh_results
        mesh_results = results.get('mesh_results', {})
        mesh = mesh_results.get('mesh')

        mesh_info = {}
        if mesh:
            mesh_info = {
                'num_vertices': mesh.num_vertices(),
                'num_cells': mesh.num_cells(),
                'hmin': mesh.hmin(),
                'hmax': mesh.hmax()
            }

        serializable_results = {
            'params': results['params'].to_dict(),
            'mass_metrics': results['mass_metrics'],
            'flux_metrics': results['flux_metrics'],
            'mesh_info': mesh_info,  # Use constructed mesh_info
            'mu_eff_comparison': results.get('mu_eff_comparison', None)
        }

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"✓ Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

def _simulation_plot(results, plots_dir):
    """Generate all plots for the simulation results."""
    plot_single_simulation(results, plots_dir)
    return

def run_simulation(mode, study_type, config_name, domain_type, params, mu_variable=False):
    """Run a simulation."""

    # -----------------------------------------------------------------------------------------
    # region Simulation Set-up
    # -----------------------------------------------------------------------------------------

    # Start clock
    start_time = time.time()

    # ----------------------
    # Validate modes
    valid_modes = ['adv-diff', 'no-adv', 'no-uptake']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

    # ----------------------
    # Validate domain types
    valid_domain_types = ['sulcus', 'rectangular']
    if domain_type not in valid_domain_types:
        raise ValueError(f"Invalid domain type '{domain_type}'. Must be one of: {valid_domain_types}")

    # ----------------------
    # Create base output directory with study type
    # Create a mapping for mode names to handle special cases
    mode_name_mapping = {
        'adv-diff': 'Adv-Diff',
        'no-adv': 'No Advection',
        'no-uptake': 'No Uptake'
    }

    base_output_dir = os.path.join(
        "Results",                                         # Top-level Results folder
        f"{mode_name_mapping.get(mode, mode.replace('-', ' ').title())} Simulations",   # "No Advection Simulations"
        study_type,                                        # "Single Domain", "Geometry Sweep", etc.
        config_name                                        # "Sulcus", "small_sulcus", etc.
    )

    # Setup directories and define paths
    mesh_dir = os.path.join(base_output_dir, "Mesh Files")
    paraview_dir = os.path.join(base_output_dir, "ParaView Files")
    plots_dir = os.path.join(base_output_dir, "Analysis Plots")
    results_dir = os.path.join(base_output_dir, "Results Data")

    for directory in [mesh_dir, paraview_dir, plots_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
    #endregion

    # -----------------------------------------------------------------------------------------
    # region Run Simulation
    # -----------------------------------------------------------------------------------------

    # 1. Generate mesh
    mesh_results = _simulation_generate_mesh(params, domain_type, mesh_dir, paraview_dir)

    # 2. Generate velocity field
    u, p = _simulation_generate_vel(mode, domain_type, params, mesh_results, paraview_dir)

    # 3. Generate concentration field
    c = _simulation_generate_conc(u, mode, domain_type, params, mesh_results, paraview_dir, mu_variable)

    # 4. Post processing analysis
    results = _simulation_post_process(domain_type, params, mesh_results, c, u, p)

    # 5. Generate plots
    _simulation_plot(results, plots_dir)

    # 6. Save results
    results_file = os.path.join(results_dir, "simulation_results.json")
    _simulation_save_results(results, results_file)

    # Print completion message
    elapsed_time = time.time() - start_time
    print(f"\n✓ Simulation completed in {elapsed_time:.1f}s")
    print(f"  Results: {results_file}")
    print(f"  Plots: {plots_dir}")
    print(f"  Visualisation: {paraview_dir}")

    return results
    #endregion

if __name__ == "__main__":

    print("="*50)
    print("RUNNING SIMULATION TEST")
    print("="*50)

    try:

        # Run simulation
        print("\n" + "="*30)
        print("TESTING SIMUMLATION")
        print("="*30)

        # Create test parameters
        params = Parameters(mode='adv-diff')
        params.validate()
        params.nondim()

        sulcus_results = run_simulation(
            mode='adv-diff',
            study_type='Test',
            config_name='Sulcus Test',
            domain_type='sulcus',
            params=params
        )

        print("✅ Simulation completed successfully!")

    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
