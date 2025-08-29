##########################################################
# No Advection Analysis: Phase A
##########################################################
"""
Transport study for sulcus geometries under **no-advection** conditions (pe = 0).
This module mirrors the structure of the "no_uptake_analysis.py" tool but limits
it to the no-advection regime. It includes aspect ratio analysis alongside mu sweep
and other geometric analyses.

Scope:
    - Geometry comparison runs across selected uptake coefficients and geometries
    - Width and depth sweeps
    - Aspect ratio analysis (mu_eff/mu vs depth for different aspect ratios)
    - Mu parameter sweep across uptake regimes
    - CSV creation & plotting
Dependencies (should exist in the project):
    - parameters: Parameters, create_geometry_variations, create_width_variations, create_depth_variations
    - simulation: run_simulation
    - plotting: safe_plot, format_filename_value, create_study_dirs

Directory layout used by this module (consistent with other studies):
    Results/
      └── No Advection Simulations/
          ├── <Study Name> Simulations/
          └── <Study Name> Analysis/
"""

# ========================================================
# region Imports
# ========================================================

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import argparse
import matplotlib.colors as mcolors

from analysis import sample_mu_along_bottom
from parameters import Parameters, create_geometry_variations, create_width_variations, create_depth_variations
from simulation import run_simulation
from plotting import safe_plot, format_filename_value, create_study_dirs

# endregion

# ========================================================
# region Data Processing
# ========================================================

def extract_mu_sweep_data(result, config_name, peclet_num=0):
    """Extract mu sweep specific data from simulation result."""

    row = {
        'Config': config_name,
        'Regime': result.get('regime', 'unknown'),
        'Mu_Factor': result.get('mu_factor', 1.0),
        'Mu_dim': result.get('mu_dim_used'),
        'Mu': result.get('mu_used'),
        'Baseline_Mu_dim': result.get('baseline_mu_dim'),
    }


    # Extract mu_eff data if available
    if 'mu_eff_comparison' in result:
        mu_eff_data = result['mu_eff_comparison']

        # Mu_eff values
        row.update({
            'Mu_Eff_Simulation': mu_eff_data.get('mu_eff_sim'),
            'Mu_Eff_Analytical': mu_eff_data.get('mu_eff_arc'),
            'Mu_Eff_Enhanced': mu_eff_data.get('mu_eff_enh'),
            'Mu_Eff_Opening': mu_eff_data.get('mu_eff_open'),
        })

        # Ratios (mu_eff/mu)
        ratios = mu_eff_data.get('ratios', {})
        row.update({
            'Ratio_Sim': ratios.get('sim'),
            'Ratio_Analytical': ratios.get('arc'),
            'Ratio_Enhanced': ratios.get('enh'),
            'Ratio_Opening': ratios.get('open'),
        })

        # Errors vs simulation (already as percentages)
        errors_vs_sim = mu_eff_data.get('errors_vs_sim', {})
        row.update({
            'Relative_Error_Analytical': errors_vs_sim.get('arc'),
            'Relative_Error_Enhanced': errors_vs_sim.get('enh'),
            'Relative_Error_Opening': errors_vs_sim.get('open'),
        })

    # Mass and flux metrics (simplified for mu sweep focus)
    if 'mass_metrics' in result:
        mm = result['mass_metrics']
        row['Total_Mass'] = mm.get('total_mass')

    if 'flux_metrics' in result:
        fm = result['flux_metrics'] or {}
        mouth = (((fm.get('sulcus_specific') or {})
                .get('physical_flux') or {})
                .get('sulcus_opening') or {})
        row['Mouth_Flux_Total'] = mouth.get('total')

    return row

def extract_aspect_ratio_data(result, config_name, aspect_ratio_type, width, depth, aspect_ratio):
    """Extract aspect ratio specific data from simulation result."""
    row = {
        'Config': config_name,
        'Aspect_Ratio_Type': aspect_ratio_type,
        'Width': width,
        'Depth': depth,
        'Aspect_Ratio': aspect_ratio,
    }

    # Extract parameters
    if 'parameters' in result or hasattr(result, 'mu'):
        mu_value = result.get('mu_used') or result.get('mu', 0)
        row['Mu'] = mu_value

    # Extract mu_eff data if available
    if 'mu_eff_comparison' in result:
        mu_eff_data = result['mu_eff_comparison']

        # Mu_eff values
        row.update({
            'Mu_Eff_Simulation': mu_eff_data.get('mu_eff_sim'),
            'Mu_Eff_Analytical': mu_eff_data.get('mu_eff_arc'),
            'Mu_Eff_Enhanced': mu_eff_data.get('mu_eff_enh'),
            'Mu_Eff_Opening': mu_eff_data.get('mu_eff_open'),
        })

        # Ratios (mu_eff/mu)
        ratios = mu_eff_data.get('ratios', {})
        row.update({
            'Ratio_Sim': ratios.get('sim'),
            'Ratio_Analytical': ratios.get('arc'),
            'Ratio_Enhanced': ratios.get('enh'),
            'Ratio_Opening': ratios.get('open'),
        })

        # Errors vs simulation (already as percentages)
        errors_vs_sim = mu_eff_data.get('errors_vs_sim', {})
        row.update({
            'Relative_Error_Analytical': errors_vs_sim.get('arc'),
            'Relative_Error_Enhanced': errors_vs_sim.get('enh'),
            'Relative_Error_Opening': errors_vs_sim.get('open'),
        })

    # Mass and flux metrics
    if 'mass_metrics' in result:
        mm = result['mass_metrics']
        row['Total_Mass'] = mm.get('total_mass')

    if 'flux_metrics' in result:
        fm = result['flux_metrics'] or {}
        mouth = (((fm.get('sulcus_specific') or {})
                .get('physical_flux') or {})
                .get('sulcus_opening') or {})
        row['Mouth_Flux_Total'] = mouth.get('total')

    return row

def extract_geometry_analysis_data(result, config_name, geometry_name, mu_value, mu_factor, config_results=None):
    """Extract geometry analysis specific data from simulation result."""
    row = {
        'Config': config_name,
        'Geometry_Name': geometry_name,
        'Mu_Value': mu_value,
        'Mu_Factor': mu_factor,
    }

    # Extract geometry from config_results if provided
    if config_results and 'geometry_config' in config_results:
        geo_config = config_results['geometry_config']
        sulcus_width_mm = geo_config.get('sulci_w_dim')
        sulcus_depth_mm = geo_config.get('sulci_h_dim')

        row['Sulcus_Width_mm'] = sulcus_width_mm
        row['Sulcus_Depth_mm'] = sulcus_depth_mm

        # Calculate aspect ratio
        if sulcus_width_mm and sulcus_depth_mm and sulcus_width_mm > 0:
            row['Aspect_Ratio'] = sulcus_depth_mm / sulcus_width_mm

        # Add aspect ratio category
        row['Aspect_Ratio_Category'] = geo_config.get('aspect_ratio_category', 'unknown')

    # Extract mu_eff data if available
    if 'mu_eff_comparison' in result:
        mu_eff_data = result['mu_eff_comparison']

        # Mu_eff values
        row.update({
            'Mu_Eff_Simulation': mu_eff_data.get('mu_eff_sim'),
            'Mu_Eff_Analytical': mu_eff_data.get('mu_eff_arc'),
            'Mu_Eff_Enhanced': mu_eff_data.get('mu_eff_enh'),
            'Mu_Eff_Opening': mu_eff_data.get('mu_eff_open'),
        })

        # Ratios (mu_eff/mu)
        ratios = mu_eff_data.get('ratios', {})
        row.update({
            'Ratio_Sim': ratios.get('sim'),
            'Ratio_Analytical': ratios.get('arc'),
            'Ratio_Enhanced': ratios.get('enh'),
            'Ratio_Opening': ratios.get('open'),
        })

        # Errors vs simulation (already as percentages)
        errors_vs_sim = mu_eff_data.get('errors_vs_sim', {})
        row.update({
            'Relative_Error_Analytical': errors_vs_sim.get('arc'),
            'Relative_Error_Enhanced': errors_vs_sim.get('enh'),
            'Relative_Error_Opening': errors_vs_sim.get('open'),
        })

    # Mass and flux metrics (optional)
    if 'mass_metrics' in result:
        mm = result['mass_metrics']
        row['Total_Mass'] = mm.get('total_mass')

    if 'flux_metrics' in result:
        fm = result['flux_metrics'] or {}
        mouth = (((fm.get('sulcus_specific') or {})
                .get('physical_flux') or {})
                .get('sulcus_opening') or {})
        row['Mouth_Flux_Total'] = mouth.get('total')

    return row

def extract_mu_eff_analysis_data(result, config_name, mu_value, mu_factor):
    """Extract mu_eff analysis data from simulation result."""
    row = {
        'Config': config_name,
        'Mu_Value': mu_value,
        'Mu_Factor': mu_factor,
    }

    # Extract geometry parameters from the Parameters object stored in results
    params = result.get('params')
    if params is not None:
        row.update({
            'Sulcus_Width_mm': getattr(params, 'sulci_w_dim', 0.5),
            'Sulcus_Depth_mm': getattr(params, 'sulci_h_dim', 1.0),
            'Domain_Length_mm': getattr(params, 'L_dim', 10.0),
            'L_ref': getattr(params, 'L_ref', getattr(params, 'H_dim', 1.0)),
            'L_nondim': getattr(params, 'L', 10.0),
            'H_nondim': getattr(params, 'H', 1.0),
            'Sulcus_W_nondim': getattr(params, 'sulci_w', 0.5),
            'Sulcus_H_nondim': getattr(params, 'sulci_h', 1.0),
            'Mu_base_nondim': getattr(params, 'mu', mu_value),
        })
    else:
        # Fallback values
        row.update({
            'Sulcus_Width_mm': 0.5, 'Sulcus_Depth_mm': 1.0, 'Domain_Length_mm': 10.0,
            'L_ref': 1.0, 'L_nondim': 10.0, 'H_nondim': 1.0,
            'Sulcus_W_nondim': 0.5, 'Sulcus_H_nondim': 1.0, 'Mu_base_nondim': mu_value,
        })

    # Extract mu_eff comparison
    if 'mu_eff_comparison' in result:
        mu_eff_data = result['mu_eff_comparison']
        row.update({
            'Mu_Eff_Simulation': mu_eff_data.get('mu_eff_sim'),
            'Mu_Eff_Analytical': mu_eff_data.get('mu_eff_arc'),
            'Mu_Eff_Enhanced': mu_eff_data.get('mu_eff_enh'),
            'Mu_Eff_Opening': mu_eff_data.get('mu_eff_open'),
        })
        ratios = mu_eff_data.get('ratios', {})
        row.update({
            'Ratio_Sim': ratios.get('sim'), 'Ratio_Analytical': ratios.get('arc'),
            'Ratio_Enhanced': ratios.get('enh'), 'Ratio_Opening': ratios.get('open'),
        })

    # ADD MU SAMPLING - sample mu(x) along bottom
    try:
        mu_sample = sample_mu_along_bottom(result, n_points=100)
        row.update({
            'Mu_Mean_Bottom': mu_sample['mu_mean'],
            'Mu_Min_Bottom': mu_sample['mu_min'],
            'Mu_Max_Bottom': mu_sample['mu_max'],
            # Store the arrays as JSON strings for CSV compatibility
            'Mu_X_Array': str(mu_sample['x'].tolist()),
            'Mu_Values_Array': str(mu_sample['mu'].tolist()),
        })
    except Exception as e:
        print(f"Warning: Could not sample mu for {config_name}: {e}")
        row.update({
            'Mu_Mean_Bottom': None, 'Mu_Min_Bottom': None, 'Mu_Max_Bottom': None,
            'Mu_X_Array': None, 'Mu_Values_Array': None,
        })

    return row


def create_mu_sweep_csv(results, study_dir, study_name):
    """Create CSV from simulation results."""
    data = []
    for config_name, result in results.items():
        if result:
            peclet = result.get('peclet', result.get('target_peclet', 0))
            data.append(extract_mu_sweep_data(result, config_name, peclet))

    if not data:
        print("⚠️ No valid data for CSV")
        return None

    df = pd.DataFrame(data)
    csv_path = os.path.join(study_dir, f'{study_name}_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"✅ CSV saved: {csv_path} ({df.shape[0]} rows, {len(df['Config'].unique())} configs)")
    return csv_path

def create_aspect_ratio_csv(results, study_dir, study_name):
    """Create CSV from aspect ratio simulation results."""
    data = []
    for config_name, result in results.items():
        if result and 'aspect_ratio_metadata' in result:
            metadata = result['aspect_ratio_metadata']
            data.append(extract_aspect_ratio_data(
                result, config_name,
                metadata['aspect_ratio_type'],
                metadata['width'],
                metadata['depth'],
                metadata['aspect_ratio']
            ))

    if not data:
        print("⚠️ No valid data for aspect ratio CSV")
        return None

    df = pd.DataFrame(data)
    csv_path = os.path.join(study_dir, f'{study_name}_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"✅ Aspect ratio CSV saved: {csv_path} ({df.shape[0]} rows, {len(df['Aspect_Ratio_Type'].unique())} types)")
    return csv_path

def create_geometry_analysis_csv(all_simulation_results, study_dir, study_name):
    """Create CSV from geometry analysis simulation results."""
    data = []

    for config_name, config_results in all_simulation_results.items():
        if not config_results or 'sulcus' not in config_results:
            continue

        result = config_results['sulcus']
        geometry_name = config_results.get('geometry_name', 'unknown')
        mu_value = config_results.get('mu_value', None)
        mu_factor = config_results.get('mu_factor', None)

        if result:
            row_data = extract_geometry_analysis_data(
                result=result,
                config_name=config_name,
                geometry_name=geometry_name,
                mu_value=mu_value,
                mu_factor=mu_factor,
                config_results=config_results
            )
            data.append(row_data)

    if not data:
        print("⚠️ No valid data for geometry analysis CSV")
        return None

    df = pd.DataFrame(data)
    csv_path = os.path.join(study_dir, f'{study_name}_results.csv')
    df.to_csv(csv_path, index=False)

    # Print summary
    total_sims = len(data)
    configs = df['Config'].nunique()
    geometries = df['Geometry_Name'].nunique()
    mu_values = df['Mu_Value'].nunique()

    print(f"✅ Geometry analysis CSV saved: {csv_path}")
    print(f"   Total simulations: {total_sims}")
    print(f"   Configurations: {configs}")
    print(f"   Geometries: {geometries}")
    print(f"   Mu values: {mu_values}")

    return csv_path

def create_mu_eff_analysis_csv(results, study_dir, study_name):
    """Create CSV from mu_eff analysis results."""
    data = []
    for config_name, result in results.items():
        if result:
            # Extract mu info from config name
            mu_factor = float(config_name.split('_mu_')[1].replace('x', ''))
            mu_value = result.get('mu_value', mu_factor * Parameters.MU_DIM_NO_ADV)

            data.append(extract_mu_eff_analysis_data(result, config_name, mu_value, mu_factor))

    if not data:
        print("No valid data for mu_eff analysis CSV")
        return None

    df = pd.DataFrame(data)
    csv_path = os.path.join(study_dir, f'{study_name}_results.csv')
    df.to_csv(csv_path, index=False)

    print(f"Mu_eff analysis CSV saved: {csv_path} ({df.shape[0]} rows)")
    return csv_path

# endregion

# ========================================================
# region Plotting
# ========================================================

def create_mu_sweep_plots(csv_file, legend_config=None):
    """Generate mu sweep plots: relative error, absolute error, correlation, and ratios."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"⚠ Failed to read CSV: {e}")
        return

    plots_dir = os.path.join(os.path.dirname(csv_file), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"📊 Generating mu sweep plots from {df.shape[0]} data points...")

    # =========================
    # --- Configuration -------
    # =========================

    # Individual marker sizes
    marker_sizes = {
        'relative_error': 4,
        'absolute_error': 4,
        'correlation': 4,
        'ratios': 4
    }

    # Handle legend configuration
    default_legend_config = {
        'fontsize': 11,
        'position': 'upper left',
        'frameon': True,
        'shadow': False,
        'bbox_to_anchor': None,
        'ncol': 1
    }

    if legend_config is None:
        legend_config = default_legend_config
    else:
        # Merge with defaults
        for key, value in default_legend_config.items():
            if key not in legend_config:
                legend_config[key] = value

    # Plotting configuration
    PLOT_CFG = {
        "figsize": (4, 4),
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": legend_config['fontsize'],
        "lines.linewidth": 1,
        "axes.linewidth": 0.5,
        "suptitle.size": 16,
        "suptitle.pad": 18,
    }

    # Apply font sizes and line widths globally
    plt.rcParams.update({
        "font.size": PLOT_CFG["font.size"],
        "axes.titlesize": PLOT_CFG["axes.titlesize"],
        "axes.labelsize": PLOT_CFG["axes.labelsize"],
        "xtick.labelsize": PLOT_CFG["xtick.labelsize"],
        "ytick.labelsize": PLOT_CFG["ytick.labelsize"],
        "legend.fontsize": PLOT_CFG["legend.fontsize"],
        "lines.linewidth": PLOT_CFG["lines.linewidth"],
        "axes.linewidth": PLOT_CFG["axes.linewidth"],
    })

    # Color mapping for regimes
    regime_colors = {
        'small_uptake': '#2E86AB',     # Blue
        'moderate_uptake': '#F18F01',  # Orange
        'high_uptake': '#F24236'       # Red
    }

    regime_names = {
        'small_uptake': 'Small Uptake',
        'moderate_uptake': 'Moderate Uptake',
        'high_uptake': 'High Uptake'
    }

    plt.style.use('default')  # Ensure clean plotting style

    def apply_legend(ax, plot_type='default'):
        """Apply legend with specified configuration"""
        legend_kwargs = {
            'fontsize': legend_config['fontsize'],
            'frameon': legend_config['frameon'],
            'shadow': legend_config['shadow'],
            'ncol': legend_config['ncol']
        }

        if legend_config['bbox_to_anchor'] is not None:
            legend_kwargs['bbox_to_anchor'] = legend_config['bbox_to_anchor']
            legend_kwargs['loc'] = 'upper left'
        else:
            legend_kwargs['loc'] = legend_config['position']

        return ax.legend(**legend_kwargs)

    # 2. Absolute Error vs Mu Factor (Line Plot)
    with safe_plot(os.path.join(plots_dir, 'absolute_error_vs_mu_factor.png'), PLOT_CFG["figsize"]):
        fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"])

        # Calculate and plot absolute error for each regime
        for regime in df['Regime'].unique():
            if regime in regime_colors:
                regime_data = df[df['Regime'] == regime].sort_values('Mu_Factor')

                # Calculate absolute error if we have both analytical and simulation values
                if ('Mu_Eff_Analytical' in regime_data.columns and
                    'Mu_Eff_Simulation' in regime_data.columns):

                    valid_data = regime_data.dropna(subset=['Mu_Eff_Analytical', 'Mu_Eff_Simulation'])

                    if len(valid_data) > 0:
                        absolute_error = abs(valid_data['Mu_Eff_Analytical'] - valid_data['Mu_Eff_Simulation'])

                        ax.plot(valid_data['Mu_Factor'], absolute_error,
                               marker='o', linewidth=PLOT_CFG["lines.linewidth"],
                               markersize=marker_sizes['absolute_error'],
                               color=regime_colors[regime],
                               label=regime_names.get(regime, regime.replace('_', ' ').title()))

        ax.set_xlabel(r'Mu Factor ($\times$baseline)', fontweight='bold')
        ax.set_ylabel(r'Absolute Error $|\mu_{\mathrm{eff,analytical}} - \mu_{\mathrm{eff,simulation}}|$', fontweight='bold')
        ax.set_title('Analytical Model Absolute Error vs Uptake Strength', fontweight='bold')
        ax.grid(True, alpha=0.3)
        apply_legend(ax, 'absolute_error')

        # Set log scale for x-axis to better show the wide range
        ax.set_xscale('log')
        ax.set_xlim(left=0.1)
        ax.set_ylim(bottom=0)

        plt.tight_layout()

    # 3. Correlation Plot (Analytical vs Simulation)
    with safe_plot(os.path.join(plots_dir, 'analytical_vs_simulation_correlation.png'), PLOT_CFG["figsize"]):
        fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"])

        # Plot data points colored by regime
        for regime in df['Regime'].unique():
            if regime in regime_colors:
                regime_data = df[df['Regime'] == regime]

                # Only plot if we have both analytical and simulation data
                valid_data = regime_data.dropna(subset=['Mu_Eff_Analytical', 'Mu_Eff_Simulation'])
                if len(valid_data) > 0:
                    ax.scatter(valid_data['Mu_Eff_Analytical'], valid_data['Mu_Eff_Simulation'],
                              color=regime_colors[regime], alpha=0.7,
                              s=marker_sizes['correlation']**2,  # scatter uses area, so square the size
                              label=regime_names.get(regime, regime.replace('_', ' ').title()))

        # Perfect correlation line
        if len(df.dropna(subset=['Mu_Eff_Analytical', 'Mu_Eff_Simulation'])) > 0:
            all_values = pd.concat([
                df['Mu_Eff_Analytical'].dropna(),
                df['Mu_Eff_Simulation'].dropna()
            ])
            min_val = all_values.min()
            max_val = all_values.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, linewidth=2,
                   label='Perfect Correlation')

        ax.set_xlabel(r'Analytical $\mu_{\mathrm{eff}}$', fontweight='bold')
        ax.set_ylabel(r'Simulation $\mu_{\mathrm{eff}}$', fontweight='bold')
        ax.set_title(r'Analytical vs Simulation $\mu_{\mathrm{eff}}$ Correlation', fontweight='bold')
        ax.grid(True, alpha=0.3)
        apply_legend(ax, 'correlation')

        # Equal aspect ratio for correlation plot
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

    # 1 & 4. Combined: Relative Error and Enhancement Ratios vs Mu Factor
    with safe_plot(os.path.join(plots_dir, 'relative_error_and_ratios_vs_mu_factor.png'), (8, 4)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # Add the suptitle here
        fig.suptitle(r'Uptake Coefficient $\mu$ Sweep Analysis Plots',
                    fontsize=PLOT_CFG["suptitle.size"],
                    y=0.97)

        # Plot 1: Relative Error vs Mu Factor (Left subplot)
        for regime in df['Regime'].unique():
            if regime in regime_colors:
                regime_data = df[df['Regime'] == regime].sort_values('Mu_Factor')
                base_color = regime_colors[regime]

                # Plot Analytical relative error (solid line, circles)
                if 'Relative_Error_Analytical' in regime_data.columns and regime_data['Relative_Error_Analytical'].notna().any():
                    ax1.plot(regime_data['Mu_Factor'], regime_data['Relative_Error_Analytical'],
                        marker='o', linewidth=PLOT_CFG["lines.linewidth"],
                        markersize=marker_sizes['relative_error'],
                        color=base_color, linestyle='-',
                        label='_nolegend_')

                # Plot Enhanced relative error (dashed line, diamonds)
                if 'Relative_Error_Enhanced' in regime_data.columns and regime_data['Relative_Error_Enhanced'].notna().any():
                    import matplotlib.colors as mcolors
                    light_color = mcolors.to_rgba(base_color, alpha=0.8)
                    ax1.plot(regime_data['Mu_Factor'], regime_data['Relative_Error_Enhanced'],
                        marker='D', linewidth=PLOT_CFG["lines.linewidth"],
                        markersize=marker_sizes['relative_error'],
                        color=light_color, linestyle='--',
                        label='_nolegend_')

        ax1.set_xlabel(r'Mu Factor ($\times$baseline)', fontweight='bold')
        ax1.set_ylabel(r'Relative Error (\%)', fontweight='bold')
        ax1.set_title(r'Model Relative Errors for $\mu_{\mathrm{eff}}$ vs $\mu_{\mathrm{eff}}^{\mathrm{sim}}$', fontweight='bold', fontsize = 11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_xlim(left=0.1)
        ax1.set_ylim(bottom=0)

        # Plot 4: Enhancement Ratios vs Mu Factor (Right subplot) - Now includes Enhanced
        for regime in df['Regime'].unique():
            if regime in regime_colors:
                regime_data = df[df['Regime'] == regime].sort_values('Mu_Factor')
                base_color = regime_colors[regime]

                # Plot analytical ratios (solid line, circles)
                if 'Ratio_Analytical' in regime_data.columns and regime_data['Ratio_Analytical'].notna().any():
                    ax2.plot(regime_data['Mu_Factor'], regime_data['Ratio_Analytical'],
                        marker='o', linewidth=PLOT_CFG["lines.linewidth"] + 0.5,
                        markersize=marker_sizes['ratios'],
                        color=base_color, linestyle='-',
                        label='_nolegend_')

                # Plot enhanced ratios (dashed line, diamonds) - CHANGED ALPHA TO MATCH ENHANCED MU
                if 'Ratio_Enhanced' in regime_data.columns and regime_data['Ratio_Enhanced'].notna().any():
                    import matplotlib.colors as mcolors
                    light_color = mcolors.to_rgba(base_color, alpha=0.8)
                    ax2.plot(regime_data['Mu_Factor'], regime_data['Ratio_Enhanced'],
                        marker='D', linewidth=PLOT_CFG["lines.linewidth"],
                        markersize=marker_sizes['ratios']-0.5, linestyle='--',
                        color=light_color,
                        label='_nolegend_')

                # Plot simulation ratios (dotted line, triangles)
                if 'Ratio_Sim' in regime_data.columns and regime_data['Ratio_Sim'].notna().any():
                    lighter_color = mcolors.to_rgba(base_color, alpha=0.8)
                    ax2.plot(regime_data['Mu_Factor'], regime_data['Ratio_Sim'],
                        marker='^', linewidth=PLOT_CFG["lines.linewidth"],
                        markersize=marker_sizes['ratios'], linestyle=':',
                        color=lighter_color,
                        label='_nolegend_')

        # Reference line at y=1 (no enhancement)
        ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, label='_nolegend_')

        ax2.set_xlabel(r'Mu Factor ($\times$baseline)', fontweight='bold')
        ax2.set_ylabel(r'$\mu_{\mathrm{eff}}/\mu$ Ratio', fontweight='bold')
        ax2.set_title(r'Enhancement Factor $\mu_{\mathrm{eff}}$ / $\mu$', fontweight='bold', fontsize = 11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_xlim(left=0.1)
        ax2.set_ylim(bottom=0.85)

        # Create comprehensive legend in the top left of the first plot
        legend_elements = [
            # Color/regime entries
            plt.Line2D([0], [0], color=regime_colors['small_uptake'], linewidth=2,
                      label='Small'),
            plt.Line2D([0], [0], color=regime_colors['moderate_uptake'], linewidth=2,
                      label='Moderate'),
            plt.Line2D([0], [0], color=regime_colors['high_uptake'], linewidth=2,
                      label='High'),
            # Marker/method type entrie
            plt.Line2D([0], [0], marker='o', color='gray', linewidth=0, markersize=4,
                      label=r'Arc $\mu_{\mathrm{eff}}^{\mathrm{arc}}$'),
            plt.Line2D([0], [0], marker='D', color='gray', linewidth=0, markersize=4,
                      label=r'Enh $\mu_{\mathrm{eff}}^{\mathrm{enh}}$'),
            plt.Line2D([0], [0], marker='^', color='gray', linewidth=0, markersize=4,
                      label=r'Sim $\mu_{\mathrm{eff}}^{\mathrm{sim}}$')
        ]

        # Place compact legend in top left of first plot
        ax1.legend(handles=legend_elements,
                  loc='upper left',
                  fontsize=6,
                  frameon=True,
                  shadow=False,
                  ncol=1,
                  columnspacing=0.5,
                  handlelength=1.5,
                  handletextpad=0.3,
                  labelspacing=0.3,
                  framealpha=0.9)

        plt.tight_layout()

    print(f"✅ Combined subplot saved: relative_error_and_ratios_vs_mu_factor.png")

    print(f"✅ Mu sweep plots saved to {plots_dir}")
    print(f"   • Relative error plot: relative_error_vs_mu_factor.png (no legend)")
    print(f"   • Absolute error plot: absolute_error_vs_mu_factor.png")
    print(f"   • Correlation plot: analytical_vs_simulation_correlation.png")
    print(f"   • Enhancement ratios: mu_eff_ratios_vs_mu_factor.png (no legend)")

def create_aspect_ratio_plots(csv_file):
    """Generate aspect ratio analysis plots."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"⚠️ Failed to read CSV: {e}")
        return

    plots_dir = os.path.join(os.path.dirname(csv_file), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"📊 Generating aspect ratio plots from {df.shape[0]} data points...")

    # Aspect ratio styles and colors
    aspect_styles = {
        'h_equals_w': {'base_color': '#008080', 'name': r'$h = w$ (square)'},
        'h_equals_2w': {'base_color': '#FF8C00', 'name': r'$h = 2w$ (narrow and deep)'},
        'h_equals_half_w': {'base_color': '#8B008B', 'name': r'$h = 0.5w$ (wide and shallow)'}
    }

    # Line styles for different ratio types (REMOVED Analytical)
    ratio_styles = {
        'Ratio_Sim': {'linestyle': '-', 'marker': '^', 'alpha': 0.9},
        'Ratio_Enhanced': {'linestyle': '-', 'marker': 'D', 'alpha': 0.9}
    }

    # 1. Main plot: Enhancement ratios for each aspect ratio
    with safe_plot(os.path.join(plots_dir, 'aspect_ratio_all_methods.png'), (6, 4)):
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot each aspect ratio with both methods
        for ar_type in df['Aspect_Ratio_Type'].unique():
            if ar_type in aspect_styles:
                ar_subset = df[df['Aspect_Ratio_Type'] == ar_type].sort_values('Depth')
                base_color = aspect_styles[ar_type]['base_color']

                # Plot both ratio types for this aspect ratio
                for ratio_col, style in ratio_styles.items():
                    if ratio_col in ar_subset.columns and ar_subset[ratio_col].notna().any():
                        ax.plot(ar_subset['Depth'], ar_subset[ratio_col],
                               color=base_color,
                               linestyle=style['linestyle'],
                               marker=style['marker'],
                               markersize=6,
                               alpha=style['alpha'],
                               label='_nolegend_',
                               linewidth=1)

        # Formatting
        ax.set_xlabel(r'Sulcus Depth $h$ (mm)', fontsize=16, fontweight='bold')
        ax.set_ylabel(r'$\mu_{\mathrm{eff}} / \mu$ Ratio', fontsize=16, fontweight='bold')
        ax.set_title(r'Enhancement Ratio $\mu_{\mathrm{eff}}$ / $\mu$ vs Depth', fontsize=18, fontweight='bold', y=1.02)

        # Increase tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)

        ax.set_xscale('log')
        ax.set_xlim(left=0.01, right=2.0)

        # Reference line at y=1 (no enhancement)
        ax.axhline(y=1.0, color='grey', linestyle='--', alpha=0.7, linewidth=1)

        # Grid
        ax.grid(True, alpha=0.3)

        # Create comprehensive legend similar to mu sweep plots
        legend_elements = [
            # Aspect ratio entries (color lines only)
            plt.Line2D([0], [0], color=aspect_styles['h_equals_w']['base_color'], linewidth=1,
                      label=r'$h = w$ (square)'),
            plt.Line2D([0], [0], color=aspect_styles['h_equals_2w']['base_color'], linewidth=1,
                      label=r'$h = 2w$ (narrow and deep)'),
            plt.Line2D([0], [0], color=aspect_styles['h_equals_half_w']['base_color'], linewidth=1,
                      label=r'$h = 0.5w$ (wide and shallow)'),
            # Method type entries (marker shapes only)
            plt.Line2D([0], [0], marker='^', color='gray', linewidth=0, markersize=6,
                      label=r'Sim $\mu_{\mathrm{eff}}^{\mathrm{sim}}$'),
            plt.Line2D([0], [0], marker='D', color='gray', linewidth=0, markersize=6,
                      label=r'Enh $\mu_{\mathrm{eff}}^{\mathrm{enh}}$')
        ]

        # Place legend in bottom left
        ax.legend(handles=legend_elements,
                  loc='lower left',
                  fontsize=12,
                  frameon=True,
                  shadow=False,
                  ncol=1,
                  columnspacing=0.5,
                  handlelength=1.5,
                  handletextpad=0.3,
                  labelspacing=0.3,
                  framealpha=0.9)

        plt.tight_layout()

    # 2. Separate subplots for each aspect ratio
    with safe_plot(os.path.join(plots_dir, 'aspect_ratio_subplots.png'), (8, 6)):
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        fig.suptitle(r'Enhancement Ratio $\mu_{\mathrm{eff}}$ / $\mu$ by Aspect Ratio', fontsize=18, fontweight='bold', y=1.02)

        for i, ar_type in enumerate(df['Aspect_Ratio_Type'].unique()):
            if ar_type in aspect_styles and i < len(axes):
                ar_subset = df[df['Aspect_Ratio_Type'] == ar_type].sort_values('Depth')
                ax = axes[i]
                base_color = aspect_styles[ar_type]['base_color']
                base_name = aspect_styles[ar_type]['name']

                # Plot both methods
                for ratio_col, style in ratio_styles.items():
                    if ratio_col in ar_subset.columns and ar_subset[ratio_col].notna().any():
                        if 'Sim' in ratio_col:
                            method_name = 'Simulation'
                        elif 'Enhanced' in ratio_col:
                            method_name = 'Enhanced'
                        else:
                            method_name = ratio_col

                        ax.plot(ar_subset['Depth'], ar_subset[ratio_col],
                               color=base_color,
                               linestyle=style['linestyle'],
                               marker=style['marker'],
                               markersize=6,
                               label=method_name,
                               linewidth=1,
                               alpha=style['alpha'])

                # Reference line at y=1 (dashed grey)
                ax.axhline(y=1.0, color='grey', linestyle='--', alpha=0.7, linewidth=1)

                ax.set_xscale('log')
                ax.set_xlim(left=0.01, right=2.0)

                # Formatting with larger fonts
                ax.set_title(base_name, fontsize=18, fontweight='bold')
                ax.set_ylabel(r'$\mu_{\mathrm{eff}} / \mu$ Ratio', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=12)
                ax.set_ylim(bottom=0.95)

        # X-label only on bottom plot with larger font
        if len(axes) > 0:
            axes[-1].set_xlabel(r'Sulcus Depth $h$ (mm)', fontsize=16, fontweight='bold')

        plt.tight_layout()

    # 3. Relative Error comparison plot (Enhanced and Analytical)
    with safe_plot(os.path.join(plots_dir, 'model_error_comparison.png'), (6, 4)):
        fig, ax = plt.subplots(figsize=(6, 4))

        for ar_type in df['Aspect_Ratio_Type'].unique():
            if ar_type in aspect_styles:
                ar_subset = df[df['Aspect_Ratio_Type'] == ar_type].sort_values('Depth')
                base_color = aspect_styles[ar_type]['base_color']

                # Analytical model error (solid line, circle markers)
                if 'Mu_Eff_Analytical' in ar_subset.columns and 'Mu_Eff_Simulation' in ar_subset.columns:
                    analytical_error = abs(ar_subset['Mu_Eff_Simulation'] - ar_subset['Mu_Eff_Analytical'])
                    analytical_rel_error = 100 * analytical_error / ar_subset['Mu_Eff_Simulation']

                    ax.plot(ar_subset['Depth'], analytical_rel_error,
                            color=base_color, linestyle='-', marker='o', markersize=6,
                            label='_nolegend_', linewidth=1, alpha=0.9)

                # Enhanced model error
                if 'Mu_Eff_Enhanced' in ar_subset.columns and 'Mu_Eff_Simulation' in ar_subset.columns:
                    enhanced_error = abs(ar_subset['Mu_Eff_Simulation'] - ar_subset['Mu_Eff_Enhanced'])
                    enhanced_rel_error = 100 * enhanced_error / ar_subset['Mu_Eff_Simulation']

                    ax.plot(ar_subset['Depth'], enhanced_rel_error,
                            color=base_color, linestyle='-', marker='D', markersize=6,
                            label='_nolegend_', linewidth=1, alpha=0.9)

        ax.set_xscale('log')
        ax.set_xlim(left=0.01, right=2.0)

        # Larger font sizes for labels and title
        ax.set_xlabel(r'Sulcus Depth $h$ (mm)', fontsize=16, fontweight='bold')
        ax.set_ylabel(r'Relative Error (\%) for $\mu_{\mathrm{eff}}$ vs $\mu_{\mathrm{eff}}^{\mathrm{sim}}$', fontsize=16)
        ax.set_title(r'Model Relative Error vs Depth', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)

        # Create comprehensive legend like the main plot
        legend_elements = [
            # Aspect ratio entries (color lines only)
            plt.Line2D([0], [0], color=aspect_styles['h_equals_w']['base_color'], linewidth=1,
                      label=r'$h = w$ (square)'),
            plt.Line2D([0], [0], color=aspect_styles['h_equals_2w']['base_color'], linewidth=1,
                      label=r'$h = 2w$ (narrow and deep)'),
            plt.Line2D([0], [0], color=aspect_styles['h_equals_half_w']['base_color'], linewidth=1,
                      label=r'$h = 0.5w$ (wide and shallow)'),
            # Method type entries (marker shapes only)
            plt.Line2D([0], [0], marker='o', color='gray', linewidth=0, markersize=6,
                      label=r'Arc $\mu_{\mathrm{eff}}^{\mathrm{arc}}$'),
            plt.Line2D([0], [0], marker='D', color='gray', linewidth=0, markersize=6,
                      label=r'Enh $\mu_{\mathrm{eff}}^{\mathrm{enh}}$')
        ]

        # Place legend
        ax.legend(handles=legend_elements,
                  loc='upper left',
                  fontsize=12,
                  frameon=True,
                  shadow=False,
                  ncol=1,
                  columnspacing=0.5,
                  handlelength=1.5,
                  handletextpad=0.3,
                  labelspacing=0.3,
                  framealpha=0.9)

        plt.tight_layout()

    print(f"✅ Aspect ratio plots saved to {plots_dir}")

def create_geometry_analysis_plots(csv_file):
    """Generate geometry analysis error heatmap plots as two multi-panel PNG files."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"⚠️ Failed to read CSV: {e}")
        return

    plots_dir = os.path.join(os.path.dirname(csv_file), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"📊 Generating geometry analysis plots from {df.shape[0]} data points...")

    # Get unique mu values and sort them
    mu_values = sorted(df['Mu_Value'].dropna().unique())
    if not mu_values:
        print("     ⚠️ No Mu values found")
        return

    # Get corresponding mu factors for titles
    mu_factors = {}
    for mu_val in mu_values:
        mu_factor = df[df['Mu_Value'] == mu_val]['Mu_Factor'].iloc[0]
        mu_factors[mu_val] = mu_factor

    print(f"     📊 Creating geometry analysis error heatmaps for mu values: {mu_values}")
    print(f"     📊 Corresponding mu factors: {list(mu_factors.values())}")

    # Check if we have the error columns
    required_cols = ['Absolute_Error', 'Relative_Error_Pct', 'Sulcus_Width_mm', 'Sulcus_Depth_mm']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"     ⚠️ Missing required columns: {missing_cols}")
        return

    # Get color limits for both error types
    abs_errors = df['Absolute_Error'].dropna()
    rel_errors = df['Relative_Error_Pct'].dropna()

    if abs_errors.empty or rel_errors.empty:
        print("     ⚠️ No valid error values found")
        return

    abs_vmin, abs_vmax = abs_errors.min(), abs_errors.max()
    rel_vmin, rel_vmax = rel_errors.min(), rel_errors.max()

    # 1. Create Absolute Error Multi-Panel Plot
    filename_abs = "geometry_analysis_absolute_error_combined.png"

    with safe_plot(os.path.join(plots_dir, filename_abs), (18, 6)):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(r'Absolute Error Analysis', fontsize=16, fontweight='bold', y=0.95)

        # Create a list to store scatter objects for colorbar
        scatters = []

        for i, mu_val in enumerate(mu_values):
            ax = axes[i]
            mu_data = df[df['Mu_Value'] == mu_val]
            abs_data = mu_data.dropna(subset=['Sulcus_Width_mm', 'Sulcus_Depth_mm', 'Absolute_Error'])

            if not abs_data.empty:
                scatter = ax.scatter(
                    abs_data['Sulcus_Width_mm'],
                    abs_data['Sulcus_Depth_mm'],
                    c=abs_data['Absolute_Error'],
                    s=120, alpha=0.9, cmap='Reds',
                    vmin=abs_vmin, vmax=abs_vmax, marker='o',
                    edgecolors='black', linewidth=1
                )
                scatters.append(scatter)

                # Annotations
                for _, row in abs_data.iterrows():
                    abs_err = row["Absolute_Error"]
                    if abs_err < 0.01:
                        label = f'{abs_err:.2e}'
                    else:
                        label = f'{abs_err:.3f}'

                    ax.annotate(label,
                              (row['Sulcus_Width_mm'], row['Sulcus_Depth_mm']),
                              xytext=(3, 3), textcoords='offset points', fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, r'No data for $\mu_{\mathrm{factor}} = ' + f'{mu_factors[mu_val]}' + r'$',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)

            ax.set_title(r'$\mu_{\mathrm{factor}} = ' + f'{mu_factors[mu_val]}' + r'$',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Sulcus Width (mm)', fontsize=12)
            if i == 0:  # Only label y-axis for leftmost panel
                ax.set_ylabel('Sulcus Depth (mm)', fontsize=12)
            ax.grid(True, alpha=0.3)

        # Add shared horizontal colorbar
        if scatters:
            fig.subplots_adjust(bottom=0.25, top=0.85)
            cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
            cbar = fig.colorbar(scatters[0], cax=cbar_ax, orientation='horizontal')
            cbar.set_label(r'Absolute Error $|\mu_{\mathrm{eff,analytical}} - \mu_{\mathrm{eff,simulation}}|$',
                          fontsize=12, fontweight='bold')
        else:
            fig.subplots_adjust(top=0.85)

    print(f"     ✅ Created: {filename_abs}")

    # 2. Create Relative Error Multi-Panel Plot
    filename_rel = "geometry_analysis_relative_error_combined.png"

    with safe_plot(os.path.join(plots_dir, filename_rel), (18, 6)):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(r'Relative Error Analysis', fontsize=16, fontweight='bold', y=0.95)

        # Create a list to store scatter objects for colorbar
        scatters = []

        for i, mu_val in enumerate(mu_values):
            ax = axes[i]
            mu_data = df[df['Mu_Value'] == mu_val]
            rel_data = mu_data.dropna(subset=['Sulcus_Width_mm', 'Sulcus_Depth_mm', 'Relative_Error_Pct'])

            if not rel_data.empty:
                scatter = ax.scatter(
                    rel_data['Sulcus_Width_mm'],
                    rel_data['Sulcus_Depth_mm'],
                    c=rel_data['Relative_Error_Pct'],
                    s=120, alpha=0.9, cmap='RdYlBu_r',
                    vmin=rel_vmin, vmax=rel_vmax, marker='o',
                    edgecolors='black', linewidth=1
                )
                scatters.append(scatter)

                # Annotations
                for _, row in rel_data.iterrows():
                    ax.annotate(f'{row["Relative_Error_Pct"]:.1f}%',
                               (row['Sulcus_Width_mm'], row['Sulcus_Depth_mm']),
                               xytext=(3, 3), textcoords='offset points', fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, r'No data for $\mu_{\mathrm{factor}} = ' + f'{mu_factors[mu_val]}' + r'$',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)

            ax.set_title(r'$\mu_{\mathrm{factor}} = ' + f'{mu_factors[mu_val]}' + r'$',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Sulcus Width (mm)', fontsize=12)
            if i == 0:  # Only label y-axis for leftmost panel
                ax.set_ylabel('Sulcus Depth (mm)', fontsize=12)
            ax.grid(True, alpha=0.3)

        # Add shared horizontal colorbar
        if scatters:
            fig.subplots_adjust(bottom=0.25, top=0.85)
            cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
            cbar = fig.colorbar(scatters[0], cax=cbar_ax, orientation='horizontal')
            cbar.set_label(r'Relative Error (\%)', fontsize=12, fontweight='bold')
        else:
            fig.subplots_adjust(top=0.85)

    print(f"     ✅ Created: {filename_rel}")
    print(f"✅ Geometry analysis plots saved to {plots_dir}")

def create_mu_eff_spatial_plots(csv_file):
    """Generate spatial mu_eff plots showing only the step function mu(x) and simulation mu_eff (single row)."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    plots_dir = os.path.join(os.path.dirname(csv_file), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Generating mu_eff spatial plots from {df.shape[0]} data points...")

    required_cols = ['Mu_Factor', 'Domain_Length_mm', 'Sulcus_Width_mm', 'Sulcus_Depth_mm']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns. Available: {list(df.columns)}")
        return

    mu_factors = sorted(df['Mu_Factor'].unique())
    first_row = df.iloc[0]
    L_dim, H_dim = first_row['Domain_Length_mm'], 1.0
    sulcus_w_dim = first_row['Sulcus_Width_mm']

    # Setup coordinates
    L_ref = H_dim
    sulcus_left_dim = (L_dim - sulcus_w_dim) / 2.0
    sulcus_right_dim = sulcus_left_dim + sulcus_w_dim

    # Convert to non-dimensional
    L = L_dim / L_ref
    sulcus_left_nondim = sulcus_left_dim / L_ref
    sulcus_right_nondim = sulcus_right_dim / L_ref
    sulcus_w_nondim = sulcus_w_dim / L_ref

    colors = ['#2E86AB', '#F18F01', '#F24236', '#A23B72', '#C73E1D']

    # --- Figure sizing ---
    fig_width = 6.5  # fixed total width in inches
    subplot_width = fig_width / len(mu_factors)
    fig_height = subplot_width + 1.2  # adding padding for title/x-labels

    out_path = os.path.join(plots_dir, 'mu_eff_spatial_distribution.png')
    with safe_plot(out_path, (fig_width, fig_height)):
        fig, axes = plt.subplots(1, len(mu_factors), figsize=(fig_width, fig_height),
                                 sharex=True, sharey=True)
        axes = np.atleast_2d(axes)

        # Move suptitle further up
        fig.suptitle(r'$\mu_{\mathrm{eff}}$ Spatial Distribution (Step and Simulation)',
                     fontsize=16, fontweight='bold', y=0.9)

        x_coords = np.linspace(0, L, 1000)

        for i, mu_factor in enumerate(mu_factors):
            row = df[df['Mu_Factor'] == mu_factor].iloc[0]

            mu_base_nondim = row.get('Mu_base_nondim', 1.0)
            mu_eff_sim = row.get('Mu_Eff_Simulation')
            mu_eff_open = row.get('Mu_Eff_Opening')

            step_values = None
            if mu_eff_open is not None:
                try:
                    from parameters import StepUptakeOpen
                    step_function = StepUptakeOpen(
                        mu_base=mu_base_nondim,
                        mu_eff_target=mu_eff_open,
                        sulcus_left_x=sulcus_left_nondim,
                        sulcus_right_x=sulcus_right_nondim,
                        L_c=0.1 * sulcus_w_nondim,
                        Gamma=5.0
                    )
                    vals = []
                    for x in x_coords:
                        v = [0.0]
                        step_function.eval(v, [x, 0.0])
                        vals.append(v[0])
                    step_values = np.array(vals)
                except ImportError:
                    print('StepUptakeOpen not available, using baseline mu only.')

            ax = axes[0, i]

            if step_values is not None:
                ax.plot(x_coords * L_ref, step_values, 'k-', linewidth=1, alpha=0.8,
                        label=r'Step function $\mu(x)_{\mathrm{eff}}^{\mathrm{step}}$')
            else:
                ax.plot(x_coords * L_ref, np.full_like(x_coords, mu_base_nondim),
                        'k--', alpha=0.6, linewidth=2, label=r'Baseline $\mu$')

            if mu_eff_sim is not None:
                label=r'Simulation $\mu_{\mathrm{eff}}^{\mathrm{sim}}$'
                ax.axhline(y=mu_eff_sim, color=colors[0], linestyle='-',
                           label=label, linewidth=2, alpha=0.8)

            label = 'Base $\mu$'
            ax.axhline(y=mu_base_nondim, color='#F24236', alpha=0.8, label=label, linewidth=1.5, linestyle=':')

            ax.axvspan(sulcus_left_dim, sulcus_right_dim, alpha=0.2, color='gray')

            try:
                ax.set_box_aspect(1)
            except AttributeError:
                # Fallback
                ax.set_aspect('auto')

            ax.grid(True, alpha=0.3)

            # Zoom into region of interest
            ax.set_xlim(4, 6)

            ax.text(0.5, 1.05, rf'$\mu$ factor = {mu_factor}x',
                    transform=ax.transAxes, fontweight='bold', fontsize=11,
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        axes[0, 0].set_ylabel(r'$\mu$ (non-dimensional)', fontweight='bold')
        for i in range(len(mu_factors)):
            axes[0, i].set_xlabel('x-coordinate (mm)', fontweight='bold')

        handles, labels = ax.get_legend_handles_labels()

        # Place one legend centered below all subplots
        fig.legend(handles, labels,
                loc='lower center',
                fontsize=11,
                ncol=len(labels),
                frameon=True,                # turn the frame back on
                fancybox=True,               # rounded corners
                framealpha=0.55,              # transparency
                facecolor='lightgray',       # grey background
                bbox_to_anchor=(0.5, -0.15)) # position

        plt.subplots_adjust(wspace=0.4)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # leave space for title

    print(f"Mu_eff spatial plot saved to {out_path}")

def generate_all_plots(csv_file, plot_type="mu"):
    """Dispatcher to generate plots from a CSV."""
    print(f"📊 Generating {plot_type} plots from: {os.path.basename(csv_file)}")
    if plot_type == 'mu':
        create_mu_sweep_plots(csv_file)
    elif plot_type == 'aspect_ratio':
        create_aspect_ratio_plots(csv_file)
    elif plot_type == 'geometry_analysis':
        create_geometry_analysis_plots(csv_file)
    else:
        print("ℹUnknown plot_type; supported types: 'mu', 'aspect_ratio', 'geometry_analysis'")

# endregion

# ========================================================
# region Study Runners
# ========================================================

def run_mu_sweep():
    """Run mu parameter sweep across three uptake regimes."""
    print("\n" + "="*60)
    print("MU PARAMETER SWEEP STUDY")
    print("="*60)

    # Get baseline parameters
    base_params = Parameters(mode='no-adv')
    base_params.sulci_w_dim = 0.05  # Fixed sulcus geometry
    base_params.sulci_h_dim = 0.05
    base_params.validate()
    base_params.nondim()

    # Get baseline mu_dim for scaling
    try:
        baseline_mu_dim = getattr(Parameters, 'MU_DIM_NO_ADV')
    except AttributeError:
        baseline_mu_dim = base_params.mu_dim

    # Define the three uptake regimes
    regime_configs = {
        'small_uptake': {
            'description': 'Small uptake regime - detailed sampling around baseline',
            'mu_factors': [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
            'color': 'lightblue'
        },
        'moderate_uptake': {
            'description': 'Moderate uptake regime - ~10x baseline',
            'mu_factors': [5.0, 7.5, 10.0, 12.5, 15.0],
            'color': 'orange'
        },
        'high_uptake': {
            'description': 'High uptake regime - ~100x baseline',
            'mu_factors': [50.0, 75.0, 100.0, 125.0, 150.0],
            'color': 'red'
        }
    }

    # Setup directories
    study_name = "Mu Parameter Sweep"
    study_dir, sim_dir = create_study_dirs(study_name, base_dir="Results/No Advection Simulations/Phase A")
    results = {}
    successful = 0
    total_runs = sum(len(regime_data['mu_factors']) for regime_data in regime_configs.values())

    print(f"Total simulations planned: {total_runs}")
    print(f"Baseline mu_dim: {baseline_mu_dim:.3e}")

    # Run simulations for each regime
    for regime_name, regime_data in regime_configs.items():
        print(f"\n{regime_name.upper()} REGIME:")
        print(f"  {regime_data['description']}")

        for factor in regime_data['mu_factors']:
            # Create parameters for this run
            params = Parameters(mode='no-adv')
            params.sulci_w_dim = 0.25
            params.sulci_h_dim = 0.25
            params.mu_dim = baseline_mu_dim * factor
            params.validate()
            params.nondim()

            config_name = f"{regime_name}_mu_{factor:.1f}x"
            print(f"  Running {config_name}: mu_dim={params.mu_dim:.3e}, mu={params.mu:.6g}")

            try:
                result = run_simulation(
                    mode='no-adv',
                    study_type=f"Phase A/{study_name} Simulations",
                    config_name=config_name,
                    domain_type='sulcus',
                    params=params
                )

                # Add metadata for analysis
                result.update({
                    'regime': regime_name,
                    'mu_factor': factor,
                    'regime_color': regime_data['color'],
                    'mu_dim_used': params.mu_dim,
                    'mu_used': params.mu,
                    'baseline_mu_dim': baseline_mu_dim
                })

                results[config_name] = result
                successful += 1
                print(f"    ✅ Success ({successful}/{total_runs})")

            except Exception as e:
                print(f"    ⚠ Failed: {e}")
                results[config_name] = None

    print(f"\n{'='*60}")
    print(f"MU SWEEP COMPLETED: {successful}/{total_runs} successful")
    print(f"{'='*60}")

    # Generate outputs
    if successful > 0:
        csv_path = create_mu_sweep_csv(results, study_dir, "mu_parameter_sweep")
        if csv_path:
            generate_all_plots(csv_path, "mu")

    return results

def run_aspect_ratio_analysis():
    """Run aspect ratio analysis focusing on mu_eff/mu ratio vs depth."""
    print("\n" + "="*60)
    print("ASPECT RATIO ANALYSIS")
    print("="*60)

    # Depth values for x-axis
    micro = np.logspace(np.log10(0.01), np.log10(0.10), 10)   # 10 pts in [0.01, 0.10] mm
    meso  = np.array([0.12, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00])
    macro = np.array([1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00])

    depths = sorted(set(np.round(np.concatenate([micro, meso, macro]), 4)))

    # Aspect ratios to analyse
    aspect_ratios = {
        'h_equals_w': 1.0,      # h = w
        'h_equals_2w': 2.0,     # h = 2w (narrow & deep)
        'h_equals_half_w': 0.5  # h = 0.5w (wide & shallow)
    }

    # Setup directories
    study_name = "Aspect Ratio Study"
    study_dir, sim_dir = create_study_dirs(study_name, base_dir="Results/No Advection Simulations/Phase A")
    results = {}
    successful = 0
    total_runs = 0

    print(f"Running aspect ratio sweeps...")

    # Aspect ratio sweeps only
    for ar_name, ar_value in aspect_ratios.items():
        print(f"\n  {ar_name.replace('_', ' ')} (AR = {ar_value}):")
        for h in depths:
            w = h / ar_value  # Calculate width from aspect ratio

            # Skip if width would exceed 1.0mm
            if w > 1.0:
                print(f"    h={h}mm skipped (would need w={w:.2f}mm > 1.0mm)")
                continue

            total_runs += 1
            config_name = f"{ar_name}_h{h}"

            try:
                # Create parameters
                params = Parameters(mode='no-adv')
                params.sulci_w_dim = w
                params.sulci_h_dim = h
                params.validate()
                params.nondim()

                # Run simulation
                result = run_simulation(
                    mode='no-adv',
                    study_type=f"Phase A/{study_name} Simulations",
                    config_name=config_name,
                    domain_type='sulcus',
                    params=params
                )

                # Add metadata for analysis
                result.update({
                    'aspect_ratio_metadata': {
                        'aspect_ratio_type': ar_name,
                        'width': w,
                        'depth': h,
                        'aspect_ratio': ar_value,
                        'label': ar_name.replace('_', ' ')
                    }
                })

                results[config_name] = result
                successful += 1

                # Extract mu_eff ratios for progress display
                if 'mu_eff_comparison' in result:
                    mu_eff_data = result['mu_eff_comparison']
                    def _fmt(x):
                        return f"{x:.3f}" if isinstance(x, (int, float)) else "N/A"
                    sim_ratio = mu_eff_data.get('mu_eff_ratios', {}).get('simulation_full_ratio')
                    ana_ratio = mu_eff_data.get('mu_eff_ratios', {}).get('analytical_ratio')
                    enh_ratio = mu_eff_data.get('mu_eff_ratios', {}).get('enhanced_ratio')
                    print(f"    h={h}mm, w={w:.2f}mm ✅ (sim={_fmt(sim_ratio)}, ana={_fmt(ana_ratio)}, enh={_fmt(enh_ratio)})")
                else:
                    print(f"    h={h}mm, w={w:.2f}mm ✅")

            except Exception as e:
                print(f"    h={h}mm, w={w:.2f}mm ⚠ ({str(e)[:50]}...)")
                results[config_name] = None

    print(f"\n{'='*60}")
    print(f"ASPECT RATIO ANALYSIS COMPLETE: {successful}/{total_runs} successful")
    print(f"{'='*60}")

    # Generate outputs
    if successful > 0:
        csv_path = create_aspect_ratio_csv(results, study_dir, "aspect_ratio_analysis")
        if csv_path:
            generate_all_plots(csv_path, "aspect_ratio")

    return results

def run_geometry_analysis(mu_factors=[0.1, 1.0, 10]):
    """
    Run geometry analysis across different geometries and mu factor values.
    Tests analytical mu_eff predictions against simulation results.
    """
    print("\n" + "="*60)
    print("GEOMETRY ANALYSIS STUDY")
    print("="*60)

    # Get baseline parameters and geometry variations
    base_params = Parameters(mode='no-adv')
    base_params.sulci_w_dim = 0.25  # Will be overridden by geometry variations
    base_params.sulci_h_dim = 0.25  # Will be overridden by geometry variations
    base_params.validate()
    base_params.nondim()

    # Get baseline mu_dim for scaling
    try:
        baseline_mu_dim = getattr(Parameters, 'MU_DIM_NO_ADV')
    except AttributeError:
        baseline_mu_dim = base_params.mu_dim

    # Get geometry variations
    geometries = create_geometry_variations(base_params)

    # Limit to a manageable subset for testing (removed this line for full analysis)
    # geometries = dict(list(geometries.items())[:10])  # First 10 geometries

    print(f"Running geometry analysis for {len(geometries)} geometries and {len(mu_factors)} mu factors:")
    print(f"Geometries: {list(geometries.keys())}")
    print(f"Mu factors: {mu_factors}")
    print(f"Baseline mu_dim: {baseline_mu_dim:.3e}")

    # Setup directories
    study_name = "Geometry Comparison"
    study_dir, sim_dir = create_study_dirs(study_name, base_dir="Results/No Advection Simulations/Phase A")

    all_simulation_results = {}
    successful_configs = 0
    total_configs = len(geometries) * len(mu_factors)
    config_count = 0

    # Run simulations for each geometry and mu factor
    for geo_name, geo_config in geometries.items():
        for mu_factor in mu_factors:
            config_count += 1
            config_name = f"{geo_name}_mu_{mu_factor}"

            print(f"\n{'-'*50}")
            print(f"CONFIG {config_count}/{total_configs}: {config_name}")
            print(f"  Geometry: {geo_config['name']}")
            print(f"  Mu factor: {mu_factor}x")
            print(f"{'-'*50}")

            config_results = {
                'geometry_name': geo_name,
                'mu_value': baseline_mu_dim * mu_factor,  # Dimensional mu value
                'mu_factor': mu_factor,
                'geometry_config': geo_config,
                'sulcus': None
            }

            try:
                # Create parameters for this run
                params = Parameters(mode='no-adv')
                params.sulci_w_dim = geo_config['sulci_w_dim']
                params.sulci_h_dim = geo_config['sulci_h_dim']
                params.mu_dim = baseline_mu_dim * mu_factor
                params.validate()
                params.nondim()

                print(f"  Running sulcus simulation...")
                print(f"    Dimensions: {params.sulci_w_dim}×{params.sulci_h_dim}mm")
                print(f"    mu_dim: {params.mu_dim:.3e}, mu: {params.mu:.6g}")

                result = run_simulation(
                    mode='no-adv',
                    study_type=f"Phase A/{study_name} Simulations",
                    config_name=config_name,
                    domain_type='sulcus',
                    params=params
                )

                config_results['sulcus'] = result
                successful_configs += 1

                # Extract and display mu_eff comparison if available
                if 'mu_eff_comparison' in result:
                    comp = result['mu_eff_comparison']
                    sim_val = comp.get('mu_eff_simulation_full')
                    ana_val = comp.get('mu_eff_analytical')

                    if sim_val and ana_val:
                        abs_error = abs(ana_val - sim_val)
                        rel_error = abs_error / sim_val * 100 if sim_val > 0 else float('inf')
                        print(f"    mu_eff_sim: {sim_val:.6g}, mu_eff_ana: {ana_val:.6g}")
                        print(f"    Abs error: {abs_error:.6g}, Rel error: {rel_error:.2f}%")
                    else:
                        print(f"    mu_eff values: sim={sim_val}, ana={ana_val}")

                print(f"    ✅ Success ({successful_configs}/{total_configs})")

            except Exception as e:
                print(f"    ⚠️ Failed: {e}")
                config_results['sulcus'] = None

            all_simulation_results[config_name] = config_results

    print(f"\n{'='*60}")
    print(f"GEOMETRY ANALYSIS COMPLETED: {successful_configs}/{total_configs} successful")
    print(f"{'='*60}")

    # Generate outputs
    if successful_configs > 0:
        csv_path = create_geometry_analysis_csv(all_simulation_results, study_dir, "geometry_analysis")
        if csv_path:
            generate_all_plots(csv_path, "geometry_analysis")

    return all_simulation_results

def run_mu_eff_analysis():
    """Run mu_eff spatial distribution analysis - streamlined version."""
    print("\n" + "="*60)
    print("MU_EFF SPATIAL ANALYSIS")
    print("="*60)

    # Fixed geometry: 0.5mm x 1.0mm sulcus
    sulcus_width = 0.5   # mm
    sulcus_depth = 1.0   # mm

    # Three mu values to test
    mu_factors = [0.1, 1.0, 10.0]

    # Get baseline parameters
    base_params = Parameters(mode='no-adv')
    base_params.sulci_w_dim = sulcus_width
    base_params.sulci_h_dim = sulcus_depth
    base_params.validate()
    base_params.nondim()

    # Get baseline mu_dim for scaling
    baseline_mu_dim = getattr(Parameters, 'MU_DIM_NO_ADV', 0.0003)

    # Setup directories
    study_name = "Mu_Eff Spatial Analysis"
    study_dir, sim_dir = create_study_dirs(study_name, base_dir="Results/No Advection Simulations/Phase A")
    results = {}
    successful = 0
    total_runs = len(mu_factors)

    print(f"Running mu_eff analysis for {total_runs} mu values:")
    print(f"Sulcus geometry: {sulcus_width}×{sulcus_depth}mm")
    print(f"Mu factors: {mu_factors}")
    print(f"Baseline mu_dim: {baseline_mu_dim:.3e}")

    # Run simulations for each mu factor
    for factor in mu_factors:
        # Create parameters for this run
        params = Parameters(mode='no-adv')
        params.sulci_w_dim = sulcus_width
        params.sulci_h_dim = sulcus_depth
        params.mu_dim = baseline_mu_dim * factor
        params.validate()
        params.nondim()

        config_name = f"mu_eff_analysis_mu_{factor}x"
        print(f"\nRunning {config_name}:")
        print(f"  mu_dim={params.mu_dim:.3e}, mu={params.mu:.6g}")

        try:
            result = run_simulation(
                mode='no-adv',
                study_type=f"Phase A/{study_name} Simulations",
                config_name=config_name,
                domain_type='sulcus',
                params=params
            )

            # Add metadata for analysis - minimal since params are already stored
            result.update({
                'mu_factor': factor,
                'mu_value': params.mu_dim,  # dimensional
                'mu_dim_used': params.mu_dim,
                'mu_used': params.mu,  # non-dimensional
                'baseline_mu_dim': baseline_mu_dim
            })

            results[config_name] = result
            successful += 1
            print(f"    Success ({successful}/{total_runs})")

            # Display mu_eff values if available
            if 'mu_eff_comparison' in result:
                comp = result['mu_eff_comparison']
                mu_sim = comp.get('mu_eff_sim', 'N/A')
                mu_ana = comp.get('mu_eff_arc', 'N/A')
                mu_enh = comp.get('mu_eff_enh', 'N/A')
                mu_open = comp.get('mu_eff_open', 'N/A')

                print(f"    Mu_eff values:")
                print(f"      Simulation: {mu_sim:.6f}" if mu_sim != 'N/A' else f"      Simulation: {mu_sim}")
                print(f"      Analytical: {mu_ana:.6f}" if mu_ana != 'N/A' else f"      Analytical: {mu_ana}")
                print(f"      Enhanced:   {mu_enh:.6f}" if mu_enh != 'N/A' else f"      Enhanced:   {mu_enh}")
                print(f"      Opening:    {mu_open:.6f}" if mu_open != 'N/A' else f"      Opening:    {mu_open}")

        except Exception as e:
            print(f"    Failed: {e}")
            results[config_name] = None

    print(f"\n{'='*60}")
    print(f"MU_EFF ANALYSIS COMPLETED: {successful}/{total_runs} successful")
    print(f"{'='*60}")

    # Generate outputs
    if successful > 0:
        csv_path = create_mu_eff_analysis_csv(results, study_dir, "mu_eff_analysis")
        if csv_path:
            create_mu_eff_spatial_plots(csv_path)

    return results

def replot_from_csv():
    """Find existing CSV files and regenerate plots."""
    print("🔍 Searching for existing CSV files...")

    # Search for CSV files in the Results directory
    csv_pattern = "Results/No Advection Simulations/Phase A/**/**.csv"
    csv_files = glob(csv_pattern, recursive=True)

    if not csv_files:
        print("⚠ No CSV files found in Results/No Advection Simulations/Phase A/")
        return

    print(f"🔍 Found {len(csv_files)} CSV files:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file}")

    # Try to determine plot type from filename and regenerate plots
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).lower()
        if 'mu_parameter' in filename or 'mu_sweep' in filename:
            print(f"📊 Regenerating mu sweep plots for: {csv_file}")
            generate_all_plots(csv_file, "mu")
        elif 'mu_eff' in filename:
            print(f"📊 Regenerating mu_eff spatial plots for: {csv_file}")
            create_mu_eff_spatial_plots(csv_file)
        elif 'aspect_ratio' in filename:
            print(f"📊 Regenerating aspect ratio plots for: {csv_file}")
            generate_all_plots(csv_file, "aspect_ratio")
        elif 'geometry_analysis' in filename:
            print(f"📊 Regenerating geometry analysis plots for: {csv_file}")
            generate_all_plots(csv_file, "geometry_analysis")
        else:
            print(f"❓ Unknown CSV type, trying mu plots: {csv_file}")
            generate_all_plots(csv_file, "mu")

#endregion

# ========================================================
# region Main Execution
# ========================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NO-ADVECTION ANALYSIS")
    print("="*60)

    print("\nOptions:")
    print("1. Mu parameter sweep study")
    print("2. Aspect ratio study")
    print("3. Geometry analysis study")
    print("4. Mu_eff spatial analysis")
    print("5. Run all studies")
    print("6. Replot from saved CSV")

    choice = input("\nSelect (1-6): ").strip()

    studies = {
        "1": run_mu_sweep,
        "2": run_aspect_ratio_analysis,
        "3": run_geometry_analysis,
        "4": run_mu_eff_analysis,
        "5": lambda: (run_mu_sweep(), run_aspect_ratio_analysis(), run_geometry_analysis(), run_mu_eff_analysis()),
        "6": replot_from_csv
    }

    # ADD THIS PART:
    if choice in studies:
        try:
            studies[choice]()
        except KeyboardInterrupt:
            print("\nStudy interrupted")
        except Exception as e:
            print(f"\nError: {e}")
    else:
        print("Invalid choice")

# endregion