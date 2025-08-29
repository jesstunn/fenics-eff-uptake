##########################################################
# No Uptake Analysis (mu = 0)
##########################################################
"""
Geometry study at multiple Peclet numbers with mu=0.
Computes concentration and velocity ratios, generates heatmaps.
"""

# Pe = U_ref_dim * H_dim) / D_dim
# Thus
# U_ref_dim = Pe * D_dim / H_dim

# ========================================================
# Imports
# ========================================================

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from dolfin import assemble, Constant

# Module imports
from parameters import Parameters, create_geometry_variations
from simulation import run_simulation
from analysis import compute_conc_profiles
from plotting import safe_plot, format_filename_value, create_study_dirs, latexify_label, Config

# Constants
MARKERS = {
    'left': 1, 'right': 2, 'top': 3, 'bottom': 4,
    'bottom_left': 5, 'sulcus': 6, 'bottom_right': 7, 'sulcus_opening': 8,
    'y0_line': 10
}

LATEX_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
}

# ========================================================
# Rectangular Baselines
# ========================================================

def extract_rectangular_data(result):
    """
    Extract data from rectangular baseline simulation using stored parameters.
    """
    params = result.get('params')
    if not params:
        return None

    # Get params number
    peclet = getattr(params, 'Pe', None)
    U_ref = getattr(params, 'U_ref', None)
    U_ref_dim = getattr(params, 'U_ref_dim', None)

    D_dim = getattr(params, 'U_ref_dim', None) * getattr(params, 'H_dim', None) / getattr(params, 'Pe', None)
    delta = D_dim / getattr(params, 'U_ref_dim', None)

    # Generate config name
    config_name = f'rect_Pe{peclet:.1f}'.replace('.', 'p')

    mm = result.get('mass_metrics', {})
    vm = result.get('vel_metrics', {})
    fm = result.get('flux_metrics', {})

    # Validation
    pf = fm.get('physical_flux', {}) if isinstance(fm, dict) else {}
    inlet_flux = pf.get('left', {}).get('total', 0) if isinstance(pf.get('left'), dict) else 0
    outlet_flux = pf.get('right', {}).get('total', 0) if isinstance(pf.get('right'), dict) else 0

    return {
        'Domain': 'rectangle',
        'Mode': getattr(params, 'mode', None),
        'Peclet': peclet,
        'U_ref': U_ref,
        'Sulcus Width (mm)': None,
        'Sulcus Depth (mm)': None,
        'Aspect_Ratio': None,
        'U_ref (Dim)': U_ref_dim,
        'Diff Coef (Dim)': D_dim,
        'Delta (mm)': delta,
        'Total Mass': mm.get('total_mass'),
        'Sulcus Mass': None,
        'Main Channel Mass': mm.get('total_mass', None),
        'Avg Concentration': mm.get('average_concentration', None),
        'Sulcus Avg Concentration': None,
        'Main Channel Avg Concentration': mm.get('average_concentration', None),
        'Mouth_Flux_Total': None,
        'Mouth E_L1': None,
        'Mouth E_avg': None,
        'Mouth Q_in': None,
        'Mouth Q_out': None,
        'Mouth Net Check': None,
        'Mouth Length': None,
        'Max_Ux_mid_channel': vm.get('max_ux_mid_channel') if isinstance(vm, dict) else None,
        'Avg_Ux_mid_channel': vm.get('avg_ux_mid_channel') if isinstance(vm, dict) else None,
        'Max_Ux_sulcus_level': None,  # N/A for rectangle
        'Avg_Ux_sulcus_level': None,  # N/A for rectangle
        'Inlet-Outlet Flux': inlet_flux + outlet_flux,
    }

def run_rectangular_baselines(peclet_numbers=[0.1, 1.0, 10.0]):
    """Run rectangular baseline simulations for each Peclet number."""

    rows = []
    for pe in peclet_numbers:
        try:
            params = Parameters(mode='no-uptake')
            params.mu_dim = 0.0

            # Pe = U_ref_dim * H_dim) / D_dim
            # Thus
            # U_ref_dim = Pe * D_dim / H_dim
            params.U_ref_dim = pe * params.D_dim / params.H_dim

            params.validate()
            params.nondim()
            params.D_dim = params.U_ref_dim * params.H_dim / params.Pe
            result = run_simulation(
                mode='no-uptake',
                study_type='Rectangular Baselines',
                config_name=f'rect_Pe{format_filename_value(pe)}',
                domain_type='rectangular',
                params=params
            )

            rows.append(extract_rectangular_data(result))
            print(f'  Rectangle baseline Pe={pe}: success')

        except Exception as e:
            print(f'  Rectangle baseline Pe={pe}: failed - {e}')

    return rows

# ========================================================
# Data Extraction and CSV Generation
# ========================================================

def extract_simulation_data(result):
    """
    Extract key data from simulation result using only the stored parameters.
    """
    params = result.get('params')
    if not params:
        return None

    # Extract parameters
    sulcus_width = getattr(params, 'sulci_w_dim', None)
    sulcus_depth = getattr(params, 'sulci_h_dim', None)
    mode = getattr(params, 'mode', None)
    peclet = getattr(params, 'Pe', None)
    U_ref = getattr(params, 'U_ref', None)
    U_ref_dim = getattr(params, 'U_ref_dim', None)
    D_dim = getattr(params, 'U_ref_dim', None) * getattr(params, 'H_dim', None) / getattr(params, 'Pe', None)
    delta = D_dim / getattr(params, 'U_ref_dim', None)

    # Calculate aspect ratio if both dimensions exist
    aspect_ratio = sulcus_depth / sulcus_width if (sulcus_width and sulcus_width > 0) else None

    # Generate config name from parameters
    config_name = f"w{sulcus_width:.2f}_h{sulcus_depth:.2f}_Pe{peclet:.1f}".replace('.', 'p')

    row = {
        'Domain': 'sulcus',
        'Mode': mode,
        'Peclet': peclet,
        'U_ref': U_ref,
        'Sulcus Width (mm)': sulcus_width,
        'Sulcus Depth (mm)': sulcus_depth,
        'Aspect_Ratio': aspect_ratio,
        'U_ref (Dim)': U_ref_dim,
        'Diff Coef (Dim)': D_dim,
        'Delta (mm)': delta
    }

    # Mass & concentration metrics
    mm = result.get('mass_metrics', {})
    if isinstance(mm, dict):
        avg_conc = mm.get('average_concentration', {})
        row.update({
            'Total Mass': mm.get('total_mass'),
            'Sulcus Mass': mm.get('sulcus_mass'),
            'Main Channel Mass': mm.get('rectangle_mass'),
            'Avg Concentration': avg_conc.get('total') if isinstance(avg_conc, dict) else avg_conc,
            'Sulcus Avg Concentration': avg_conc.get('sulcus_region') if isinstance(avg_conc, dict) else None,
            'Main Channel Avg Concentration': avg_conc.get('rectangle_region') if isinstance(avg_conc, dict) else None,
        })

    # Flux metrics
    fm = result.get('flux_metrics', {})
    if isinstance(fm, dict):
        # Mouth flux (signed)
        mouth = fm.get('sulcus_specific', {}).get('physical_flux', {}).get('sulcus_opening', {})
        row['Mouth_Flux_Total'] = mouth.get('total')

        # Validation checks
        pf = fm.get('physical_flux', {})
        inlet_flux = pf.get('left', {}).get('total', 0)
        outlet_flux = pf.get('right', {}).get('total', 0)
        row['Inlet-Outlet Flux'] = inlet_flux + outlet_flux

    # Sulcus mouth metrics
    mouth_metrics = (
        fm.get('sulcus_specific', {})
        .get('physical_flux', {})
        .get('sulcus_opening_extra', {})
    )

    if isinstance(mouth_metrics, dict):
        row.update({
            'Mouth E_L1':       mouth_metrics.get('E_L1', None),
            'Mouth E_avg':      mouth_metrics.get('E_avg', None),
            'Mouth Q_in':       mouth_metrics.get('Q_in', None),
            'Mouth Q_out':      mouth_metrics.get('Q_out', None),
            'Mouth Net Check':  mouth_metrics.get('net_check', None),
            'Mouth Length':     mouth_metrics.get('length', None),
        })

    # Velocity metrics
    vm = result.get('vel_metrics', {})
    if isinstance(vm, dict):
        row.update({
            'Max_Ux_mid_channel': vm.get('max_ux_mid_channel'),
            'Avg_Ux_mid_channel': vm.get('avg_ux_mid_channel'),
            'Max_Ux_sulcus_level': vm.get('max_ux_sulcus_level'),
            'Avg_Ux_sulcus_level': vm.get('avg_ux_sulcus_level'),
        })

    return row

def create_combined_csv(sulcus_results, peclet_numbers, study_dir):
    """Create combined CSV with sulcus and rectangular data."""

    # Extract sulcus data
    sulcus_rows = []
    for config_name, result in sulcus_results.items():
        if result:
            sulcus_rows.append(extract_simulation_data(result))

    # Run rectangular baselines
    rect_rows = run_rectangular_baselines(peclet_numbers)

    # Combine and save
    all_rows = sulcus_rows + rect_rows
    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(study_dir, 'geometry_comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'Combined CSV saved: {csv_path} ({len(all_rows)} rows)')

    return csv_path

def add_ratio_metrics(csv_path):
    """Add concentration and velocity ratios using rectangular baselines from same columns."""
    df = pd.read_csv(csv_path)

    # Get rectangular baseline values by Peclet number
    rect_baselines = df[df['Domain'] == 'rectangle'].groupby('Peclet').agg({
        'Avg Concentration': 'mean',
        'Max_Ux_mid_channel': 'mean',
        'Avg_Ux_mid_channel': 'mean'
    })

    # Initialise ratio columns with NaN
    ratio_cols = ['Concentration_Ratio', 'Channel_Conc_Ratio', 'Intradomain_Enrichment', 'VR_mid_avg',
                  'VR_mid_max', 'VR_intradomain_avg', 'VR_intradomain_max']
    for col in ratio_cols:
        df[col] = np.nan

    # Calculate ratios for sulcus rows only
    for pe in rect_baselines.index:

        # Create mask for sulcus rows with this Peclet number
        mask = (df['Domain'] == 'sulcus') & (df['Peclet'] == pe)

        if not mask.any():
            continue

        # Get baseline values
        rect_avg_conc = rect_baselines.loc[pe, 'Avg Concentration']
        rect_max_ux = rect_baselines.loc[pe, 'Max_Ux_mid_channel']
        rect_avg_ux = rect_baselines.loc[pe, 'Avg_Ux_mid_channel']

        # Calculate ratios directly on the masked rows
        df.loc[mask, 'Concentration_Ratio'] = df.loc[mask, 'Avg Concentration'] / rect_avg_conc
        df.loc[mask, 'Channel_Conc_Ratio'] = df.loc[mask, 'Main Channel Avg Concentration'] / rect_avg_conc
        df.loc[mask, 'VR_mid_avg'] = df.loc[mask, 'Avg_Ux_mid_channel'] / rect_avg_ux
        df.loc[mask, 'VR_mid_max'] = df.loc[mask, 'Max_Ux_mid_channel'] / rect_max_ux

        # Intradomain ratios (sulcus vs sulcus, no rectangle involved)
        df.loc[mask, 'Intradomain_Enrichment'] = (
            df.loc[mask, 'Sulcus Avg Concentration'] / df.loc[mask, 'Main Channel Avg Concentration']
        )
        df.loc[mask, 'VR_intradomain_avg'] = (
            df.loc[mask, 'Avg_Ux_sulcus_level'] / df.loc[mask, 'Avg_Ux_mid_channel']
        )
        df.loc[mask, 'VR_intradomain_max'] = (
            df.loc[mask, 'Max_Ux_sulcus_level'] / df.loc[mask, 'Max_Ux_mid_channel']
        )

    df.to_csv(csv_path, index=False)
    print(f'Ratios added to CSV: {csv_path}')

    return csv_path

def collect_profile_rows(result, geometry_key=None):
    """
    Turn result's profiles_full into a list of tidy rows (one row per sample point).
    """
    rows = []
    if not result:
        return rows

    params = result.get('params', {})
    profiles_full = ((result.get('mass_metrics') or {}).get('profiles_full') or {})
    meta = ((result.get('mass_metrics') or {}).get('profiles_meta') or {})

    domain = result.get('domain_type', 'unknown')
    config = result.get('config_name', result.get('geometry'))
    pe = getattr(params, 'Pe', None)

    x_rng = meta.get('x_range')
    y_rng = meta.get('y_range')
    n_points = meta.get('n_points')

    # horizontal
    for name, payload in (profiles_full.get('horizontal') or {}).items():
        y = float(payload['y'])
        xs = np.asarray(payload['x'])
        cs = np.asarray(payload['c'])
        for i, (xx, cc) in enumerate(zip(xs, cs)):
            rows.append({
                'Domain': domain,
                'Geometry': geometry_key or result.get('geometry'),
                'Config': config,
                'Peclet': pe,
                'LineType': 'horizontal',
                'LineName': name,
                'Index': i,
                'x': float(xx),
                'y': y,
                'c': float(cc),
                'n_points': n_points,
                'x_min': None if x_rng is None else float(x_rng[0]),
                'x_max': None if x_rng is None else float(x_rng[1]),
                'y_min': None if y_rng is None else float(y_rng[0]),
                'y_max': None if y_rng is None else float(y_rng[1]),
            })

    return rows

def export_profile_samples_csv_sulci(results, geom_keys, peclets, out_dir):
    """
    Export full profile samples for selected sulcus geometries across given Pe.
    Writes one CSV per geometry: Profiles/profiles_samples_<geometry>.csv
    """
    import os
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)

    for gkey in geom_keys:
        rows = []
        # find all runs for this geometry + requested Pe values
        for pe in peclets:
            run = next(
                (r for r in results.values()
                 if r and r.get('geometry') == gkey and r.get('peclet') == pe),
                None
            )
            if not run:
                print(f"[profiles-samples] missing {gkey} at Pe={pe} — skipping.")
                continue

            # ensure profiles are computed (in case not already)
            compute_conc_profiles(run)

            rows.extend(collect_profile_rows(run, geometry_key=gkey))

        if not rows:
            continue

        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, f"profiles_samples_{gkey}.csv")
        df.to_csv(path, index=False)
        print(f"[profiles-samples] wrote {path} ({len(df)} rows))")

def export_profile_stats_csv(results, geom_keys, peclets, out_dir):
    """
    Export the profile statistics stored in result['mass_metrics']['profiles'] for selected geometries across the given Peclet numbers.
    """
    os.makedirs(out_dir, exist_ok=True)

    for gkey in geom_keys:
        rows = []
        # find all runs for this geometry + requested Pe values
        for pe in peclets:
            run = next(
                (r for r in results.values()
                 if r and r.get('geometry') == gkey and r.get('peclet') == pe),
                None
            )
            if not run:
                print(f"[profiles] missing {gkey} at Pe={pe} — skipping.")
                continue

            mm = (run.get('mass_metrics') or {})
            profs = (mm.get('profiles') or {})
            # horizontal lines
            for name, st in (profs.get('horizontal') or {}).items():
                rows.append({
                    'Geometry': gkey, 'Peclet': pe,
                    'line_type': 'horizontal', 'name': name,
                    'x': None, 'y': st.get('y'),
                    'min_c': st.get('min_c'), 'max_c': st.get('max_c'),
                    'avg_c': st.get('avg_c'), 'n_samples': st.get('n_samples')
                })

        if not rows:
            continue

        dfp = pd.DataFrame(rows)
        path = os.path.join(out_dir, f"profiles_{gkey}.csv")
        dfp.to_csv(path, index=False)
        print(f"[profiles] wrote {path} ({len(dfp)} rows)")

# ========================================================
# Plotting Functions
# ========================================================

def choose_colormap(vals):
    """Pick colormap and colour limits based on sign of values."""
    vmin, vmax = float(vals.min()), float(vals.max())

    if vmin < 0 and vmax > 0:
        # Mixed sign -> diverging around 0
        limit = max(abs(vmin), abs(vmax))
        cmap = 'RdBu_r'
        return cmap, -limit, limit
    elif vmin >= 0:
        # All non-negative -> sequential white->red, anchored at 0
        cmap = 'Reds'
        return cmap, 0.0, vmax
    else:
        # All non-positive -> sequential white->blue, anchored at 0
        cmap = 'Blues_r'
        return cmap, vmin, 0.0

def create_heatmap(df, col_name, title, cbar_label, filename, plots_dir, show_deviation=False, reference_value=1.0, symmetric=False):
    """Create geometry heatmap for given column."""
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found")
        return

    peclets = sorted(df['Peclet'].dropna().unique())
    if not peclets:
        print("No Peclet values found")
        return

    values = pd.to_numeric(df[col_name], errors='coerce')

    #if show_deviation:
    #    plot_values = values - reference_value
    #    max_dev = abs(plot_values.dropna()).max()
    #    vmin, vmax = -max_dev, max_dev
    #else:
    #    valid_vals = values.dropna()
    #    if symmetric or (valid_vals.min() < 0 < valid_vals.max()):
    #        limit = max(abs(valid_vals.min()), abs(valid_vals.max()))
    #        vmin, vmax = -limit, limit
    #    else:
    #        vmin, vmax = float(valid_vals.min()), float(valid_vals.max())

    if show_deviation:
        plot_values = values - reference_value
    else:
        plot_values = values

    valid_vals = plot_values.dropna()
    cmap, vmin, vmax = choose_colormap(valid_vals)

    # Figure sizing
    n_cols = len(peclets)
    panel_w = max(2.2, 6.7 / max(1, n_cols))
    fig_width = min(6.7, panel_w * n_cols)
    fig_height = 0.65 * panel_w + 0.7

    with mpl.rc_context(LATEX_RC):
        with safe_plot(os.path.join(plots_dir, filename), (fig_width, fig_height)):
            fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height))
            if n_cols == 1:
                axes = [axes]

            fig.suptitle(latexify_label(title), fontsize=11, fontweight='bold', y=1.08)
            plt.subplots_adjust(bottom=0.32, wspace=0.25)

            scatter = None
            for i, pe in enumerate(peclets):
                ax = axes[i]
                valid = df[df['Peclet'] == pe].dropna(subset=['Sulcus Width (mm)', 'Sulcus Depth (mm)', col_name])

                if valid.empty:
                    ax.text(0.5, 0.5, f'No data\n(Pe={pe})', ha='center', va='center',
                           transform=ax.transAxes, fontsize=9)
                    continue

                if show_deviation:
                    plot_vals = pd.to_numeric(valid[col_name], errors='coerce') - reference_value
                else:
                    plot_vals = pd.to_numeric(valid[col_name], errors='coerce')

                scatter = ax.scatter(
                    valid['Sulcus Width (mm)'], valid['Sulcus Depth (mm)'],
                    c=plot_vals, s=30, alpha=0.9, cmap=cmap,
                    vmin=vmin, vmax=vmax, marker='o',
                    edgecolors='black', linewidth=0.5
                )

                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)

                ax.set_title(f'Pe = {pe}', fontsize=9, fontweight='bold')
                ax.set_xlabel('Sulcus Width (mm)', fontsize=9, fontweight='bold')
                if i == 0:
                    ax.set_ylabel('Sulcus Depth (mm)', fontsize=9, fontweight='bold')
                ax.tick_params(axis='both', labelsize=9)
                ax.grid(True, alpha=0.3)

            # Colorbar
            if scatter:
                cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.05])
                cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
                cbar.set_label(latexify_label(cbar_label), fontsize=9, fontweight='bold')
                cbar.outline.set_linewidth(0.5)
                cbar.ax.tick_params(labelsize=9)

                if show_deviation:
                    cbar.ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

    print(f"Created: {filename}")

def create_velocity_heatmaps(df, plots_dir):
    """Create velocity ratio heatmaps at Pe=1.0, one .png per metric."""
    df_pe1 = df[df['Peclet'] == 1.0]
    if df_pe1.empty:
        print("No Pe=1.0 data for velocity plots")
        return

    sulcus_data = df_pe1[df_pe1['Domain'] == 'sulcus']
    if sulcus_data.empty:
        print("No sulcus data for Pe=1.0")
        return

    # Each metric is now plotted individually
    metrics_info = {
        'VR_mid_avg': 'Flow Obstruction: Midline (Average Velocity)',
        'VR_mid_max': 'Flow Obstruction: Midline (Maximum Velocity)',
        'VR_intradomain_avg': 'Intradomain Flow: Sulcus vs Midline (Average Velocity)',
        'VR_intradomain_max': 'Intradomain Flow: Sulcus vs Midline (Maximum Velocity)'
    }

    for metric, title in metrics_info.items():
        if metric not in sulcus_data.columns:
            continue

        with mpl.rc_context(LATEX_RC):
            filename = f"{metric}.png"
            out_path = os.path.join(plots_dir, filename)

        with safe_plot(out_path, (3.2, 2.8)):   # smaller canvas
            fig, ax = plt.subplots(figsize=(3.2, 2.8))
            fig.suptitle(title, fontsize=11, fontweight='bold', y=0.97)

            vals = pd.to_numeric(sulcus_data[metric], errors='coerce') - 1.0
            if vals.dropna().empty:
                continue
            cmap, vmin, vmax = choose_colormap(vals)

            scatter = ax.scatter(
                sulcus_data['Sulcus Width (mm)'], sulcus_data['Sulcus Depth (mm)'],
                c=vals, s=50, alpha=0.9, cmap=cmap,
                vmin=vmin, vmax=vmax, marker='o',
                edgecolors='black', linewidth=0.5
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.set_xlabel('Sulcus Width (mm)', fontsize=9, fontweight='bold')
            ax.set_ylabel('Sulcus Depth (mm)', fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Colorbar
            cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
            cbar.set_label(r'$\Delta$ Velocity Ratio (Ratio - 1)', fontsize=9, fontweight='bold')
            cbar.outline.set_linewidth(0.5)
            cbar.ax.tick_params(labelsize=9)


        print(f"Created: {filename}")

def generate_all_plots(csv_file):
    """Generate all plots from CSV data."""
    print(f"Generating plots from: {os.path.basename(csv_file)}")

    df = pd.read_csv(csv_file)
    plots_dir = os.path.join(os.path.dirname(csv_file), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Heatmap configurations
    heatmaps = [
        ('Mouth_Flux_Total', 'Sulcus Mouth Exchange Flux', 'Exchange Flux',
         'mouth_flux.png', False, 1.0, True),
        ('Concentration_Ratio', 'Concentration Ratio Deviation', r'$\Delta$ Concentration Ratio (Ratio - 1)',
         'concentration_ratio.png', True, 1.0, False),
        ('Intradomain_Enrichment', 'Intradomain Enrichment Ratio Deviaiton', r'$\Delta$ Intradomain Enrichment Ratio (Ratio - 1)',
         'intradomain_enrichment.png', True, 1.0, False),
        ('Channel_Conc_Ratio', 'Channel Concentration Ratio Deviation', r'$\Delta$ Channel Concentration Ratio (Ratio - 1)',
         'channel_concentration_ratio.png', False, 1.0, False),  # FIXED: Added missing comma here

        # Exchange strength metrics (no deviation from 1; just magnitude)
        ('Mouth E_L1',  'Mouth Exchange Strength', r'$\int_{\Sigma}|J\!\cdot\!n|\,ds$',
         'mouth_exchange_L1.png', False, 1.0, False),
        ('Mouth E_avg', 'Mouth Exchange (Average)', r'$\langle |J\!\cdot\!n| \rangle_{\Sigma}$',
         'mouth_exchange_Eavg.png', False, 1.0, False)
    ]

    for col, title, cbar_label, filename, show_dev, ref_val, symmetric in heatmaps:
        if col in df.columns:
            create_heatmap(df, col, title, cbar_label, filename, plots_dir,
                          show_deviation=show_dev, reference_value=ref_val, symmetric=symmetric)

    # Velocity plots
    create_velocity_heatmaps(df, plots_dir)

def plot_profiles_grid_from_samples_csv(csv_path, *, line_type='horizontal', profile_names=None, out_dir=None, filename_prefix=None):
    """
    Read a profiles_samples_*.csv (one geometry per file) and produce a grid:
      columns = Peclet numbers, rows = selected profile names (in given order),
      plotting c vs x (horizontal).
    """
    # --- helpers -------------------------------------------------------------
    def format_label_for_display(label):
        mapping = {
            'sulcus_mid': 'Sulcus Mid',
            'mouth_level': 'Mouth Level',
            'mid_channel': 'Mid Channel',
            'lower_channel': 'Lower Channel',
            'upper_channel': 'Upper Channel',
            'sulcus_level': 'Sulcus Level',
            'x_quarter': 'x = 1/4',
            'x_mid': 'x = 1/2',
            'x_three_quarters': 'x = 3/4',
        }
        return mapping.get(label, label.replace('_', ' ').title())

    def get_geometry_dimensions(geometry_key):
        geometry_dims = {
            'very_wide_tiny': (1.0, 0.2),
            'very_wide_medium': (1.0, 0.3),
            'very_wide_large': (1.0, 0.5),
            'mod_wide_small': (0.5, 0.3),
            'mod_wide_medium': (0.8, 0.6),
            'mod_wide_large': (1.0, 0.9),
            'square_small': (0.2, 0.2),
            'square_medium': (0.5, 0.5),
            'square_large': (0.7, 0.7),
            'mod_deep_small': (0.5, 0.8),
            'reference': (0.5, 1.0),
            'mod_deep_large': (1.0, 1.5),
            'deep_small': (0.3, 1.0),
            'deep_medium': (0.5, 1.5),
            'deep_large': (0.4, 2.0),
            'very_deep_small': (0.25, 1.5),
            'very_deep_large': (0.15, 1.8),
            'very_deep_extreme': (0.1, 2.0),
            'micro_depth_wide': (1.0, 0.05),
            'micro_width_deep': (0.05, 1.0),
            'largest': (1.0, 2.0),
            'micro_square': (0.01, 0.01),
            'macro_square': (1.0, 1.0),
            'small_sq_030': (0.03, 0.03),
            'small_sq_050': (0.05, 0.05),
            'small_sq_080': (0.08, 0.08),
            'small_sq_100': (0.10, 0.10),
            'small_wide_100x050': (0.10, 0.05),
            'small_deep_050x100': (0.05, 0.10),
        }
        return geometry_dims.get(geometry_key, None)

    def round_bounds_1dp(vmin, vmax, pad_if_equal=0.1):
        lo = np.floor(vmin * 10.0) / 10.0
        hi = np.ceil(vmax * 10.0) / 10.0
        if lo == hi:
            lo -= pad_if_equal
            hi += pad_if_equal
        return lo, hi

    # --- quick inline helper: run rectangular baseline and compute profiles ---
    rect_profiles_by_pe = {}

    def _run_rect_and_get_profiles(pe):
        """
        Run rectangle (mu=0) at this Pe and compute horizontal profiles.
        Returns {'lower_channel': (x,c), 'mid_channel': (x,c), 'mouth_level': (x,c)} if available.
        """
        if pe in rect_profiles_by_pe:
            return rect_profiles_by_pe[pe]

        # Lazy imports to avoid circulars
        params = Parameters(mode='no-uptake')
        params.mu_dim = 0.0
        # Pe = (U_ref_dim * H_dim) / D_dim  =>  U_ref_dim = Pe * D_dim / H_dim
        params.U_ref_dim = pe * params.D_dim / params.H_dim
        params.validate()
        params.nondim()
        # keep Pe consistent
        params.D_dim = params.U_ref_dim * params.H_dim / params.Pe

        result = run_simulation(
            mode='no-uptake',
            study_type='Rectangular Baseline (Profiles)',
            config_name=f'rect_Pe{str(pe).replace(".", "p")}',
            domain_type='rectangular',
            params=params
        )

        # populate result['mass_metrics']['profiles_full']['horizontal']
        compute_conc_profiles(result)

        out = {}
        profs = ((result.get('mass_metrics') or {}).get('profiles_full') or {}).get('horizontal', {})
        for name in ('lower_channel', 'mouth_level'):
            if name in profs:
                d = profs[name]
                out[name] = (np.asarray(d['x']), np.asarray(d['c']))
        rect_profiles_by_pe[pe] = out
        return out

    # --- load & prepare ------------------------------------------------------
    df = pd.read_csv(csv_path)
    required = {'LineType', 'LineName', 'Peclet', 'Index', 'x', 'c'}
    if not required.issubset(df.columns):
        print(f"[profiles-plot-csv] Missing columns in {csv_path}: {sorted(required - set(df.columns))}")
        return

    geometry = None
    if 'Geometry' in df.columns and df['Geometry'].notna().any():
        geometry = str(df['Geometry'].dropna().iloc[0])
    if geometry is None or geometry == '' or str(geometry).lower() == 'nan':
        geometry = os.path.splitext(os.path.basename(csv_path))[0].replace('profiles_samples_', '')

    dfg = df[df['LineType'] == line_type].copy()
    if dfg.empty:
        print(f"[profiles-plot-csv] No rows for line_type='{line_type}' in {csv_path}")
        return

    peclets = sorted(p for p in dfg['Peclet'].unique() if not np.isnan(p))
    if not peclets:
        print(f"[profiles-plot-csv] No Peclet values found in {csv_path}")
        return

    available = list(dfg['LineName'].unique())
    rows = [n for n in (profile_names or available) if n in available]
    if not rows:
        rows = available

    n_rows, n_cols = len(rows), len(peclets)

    # --- layout --------------------------------------------------------------
    panel_w = max(2.0, 6.7 / max(1, n_cols))
    fig_width = min(6.7, panel_w * n_cols)
    panel_h = 1.6
    fig_height = max(2.5, n_rows * panel_h + 0.8)

    dims = get_geometry_dimensions(geometry)
    if dims:
        width, depth = dims
        title = f"Horizontal Concentration Profiles: Domain with Sulcus of Width {width:.2f} mm x Depth {depth:.2f} mm"
    else:
        width, depth = None, None
        title = f"{latexify_label(geometry)} - Horizontal Concentration Profiles"

    fname_base = filename_prefix or "profiles"
    out_dir = out_dir or os.path.join(os.path.dirname(csv_path), "Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fname_base}_{geometry}_{line_type}.png")

    # --- plot ---------------------------------------------------------------
    with mpl.rc_context(LATEX_RC):
        with safe_plot(out_path, (fig_width, fig_height)):

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

            # Format all y-axes to show 1 decimal place
            for ax_row in axes:
                for ax in ax_row:
                    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

            # increased wspace/hspace
            plt.subplots_adjust(top=0.88, bottom=0.16,
                                left=0.22, right=0.98,
                                wspace=0.35, hspace=0.55)

            fig.suptitle(latexify_label(title), fontsize=11, fontweight='bold', y=0.98, x=0.6)

            for ci, pe in enumerate(peclets):
                df_pe = dfg[dfg['Peclet'] == pe]

                for ri, name in enumerate(rows):
                    ax = axes[ri, ci]
                    df_cell = df_pe[df_pe['LineName'] == name].sort_values('Index')
                    if df_cell.empty:
                        ax.text(0.5, 0.5, f"Missing '{name}'", ha='center', va='center',
                                transform=ax.transAxes, fontsize=9)
                        ax.set_xticks([]); ax.set_yticks([])
                        continue

                    if line_type == 'horizontal':
                        xs = df_cell['x'].to_numpy(float)
                        cs = df_cell['c'].to_numpy(float)
                        # Sulcus curve
                        ax.plot(xs, cs, lw=1.8, label='Sulcus')

                        # Rectangular overlay for the two requested rows
                        if name in ('lower_channel','mouth_level'):
                            try:
                                rect_profs = _run_rect_and_get_profiles(pe)
                                if name in rect_profs:
                                    xr, cr = rect_profs[name]
                                    ax.plot(xr, cr, linestyle=':', color='red', lw=1.8, label='Rectangle', alpha=0.6)
                            except Exception as e:
                                print(f"[rect overlay] failed for Pe={pe}, row={name}: {e}")

                        # axis ranges
                        if name == 'sulcus_mid':
                            ax.set_xlim(xs.min(), xs.max())
                            ax.set_ylim(0.5, 1.1)
                        elif name == 'mouth_level':
                            if width is not None:
                                ax.set_xlim(0.5 - width/2.0, 0.5 + width/2.0)
                            ax.set_ylim(0, 1.1)
                        else:
                            ax.set_xlim(0, 10)
                            ax.set_ylim(0, 1.1)

                        # labels
                        if ri == n_rows - 1:
                            ax.set_xlabel(latexify_label("$x$-coordinate (mm)"), fontsize=11, fontweight='bold')
                        if ci == 0:
                            ax.set_ylabel(latexify_label("Concentration"), fontsize=11, fontweight='bold')

                    ax.grid(True, alpha=0.3)
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.5)

            # Column headers (Pe)
            for ci, pe in enumerate(peclets):
                ax_top = axes[0, ci]
                bb = ax_top.get_position()
                x_mid = 0.5 * (bb.x0 + bb.x1)
                y_top = bb.y1 + 0.018
                fig.text(x_mid, y_top, f"Pe = {pe}",
                         ha='center', va='bottom',
                         fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.15',
                                   facecolor='#f2f2f7',
                                   edgecolor='#cfcfd4'))

            # Row headers
            for ri, name in enumerate(rows):
                label = format_label_for_display(name)

                df_name = dfg[dfg['LineName'] == name]
                y_val = None
                if not df_name.empty and 'y' in df_name.columns:
                    y_val = float(df_name['y'].iloc[0])  # or .mean()
                if y_val is not None:
                    label = f"{label}\n$y$ = {y_val:.2f} mm"

                ax_left = axes[ri, 0]
                bb = ax_left.get_position()
                x_left = bb.x0 - 0.12
                y_mid  = 0.5 * (bb.y0 + bb.y1)
                fig.text(x_left, y_mid, latexify_label(label),
                         rotation=90, ha='center', va='center',
                         fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.15',
                                   facecolor='#f2f2f7',
                                   edgecolor='#cfcfd4'))

            # One global legend if any overlays exist
            handles_global, labels_global = axes[0, 0].get_legend_handles_labels()
            if any(lbl in labels_global for lbl in ('Rectangle', 'Sulcus')):
                fig.legend(
                    handles_global,
                    labels_global,
                    loc='lower center',
                    ncol=len(labels_global),
                    fontsize=Config.FONT_SIZE_LEGEND + 1,
                    frameon=True,
                    fancybox=True,
                    framealpha=0.55,
                    facecolor='lightgray',
                    bbox_to_anchor=(0.6, -0.04)
                )

    print(f"[profiles-plot-csv] Saved: {out_path}")

# ========================================================
# Main Study Functions
# ========================================================

def run_geometry_study(peclet_numbers=[0.1, 1.0, 10.0]):
    """Run the complete geometry comparison study."""
    print(f"\nGEOMETRY COMPARISON STUDY (mu=0)")
    print(f"{'='*50}")

    # Setup
    base_params = Parameters(mode='no-uptake')
    configs = create_geometry_variations(base_params, max_width=1.0)
    study_dir, _ = create_study_dirs('Geometry Comparison', "Results/No Uptake Simulations")

    print(f"Configurations: {len(configs)}")
    print(f"Peclet numbers: {peclet_numbers}")
    print(f"Total simulations: {len(configs) * len(peclet_numbers)}")

    # Run sulcus simulations
    results = {}
    successful = 0

    for config_key, config in configs.items():
        for pe in peclet_numbers:
            config_name = f"{config_key}_Pe{format_filename_value(pe)}"

            try:
                params = Parameters(mode='no-uptake')
                params.sulci_w_dim = config['sulci_w_dim']
                params.sulci_h_dim = config['sulci_h_dim']
                params.U_ref_dim = pe * params.D_dim / params.H_dim
                params.validate()
                params.nondim()
                params.D_dim = params.U_ref_dim * params.H_dim / params.Pe

                result = run_simulation(
                    mode='no-uptake',
                    study_type="Geometry Comparison",
                    config_name=config_name,
                    domain_type='sulcus',
                    params=params
                )

                # Add metadata
                result.update({
                    'geometry': config_key,
                    'peclet': pe,
                    'width': config['sulci_w_dim'],
                    'depth': config['sulci_h_dim'],
                    'aspect_ratio': config.get('aspect_ratio')
                })

                results[config_name] = result
                successful += 1
                print(f'  {config_name}: success')

            except Exception as e:
                print(f'  {config_name}: failed - {e}')
                results[config_name] = None

    # Generate CSV and plots
    if successful > 0:
        csv_path = create_combined_csv(results, peclet_numbers, study_dir)
        if csv_path:
            csv_path = add_ratio_metrics(csv_path)
            generate_all_plots(csv_path)

    # ---- SELECTIVE profile computation and export ----
    print(f"\nComputing profiles for selected geometries...")
    profiles_dir = os.path.join("Results", "No Uptake Simulations", "Geometry Comparison Analysis", "Profiles")
    os.makedirs(profiles_dir, exist_ok=True)

    # Choose sulcus geometries by their existing config keys
    sulcus_geom_keys = ['largest', 'square_small']

    # *** COMPUTE PROFILES ONLY FOR SELECTED GEOMETRIES ***
    for gkey in sulcus_geom_keys:
        for pe in peclet_numbers:
            # Find the result for this geometry and Peclet number
            matching_result = next(
                (result for result in results.values()
                 if result and result.get('geometry') == gkey and result.get('peclet') == pe),
                None
            )

            if matching_result:
                print(f'  Computing profiles for {gkey} at Pe={pe}...')
                try:
                    compute_conc_profiles(matching_result)
                    print(f'  Profiles computed for {gkey} at Pe={pe}')
                except Exception as e:
                    print(f'  Failed to compute profiles for {gkey} at Pe={pe}: {e}')

    # Export profile samples CSV
    export_profile_samples_csv_sulci(results, sulcus_geom_keys, peclet_numbers, out_dir=profiles_dir)

    # ---- Generate per-geometry profile figures FROM CSVs ----
    print(f"Generating profile plots...")
    profiles_fig_dir = os.path.join(profiles_dir, "Plots")
    os.makedirs(profiles_fig_dir, exist_ok=True)

    # Horizontal rows
    horiz_rows = ['lower_channel', 'mouth_level', 'sulcus_mid']

    for gkey in sulcus_geom_keys:
        csv_path = os.path.join(profiles_dir, f"profiles_samples_{gkey}.csv")
        if not os.path.exists(csv_path):
            print(f"[profiles-plot] Missing CSV for {gkey}: {csv_path}")
            continue

        # Horizontal (c vs x)
        plot_profiles_grid_from_samples_csv(
            csv_path, line_type='horizontal',
            profile_names=horiz_rows, out_dir=profiles_fig_dir, filename_prefix='profiles'
        )

    print(f"\nStudy completed: {successful}/{len(configs) * len(peclet_numbers)} successful")

    return results#

def replot_from_csv():
    """Replot from saved CSVs.
       - If a geometry_comparison_results.csv is chosen: (re)compute ratios if needed and draw heatmaps/velocity plots.
       - If a profiles_samples_*.csv is chosen: draw per-geometry profile grids from the samples.
    """
    csv_files = glob("Results/No Uptake Simulations/**/*.csv", recursive=True)
    if not csv_files:
        print("No CSV files found!")
        return

    # Put the summary CSVs first, then profile-sample CSVs
    csv_files_sorted = sorted(csv_files, key=lambda p: (0 if 'geometry_comparison_results.csv' in os.path.basename(p) else 1, p))

    print(f"Found {len(csv_files_sorted)} CSV files:")
    for i, csv_file in enumerate(csv_files_sorted, 1):
        print(f"{i}. {os.path.relpath(csv_file)}")

    try:
        choice = int(input(f"\nSelect file (1-{len(csv_files_sorted)}): ").strip()) - 1
        if not (0 <= choice < len(csv_files_sorted)):
            print("Invalid selection")
            return
        selected_csv = csv_files_sorted[choice]
        basename = os.path.basename(selected_csv)

        # Case A: master geometry comparison CSV
        if basename == 'geometry_comparison_results.csv':
            df = pd.read_csv(selected_csv)
            needed_cols = ['Concentration_Ratio', 'VR_mid_avg', 'VR_mid_max']
            if not all(col in df.columns for col in needed_cols):
                print("Computing missing ratio metrics...")
                selected_csv = add_ratio_metrics(selected_csv)
            generate_all_plots(selected_csv)
            return

        # Case B: per-geometry profiles samples CSV
        if basename.startswith('profiles_samples_'):

            # Output directory next to the CSV
            out_dir = os.path.join(os.path.dirname(selected_csv), "Plots")

            # Horizontal (c vs x)
            order = ['lower_channel', 'mouth_level', 'sulcus_mid']
            plot_profiles_grid_from_samples_csv(selected_csv, line_type='horizontal', profile_names=order, out_dir=out_dir, filename_prefix='Profiles')

            return

        # Fallback: try master plotting anyway
        print("File not recognised as master or profiles-samples. Trying master plotting...")
        generate_all_plots(selected_csv)

    except (ValueError, KeyboardInterrupt):
        print("Invalid input or cancelled")

# ========================================================
# Main
# ========================================================

if __name__ == "__main__":
    print("\nNO-UPTAKE ANALYSIS")
    print("="*40)
    print("\nOptions:")
    print("1. Run geometry comparison study")
    print("2. Replot from saved CSV")

    choice = input("\nSelect (1-2): ").strip()

    if choice == "1":
        try:
            run_geometry_study()
        except KeyboardInterrupt:
            print("\nStudy interrupted")
        except Exception as e:
            print(f"\nError: {e}")
    elif choice == "2":
        replot_from_csv()
    else:
        print("Invalid choice")