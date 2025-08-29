##########################################################
# Plotting Module
##########################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import seaborn as sns
import pandas as pd
from datetime import datetime
from contextlib import contextmanager
from dolfin import *

# ========================================================
# Configuration and Utilities
# ========================================================

class Config:
    DPI = 300
    FIGSIZE = (10, 6)
    COLOURS = ['#2E86AB', '#F24236', '#A23B72', '#F18F01', '#C73E1D', '#92140C']
    FONT_FAMILY = "serif"
    FONT_SIZE_TITLE = 14     # Plot titles
    FONT_SIZE_LABEL = 12     # Axis labels
    FONT_SIZE_TICK  = 11     # Tick labels
    FONT_SIZE_LEGEND = 11    # Legend text
    FONT_SIZE_VALUE = 9      # Bar value labels
    ALPHA = 0.8              # Transparency (0 transparent, 1 opaque)
    LINEWIDTH = 0.8          # Bar edge width
    LABEL_PADDING = 0.15     # Extra top padding (fraction of y-range)

def set_style(font_overrides=None):
    """
    LaTeX + Computer Modern serif everywhere.
    Optional per-figure overrides via font_overrides:
      {'title': 16, 'label': 14, 'tick': 12, 'legend': 12}
    """
    matplotlib.rcParams.update({
        "text.usetex": True,                               # use LaTeX for all text
        "font.family": "serif",                            # serif family
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "figure.dpi": Config.DPI,
        "savefig.dpi": Config.DPI,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    # Base sizes from Config, overridden per figure if requested
    sizes = {
        "title":  Config.FONT_SIZE_TITLE,
        "label":  Config.FONT_SIZE_LABEL,
        "tick":   Config.FONT_SIZE_TICK,
        "legend": Config.FONT_SIZE_LEGEND,
    }
    if isinstance(font_overrides, dict):
        sizes.update({k: v for k, v in font_overrides.items() if k in sizes})

    # Apply sizes into rc so seaborn inherits them
    matplotlib.rcParams.update({
        "axes.titlesize":   sizes["title"],
        "axes.labelsize":   sizes["label"],
        "xtick.labelsize":  sizes["tick"],
        "ytick.labelsize":  sizes["tick"],
        "legend.fontsize":  sizes["legend"],
    })

    import seaborn as sns
    sns.set_context("paper", rc={
        "lines.linewidth": 1.0,
        "axes.labelsize":  sizes["label"],
        "xtick.labelsize": sizes["tick"],
        "ytick.labelsize": sizes["tick"],
        "legend.fontsize": sizes["legend"],
    })
    sns.set_style("darkgrid", {
        "axes.edgecolor": "0.15",
        "axes.linewidth": Config.LINEWIDTH,
        "grid.linewidth": 0.5,
        "grid.color": "0.8",
        "grid.alpha": 0.6,
    })
    sns.set_palette(Config.COLOURS)

@contextmanager
def safe_plot(filename, figsize=Config.FIGSIZE, *, font_overrides=None, rc=None):
    """
    Safe plotting context.
    Saves both PDF and PNG versions automatically.
    """
    set_style(font_overrides=font_overrides)
    if isinstance(rc, dict):
        rcParams.update(rc)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig = plt.figure(figsize=figsize)
    try:
        yield fig

        # Base name without extension
        base, _ = os.path.splitext(filename)

        # Always save PNG
        plt.savefig(base + ".png", dpi=Config.DPI, bbox_inches='tight', facecolor='white')
        # Always save PDF (vector)
        plt.savefig(base + ".pdf", bbox_inches='tight', facecolor='white')

        print(f"✓ Saved: {os.path.basename(base)}.png and .pdf")
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")
    finally:
        plt.close(fig)

def format_bar_label(value: float, mode: str = "dual") -> str:
    """Format a bar label"""
    if abs(value) < 1e-12:
        return "0" if mode != "dual" else "0\n(0.0e+00)"

    if mode == "simple":
        return f"{value:.3f}"

    if mode == "sci":
        return f"{value:.2e}"

    # Default: dual
    if abs(value) >= 1e6 or abs(value) < 1e-3:
        dec_str = f"{value:.3g}"
    else:
        dec_str = f"{value:.6f}".rstrip("0").rstrip(".")

    sci_str = f"({value:.2e})"
    return f"{dec_str}\n{sci_str}"

def add_value_labels(ax: plt.Axes, label_mode: str = "dual", rotation: float = 0) -> None:
    """Add value labels above bars with optional rotation."""
    labels_info = []

    # Iterate through each Rectangle in each container
    for container in ax.containers:
        for bar in container:
            if not hasattr(bar, 'get_height'):
                continue
            height = bar.get_height()
            if height == 0 or np.isnan(height):
                continue

            # Bar centre (x) and offset slightly above bar (y)
            x = bar.get_x() + bar.get_width() / 2.0
            y_offset = height + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])

            text_obj = ax.text(
                x, y_offset, format_bar_label(height, label_mode),
                ha='center', va='bottom', rotation=rotation,
                fontsize=Config.FONT_SIZE_VALUE, fontweight='bold'
            )
            labels_info.append({'text_obj': text_obj, 'y_pos': y_offset})

    # Auto-extend y-limits so labels aren't clipped
    if labels_info:
        fig = ax.get_figure()
        fig.canvas.draw()  # ensure text extents are computed
        max_label_y = max(info['y_pos'] for info in labels_info)
        max_text_h = max(info['text_obj'].get_window_extent().height for info in labels_info)
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        # Convert pixels to data units
        text_h_data = max_text_h * yr / (ax.bbox.height * fig.dpi / 100)
        ax.set_ylim(y0, max_label_y + text_h_data + Config.LABEL_PADDING * yr)

def setup_plot_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str, scientific_y: bool = True) -> None:
    """Standardise plot axes formatting with LaTeX-safe labels."""
    title = latexify_label(title)
    xlabel = latexify_label(xlabel)
    ylabel = latexify_label(ylabel)
    ax.set_title(title, fontweight='bold', fontsize=Config.FONT_SIZE_TITLE, pad=20)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=Config.FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=Config.FONT_SIZE_LABEL)
    if scientific_y:
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

def get_domain_type(results: dict) -> str:
    """Extract domain type from results dict."""
    return 'sulcus' if 'sulcus_specific' in results.get('flux_metrics', {}) else 'rectangular'

def get_mode(results: dict) -> str:
    """Extract simulation mode from results dict (falls back to 'unknown')."""
    return getattr(results.get('params', {}), 'mode', 'unknown')

def create_barplot_with_auto_resize(
    data, x, y, hue=None, palette=None, ax=None,
    title="", xlabel="", ylabel="", scientific_y=True,
    label_mode="dual", rotation=0, **kwargs):
    """
    Create a barplot with standardised style, value labels, and auto y-limit adjustment.
    Back-compat: if callers still pass dual_labels=True/False, we translate that
    into label_mode="dual"/"simple" so the external code doesn't break.
    """
    # Backwards compatibility
    if 'dual_labels' in kwargs:
        dual = kwargs.pop('dual_labels')
        # Respect explicit label_mode if provided; otherwise map the legacy flag
        if label_mode in ("dual", "simple"):
            label_mode = "dual" if bool(dual) else "simple"

    if ax is None:
        ax = plt.gca()

    sns.barplot(
        data=data, x=x, y=y, hue=hue, palette=palette,
        alpha=Config.ALPHA, edgecolor='black', linewidth=Config.LINEWIDTH,
        ax=ax, **kwargs,
    )

    setup_plot_axes(ax, title, xlabel, ylabel, scientific_y)

    # Gentle x padding so edge bars don't kiss the frame
    x0, x1 = ax.get_xlim()
    xr = x1 - x0
    ax.set_xlim(x0 - 0.05 * xr, x1 + 0.05 * xr)

    add_value_labels(ax, label_mode=label_mode, rotation=rotation)
    return ax

def _barplot_and_legend(filename, figsize, data, x, y, hue, palette, title, xlabel, ylabel,
                        scientific_y=True, label_mode='simple', rotation=45, legend_title=None, xtick_rotation=45):
    """Small helper to reduce repetition when we make barplots then tweak legends/xticks."""
    with safe_plot(filename, figsize):
        ax = create_barplot_with_auto_resize(
            data=data, x=x, y=y, hue=hue, palette=palette,
            title=title, xlabel=xlabel, ylabel=ylabel,
            scientific_y=scientific_y, label_mode=label_mode, rotation=rotation,
        )
        if legend_title is not None and hue is not None:
            ax.legend(title=legend_title, loc='upper right', fontsize=Config.FONT_SIZE_LEGEND)
        if xtick_rotation:
            plt.xticks(rotation=xtick_rotation, ha='right')
        return ax

def create_study_dirs(study_type, base_dir):
    """Create organised directories."""
    study_dir = os.path.join(base_dir, f"{study_type} Analysis")
    sim_dir = os.path.join(base_dir, f"{study_type} Simulations")
    os.makedirs(study_dir, exist_ok=True)
    os.makedirs(sim_dir, exist_ok=True)
    return study_dir, sim_dir

def format_filename_value(value):
    """Format value for filenames."""
    if abs(value - round(value)) < 0.001:
        return f"{value:.0f}"
    return f"{value:.1f}".replace('.', 'p') if value >= 1.0 else f"{value:.3f}".replace('.', 'p')

def latexify_label(text):
    """Convert plain text or Unicode into LaTeX-safe label for easier plotting."""
    if text is None:
        return None

    replacements = {
        "μ_eff": r"$\mu_{\mathrm{eff}}$",
        "μ": r"$\mu$",
        "Pe": r"$\mathrm{Pe}$",
        "∫|j| ds": r"$\int \lvert j \rvert \, ds$",
        "×": r"$\times$",
        "≤": r"$\le$",
        "≥": r"$\ge$",
         "±": r"$\pm$",
         "λ": r"$\lambda$",
         "%": r"\%",
         "Δ": r"$\Delta$",
         "−": r"$-$",
         "–": r"--",
         "—": r"---",
         "δ": r"$\delta$",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text

def set_colorbar_label(cbar, label: str, fontsize: int = None, fontweight: str = 'bold'):
    """Apply LaTeX-safe label to a colorbar."""
    cbar.set_label(latexify_label(label), fontsize=(fontsize or Config.FONT_SIZE_LABEL), fontweight=fontweight)

# ========================================================
# Field Visualisation Plots
# ========================================================

def plot_mesh_vis(results, plots_dir):
    """Plot mesh visualisation."""
    if 'mesh_results' not in results or 'mesh' not in results['mesh_results']:
        return

    mesh_results = results['mesh_results']
    mesh = mesh_results['mesh']
    domain_type = get_domain_type(results)

    filename = os.path.join(plots_dir, 'mesh.png')
    with safe_plot(filename, (10, 8)):
        try:
            import matplotlib.tri as tri
            coords = mesh.coordinates()
            cells = mesh.cells()
            triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
            plt.triplot(triangulation, 'k-', alpha=0.6, linewidth=0.5)
            setup_plot_axes(plt.gca(), f'Mesh - {domain_type.title()}', 'x', 'y', scientific_y=False)
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            # Annotate
            num_vertices = mesh_results.get('num_vertices', mesh.num_vertices())
            num_cells = mesh_results.get('num_cells', mesh.num_cells())
            info_text = f"Vertices: {num_vertices}\nCells: {num_cells}"
            plt.figtext(0.02, 0.98, info_text, fontsize=Config.FONT_SIZE_VALUE,
                        ha='left', va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        except Exception as e:
            print(f"Error creating mesh plot: {e}")

def plot_velocity_vis(results, plots_dir):
    """Plot velocity magnitude field."""
    if 'u' not in results:
        return

    u = results['u']
    domain_type = get_domain_type(results)
    mode = get_mode(results)

    filename = os.path.join(plots_dir, 'velocity_field.png')
    with safe_plot(filename, (12, 9)):
        try:
            mesh = u.function_space().mesh()
            V_scalar = FunctionSpace(mesh, "P", 1)
            u_mag = project(sqrt(dot(u, u)), V_scalar)
            c = plot(u_mag, title=f'Velocity Field - {domain_type.title()} ({mode})')
            plt.colorbar(c, label='Velocity Magnitude', shrink=0.8)
            setup_plot_axes(plt.gca(), f'Velocity Field - {domain_type.title()} ({mode})', 'x', 'y', scientific_y=False)
            arr = u_mag.vector().get_local()
            stats = f"Max: {np.max(arr):.4f}\nMean: {np.mean(arr):.4f}"
            plt.figtext(0.02, 0.98, stats, fontsize=Config.FONT_SIZE_VALUE,
                        ha='left', va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        except Exception as e:
            print(f"Error creating velocity plot: {e}")

def plot_concentration_vis(results, plots_dir):
    """Plot concentration field."""
    if 'c' not in results:
        return

    c = results['c']
    domain_type = get_domain_type(results)
    mode = get_mode(results)

    filename = os.path.join(plots_dir, 'concentration_field.png')
    with safe_plot(filename, (12, 9)):
        try:
            pc = plot(c, title=f'Concentration Field - {domain_type.title()} ({mode})')
            plt.colorbar(pc, label='Concentration', shrink=0.8)
            setup_plot_axes(plt.gca(), f'Concentration Field - {domain_type.title()} ({mode})', 'x', 'y', scientific_y=False)
            arr = c.vector().get_local()
            stats = f"Max: {np.max(arr):.4f}\nMean: {np.mean(arr):.4f}"
            plt.figtext(0.02, 0.98, stats, fontsize=Config.FONT_SIZE_VALUE,
                        ha='left', va='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        except Exception as e:
            print(f"Error creating concentration plot: {e}")

def plot_field_visualisations(results, plots_dir):
    """Generate field visualisation plots."""
    print("📊 Generating field visualisation plots...")
    os.makedirs(plots_dir, exist_ok=True)
    plot_mesh_vis(results, plots_dir)
    plot_velocity_vis(results, plots_dir)
    plot_concentration_vis(results, plots_dir)
    print(f"✅ Field visualisation plots saved to {plots_dir}")

# ========================================================
# Single Simulation Plots
# ========================================================

def plot_flux_analysis(results, plots_dir):
    """Generate comprehensive flux analysis plots."""
    flux_metrics = results.get('flux_metrics', {})
    if not flux_metrics:
        return

    print("📊 Generating flux analysis plots...")
    os.makedirs(plots_dir, exist_ok=True)

    domain_type = get_domain_type(results)
    mode = get_mode(results)

    # Extract flux data once
    left_flux_diff   = flux_metrics['physical_flux']['left']['diffusive']
    left_flux_adv    = flux_metrics['physical_flux']['left']['advective']
    left_flux_total  = flux_metrics['physical_flux']['left']['total']
    right_flux_diff  = flux_metrics['physical_flux']['right']['diffusive']
    right_flux_adv   = flux_metrics['physical_flux']['right']['advective']
    right_flux_total = flux_metrics['physical_flux']['right']['total']
    top_flux_diff    = flux_metrics['physical_flux']['top']['diffusive']
    top_flux_adv     = flux_metrics['physical_flux']['top']['advective']
    top_flux_total   = flux_metrics['physical_flux']['top']['total']
    bottom_flux_diff = flux_metrics['physical_flux']['bottom']['diffusive']
    bottom_flux_adv  = flux_metrics['physical_flux']['bottom']['advective']
    bottom_flux_total= flux_metrics['physical_flux']['bottom']['total']

    uptake_total = flux_metrics.get('uptake_flux', 0)

    sulcus_data_available = (
        domain_type == 'sulcus' and 'sulcus_specific' in flux_metrics and 'physical_flux' in flux_metrics['sulcus_specific']
    )

    if sulcus_data_available:
        sflux = flux_metrics['sulcus_specific']['physical_flux']
        bottom_left_diff,  bottom_left_adv,  bottom_left_total  = sflux['bottom_left'].values()
        sulcus_curve_diff, sulcus_curve_adv, sulcus_curve_total = sflux['sulcus'].values()
        bottom_right_diff, bottom_right_adv, bottom_right_total = sflux['bottom_right'].values()
        s_open = sflux['sulcus_opening']
        sulcus_opening_diff  = s_open['diffusive']
        sulcus_opening_adv   = s_open['advective']
        sulcus_opening_total = s_open['total']
        y0_diff, y0_adv, y0_total = sflux['y0_flux'].values()
        bottom_combined_diff, bottom_combined_adv, bottom_combined_total = sflux['bottom_combined'].values()
        y0_combined_diff, y0_combined_adv, y0_combined_total = sflux['y0_combined'].values()
        bottom_left_uptake = flux_metrics['sulcus_specific']['uptake_flux']['bottom_left']
        sulcus_uptake      = flux_metrics['sulcus_specific']['uptake_flux']['sulcus']
        bottom_right_uptake= flux_metrics['sulcus_specific']['uptake_flux']['bottom_right']
        uptake_total_combined = flux_metrics['sulcus_specific']['uptake_flux']['total']

    # Plot 1: Flux Overview (external + sulcus + combined)
    data_rows = []
    for boundary, flux_val in [
        ('Left', abs(left_flux_total)), ('Right', abs(right_flux_total)),
        ('Top', abs(top_flux_total)),  ('Bottom', abs(bottom_flux_total))
    ]:
        data_rows.append({'Boundary': f'Ext.\n{boundary}', 'Flux': max(flux_val, 1e-16), 'Category': 'External'})

    if sulcus_data_available:
        for boundary, val in [
            ('Bottom Left', abs(bottom_left_total)),
            ('Sulcus Curve', abs(sulcus_curve_total)),
            ('Bottom Right', abs(bottom_right_total)),
            ('Sulcus Opening', abs(sulcus_opening_total)),
            ('Y=0 Line', abs(y0_total)),
        ]:
            boundary_label = boundary.replace(" ", "\n")
            data_rows.append({
                'Boundary': f"Sul.\n{boundary_label}",
                'Flux': max(val, 1e-16),
                'Category': 'Sulcus'
            })
        for boundary, val in [('Bottom Combined', abs(bottom_combined_total)), ('Y=0 Combined', abs(y0_combined_total))]:
            boundary_label = boundary.replace(" ", "\n")
            data_rows.append({
                'Boundary': f"Comb.\n{boundary_label}",
                'Flux': max(val, 1e-16),
                'Category': 'Combined'
            })

    if data_rows:
        df = pd.DataFrame(data_rows)
        _barplot_and_legend(
            os.path.join(plots_dir, 'flux_overview_comprehensive.png'), (16, 8), df,
            x='Boundary', y='Flux', hue='Category', palette=['steelblue', 'lightcoral', 'gold'],
            title=f'Comprehensive Flux Overview - {domain_type.title()} Domain ({mode})',
            xlabel='Boundary/Region', ylabel='Total Flux Magnitude',
            scientific_y=True, label_mode='simple', rotation=45, legend_title='Type'
        )

    # Plot 2: External boundaries breakdown
    ext_rows = []
    for boundary, diff, adv, total in [
        ('Left', left_flux_diff, left_flux_adv, left_flux_total),
        ('Right', right_flux_diff, right_flux_adv, right_flux_total),
        ('Top', top_flux_diff, top_flux_adv, top_flux_total),
        ('Bottom', bottom_flux_diff, bottom_flux_adv, bottom_flux_total),
    ]:
        ext_rows += [
            {'Boundary': boundary, 'Flux': abs(diff),  'Type': 'Diffusive'},
            {'Boundary': boundary, 'Flux': abs(adv),   'Type': 'Advective'},
            {'Boundary': boundary, 'Flux': abs(total), 'Type': 'Total'},
        ]

    if ext_rows:
        df_ext = pd.DataFrame(ext_rows)
        _barplot_and_legend(
            os.path.join(plots_dir, 'external_flux_breakdown.png'), (12, 8), df_ext,
            x='Boundary', y='Flux', hue='Type', palette='viridis',
            title=f'External Boundaries Flux Breakdown - {domain_type.title()} ({mode})',
            xlabel='External Boundary', ylabel='Flux Magnitude',
            scientific_y=True, label_mode='simple', rotation=45, legend_title='Flux Type'
        )

    # Plot 3: Sulcus breakdown
    if sulcus_data_available:
        s_rows = [
            {'Region': 'Ext. Bottom', 'Flux': abs(bottom_flux_diff),  'Type': 'Diffusive'},
            {'Region': 'Ext. Bottom', 'Flux': abs(bottom_flux_adv),   'Type': 'Advective'},
            {'Region': 'Ext. Bottom', 'Flux': abs(bottom_flux_total), 'Type': 'Total'},
        ]
        for region, diff, adv, total in [
            ('Bottom Left', bottom_left_diff, bottom_left_adv, bottom_left_total),
            ('Sulcus Curve', sulcus_curve_diff, sulcus_curve_adv, sulcus_curve_total),
            ('Bottom Right', bottom_right_diff, bottom_right_adv, bottom_right_total),
            ('Sulcus Opening', sulcus_opening_diff, sulcus_opening_adv, sulcus_opening_total),
            ('Y=0 Line', y0_diff, y0_adv, y0_total),
            ('Bottom Combined', bottom_combined_diff, bottom_combined_adv, bottom_combined_total),
            ('Y=0 Combined', y0_combined_diff, y0_combined_adv, y0_combined_total),
        ]:
            s_rows += [
                {'Region': region, 'Flux': abs(diff),  'Type': 'Diffusive'},
                {'Region': region, 'Flux': abs(adv),   'Type': 'Advective'},
                {'Region': region, 'Flux': abs(total), 'Type': 'Total'},
            ]

        df_s = pd.DataFrame(s_rows)
        _barplot_and_legend(
            os.path.join(plots_dir, 'sulcus_flux_breakdown.png'), (16, 8), df_s,
            x='Region', y='Flux', hue='Type', palette='viridis',
            title=f'Sulcus-Specific Flux Breakdown - {domain_type.title()} ({mode})',
            xlabel='Region/Segment', ylabel='Flux Magnitude',
            scientific_y=True, label_mode='simple', rotation=45, legend_title='Flux Type', xtick_rotation=45
        )

    # Plot 4: Consistency check (sulcus only)
    if sulcus_data_available:
        cons_rows = [
            {'Method': 'Bottom\n(External)', 'Flux': bottom_flux_diff,  'Type': 'Diffusive', 'Group': 'Bottom'},
            {'Method': 'Bottom\n(External)', 'Flux': bottom_flux_adv,   'Type': 'Advective', 'Group': 'Bottom'},
            {'Method': 'Bottom\n(External)', 'Flux': bottom_flux_total, 'Type': 'Total',     'Group': 'Bottom'},
            {'Method': 'Bottom\n(Combined)', 'Flux': bottom_combined_diff,  'Type': 'Diffusive', 'Group': 'Bottom'},
            {'Method': 'Bottom\n(Combined)', 'Flux': bottom_combined_adv,   'Type': 'Advective', 'Group': 'Bottom'},
            {'Method': 'Bottom\n(Combined)', 'Flux': bottom_combined_total, 'Type': 'Total',     'Group': 'Bottom'},
            {'Method': 'Y=0\n(Direct)', 'Flux': y0_diff,  'Type': 'Diffusive', 'Group': 'Y=0'},
            {'Method': 'Y=0\n(Direct)', 'Flux': y0_adv,   'Type': 'Advective', 'Group': 'Y=0'},
            {'Method': 'Y=0\n(Direct)', 'Flux': y0_total, 'Type': 'Total',     'Group': 'Y=0'},
            {'Method': 'Y=0\n(Combined)', 'Flux': y0_combined_diff,  'Type': 'Diffusive', 'Group': 'Y=0'},
            {'Method': 'Y=0\n(Combined)', 'Flux': y0_combined_adv,   'Type': 'Advective', 'Group': 'Y=0'},
            {'Method': 'Y=0\n(Combined)', 'Flux': y0_combined_total, 'Type': 'Total',     'Group': 'Y=0'},
        ]
        df_c = pd.DataFrame(cons_rows)
        ax = _barplot_and_legend(
            os.path.join(plots_dir, 'flux_consistency_detailed.png'), (14, 8), df_c,
            x='Method', y='Flux', hue='Type', palette='viridis',
            title=f'Flux Consistency Check - {domain_type.title()} ({mode})',
            xlabel='Calculation Method', ylabel='Flux Value',
            scientific_y=True, label_mode='simple', rotation=45, legend_title='Flux Type', xtick_rotation=45
        )
        ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Plot 5: Physical vs Robin comparison
    if mode != 'no-uptake':
        rows = [
            {'Method': 'Physical\n(Bottom)',   'Flux': abs(bottom_flux_total),      'Type': 'Physical'}
        ]
        rows.append({'Method': 'Robin BC\n(Bottom)', 'Flux': abs(uptake_total), 'Type': 'Robin'})
        if sulcus_data_available:
            rows += [
                {'Method': 'Physical\n(Combined)', 'Flux': abs(bottom_combined_total),   'Type': 'Physical'},
                {'Method': 'Robin BC\n(Combined)', 'Flux': abs(uptake_total_combined),   'Type': 'Robin'},
            ]
        df_r = pd.DataFrame(rows)
        ax = _barplot_and_legend(
            os.path.join(plots_dir, 'physical_vs_robin_detailed.png'), (10, 8), df_r,
            x='Method', y='Flux', hue='Type', palette=['steelblue', 'darkorange'],
            title=f'Physical vs Robin BC Flux - {domain_type.title()} ({mode})',
            xlabel='Flux Calculation Method', ylabel='Flux Magnitude',
            scientific_y=True, label_mode='simple', rotation=45, legend_title='Flux Type', xtick_rotation=45
        )
        if abs(uptake_total) > 1e-16:
            ratio1 = abs(bottom_flux_total) / abs(uptake_total)
            y_max = max(df_r['Flux'])
            ax.text(0.5, y_max * 1.1, f'Ext. Ratio: {ratio1:.4f}', ha='center', va='bottom',
                    fontsize=Config.FONT_SIZE_VALUE, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        if sulcus_data_available and abs(uptake_total_combined) > 1e-16:
            ratio2 = abs(bottom_combined_total) / abs(uptake_total_combined)
            ax.text(2.5, y_max * 1.05, f'Comb. Ratio: {ratio2:.4f}', ha='center', va='bottom',
                    fontsize=Config.FONT_SIZE_VALUE, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

    # Plot 6: Segment-wise ratios
    if sulcus_data_available and mode != 'no-uptake':
        seg_rows = []
        for segment, physical, uptake in [
            ('Bottom Left', bottom_left_total, bottom_left_uptake),
            ('Sulcus',      sulcus_curve_total, sulcus_uptake),
            ('Bottom Right',bottom_right_total, bottom_right_uptake),
        ]:
            ratio = abs(physical) / abs(uptake) if abs(uptake) > 1e-16 else float('nan')
            seg_rows.append({'Segment': segment, 'Ratio': ratio})
        df_seg = pd.DataFrame(seg_rows)
        with safe_plot(os.path.join(plots_dir, 'segment_flux_ratio.png'), (10, 6)):
            ax = create_barplot_with_auto_resize(
                data=df_seg, x='Segment', y='Ratio', palette='viridis',
                title=f'Segment-wise Physical vs Robin Flux Ratio\n{domain_type.title()} Domain ({mode})',
                xlabel='Segment', ylabel='|Physical Flux| / |Uptake Flux|',
                scientific_y=True, label_mode='simple', rotation=45,
            )
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='Ideal Ratio = 1')
            ax.legend(loc='upper right', fontsize=Config.FONT_SIZE_LEGEND)

    print(f"✅ Comprehensive flux analysis plots saved to {plots_dir}")

def plot_mass_analysis(results, plots_dir):
    """Generate comprehensive mass analysis plots."""
    mm = results.get('mass_metrics', {})
    if not mm:
        return

    print("📊 Generating mass analysis plots...")
    os.makedirs(plots_dir, exist_ok=True)

    domain_type = get_domain_type(results)

    # Plot 1: Total mass distribution
    if domain_type == 'sulcus':
        data = pd.DataFrame({'Region': ['Sulcus', 'Rectangle'], 'Mass': [mm['sulcus_mass'], mm['rectangle_mass']]})
        palette = ['lightcoral', 'lightsteelblue']
    else:
        data = pd.DataFrame({'Region': ['Total'], 'Mass': [mm['total_mass']]})
        palette = ['lightsteelblue']

    with safe_plot(os.path.join(plots_dir, 'total_mass.png'), (8, 7)):
        ax = create_barplot_with_auto_resize(
            data=data, x='Region', y='Mass', palette=palette,
            title=f'Mass Distribution - {domain_type.title()}',
            xlabel='Region', ylabel='Mass', scientific_y=True,
            label_mode='simple', rotation=45,
        )
        if domain_type == 'sulcus':
            ratio = mm['sulcus_mass'] / mm['rectangle_mass']
            y_max = max(data['Mass'])
            ax.text(0.5, y_max * 1.05, f'S/R Ratio: {ratio:.3f}', ha='center', va='bottom',
                    fontsize=Config.FONT_SIZE_VALUE, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))

    # Plot 2: Average concentration
    if domain_type == 'sulcus':
        avg = mm['average_concentration']
        data = pd.DataFrame({'Region': ['Sulcus', 'Rectangle', 'Total'],
                             'Concentration': [avg['sulcus_region'], avg['rectangle_region'], avg['total']]})
        palette = ['lightcoral', 'lightsteelblue', 'gold']
    else:
        data = pd.DataFrame({'Region': ['Average'], 'Concentration': [mm['average_concentration']]})
        palette = ['gold']

    with safe_plot(os.path.join(plots_dir, 'average_concentration.png'), (8, 7)):
        create_barplot_with_auto_resize(
            data=data, x='Region', y='Concentration', palette=palette,
            title=f'Average Concentration - {domain_type.title()}',
            xlabel='Region', ylabel='Average Concentration', scientific_y=True,
            label_mode='simple', rotation=45,
        )

    print(f"✅ Mass analysis plots saved to {plots_dir}")

def plot_mu_eff_analysis(results, plots_dir):
    """Generate comprehensive μ_eff analysis plots (robust to step μ(x))."""
    domain_type = get_domain_type(results)
    mode = get_mode(results)

    if mode == 'no-uptake':
        return

    print("📊 Generating μ_eff analysis plots...")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Build a small table we can always plot ---
    rows = []

    if domain_type == 'sulcus' and 'mu_eff_comparison' in results:
        mu = results['mu_eff_comparison']
        rows = [
            {'Method': 'Simulation (Full)',      'Value': mu.get('mu_eff_simulation_full')},
            {'Method': 'Simulation (Segmented)', 'Value': mu.get('mu_eff_simulation_segmented')},
            {'Method': 'Analytical',             'Value': mu.get('mu_eff_analytical')},
            {'Method': 'Enhanced',               'Value': mu.get('mu_eff_enhanced')},
        ]
    else:
        # Rectangular cases
        params = results.get('params', None)
        mu_obj = getattr(params, 'mu', None)

        # Uniform rectangle -> scalar μ
        try:
            # Constant or float/int
            from dolfin import Constant, UserExpression
        except Exception:
            Constant = object
            UserExpression = object

        if isinstance(mu_obj, (int, float)):
            rows = [{'Method': 'Baseline', 'Value': float(mu_obj)}]
        elif 'Constant' in type(mu_obj).__name__:  # works for dolfin.Constant without import fuss
            rows = [{'Method': 'Baseline', 'Value': float(mu_obj())}]
        elif hasattr(mu_obj, 'mu_base') and hasattr(mu_obj, 'mu_open'):
            # StepUptakeFunction: show its parts and the bottom-average used for matching μ_eff
            mu_base = float(mu_obj.mu_base)
            mu_open = float(mu_obj.mu_open)

            # If we can read geometry from params, compute the bottom-average actually implied:
            L = float(getattr(params, 'L', 1.0))
            w = float(getattr(params, 'sulci_w', 0.0))
            avg = ( (L - w) * mu_base + w * mu_open ) / L if L > 0 else None

            rows = [
                {'Method': 'Baseline (outside opening)', 'Value': mu_base},
                {'Method': 'Opening (inside)',           'Value': mu_open},
                {'Method': r'Bottom-avg $\mu(x)$',       'Value': avg},
            ]
        else:
            # Fallback: nothing sensible to plot
            rows = []

    data = pd.DataFrame(rows)
    if 'Value' in data:
        data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
        data = data.dropna(subset=['Value'])

    if data.empty:
        # Quietly skip if we can't assemble numeric data (e.g. weird custom expression)
        return

    with safe_plot(os.path.join(plots_dir, 'mu_eff_methods.png'), (10, 7)):
        create_barplot_with_auto_resize(
            data=data, x='Method', y='Value', palette='Set2',
            title=rf'$\mu_{{\mathrm{{eff}}}}$ Methods - {domain_type.title()}',
            xlabel='Method', ylabel=r'Effective Uptake Rate ($\mu_{\mathrm{eff}}$)',
            scientific_y=False, label_mode='simple', rotation=0,
        )

    # Optionally keep the ratio plot for sulcus only
    mu = results.get('mu_eff_comparison', {})
    ratios = mu.get('mu_eff_ratios', {})
    if ratios and domain_type == 'sulcus':
        order = ['Simulation (Full)', 'Simulation (Segmented)', 'Analytical', 'Enhanced']
        key_to_label = {
            'simulation_full_ratio': 'Simulation (Full)',
            'simulation_segmented_ratio': 'Simulation (Segmented)',
            'analytical_ratio': 'Analytical',
            'enhanced_ratio': 'Enhanced',
        }
        rows = []
        for k, label in key_to_label.items():
            v = ratios.get(k)
            if v is not None:
                rows.append({'Method': label, 'Ratio': v})
        if rows:
            df = pd.DataFrame(rows)
            with safe_plot(os.path.join(plots_dir, 'mu_eff_ratios.png'), (10, 7)):
                create_barplot_with_auto_resize(
                    data=df, x='Method', y='Ratio', palette='Set2',
                    title=rf'$\mu_{{\mathrm{{eff}}}} / \mu$ Ratios - {domain_type.title()}',
                    xlabel='Method', ylabel=rf'$\mu_{{\mathrm{{eff}}}} / \mu$',
                    scientific_y=False, label_mode='simple', rotation=0,
                )

def plot_single_simulation(results, plots_dir):
    """Generate all plots for a single simulation."""
    print("📊 Generating single simulation plots...")
    os.makedirs(plots_dir, exist_ok=True)

    domain_type = get_domain_type(results)
    mode = get_mode(results)
    print(f"    Domain: {domain_type}, Mode: {mode}")

    plot_flux_analysis(results, plots_dir)
    plot_mass_analysis(results, plots_dir)
    plot_mu_eff_analysis(results, plots_dir)
    plot_field_visualisations(results, plots_dir)

    print(f"✅ Single simulation plots saved to {plots_dir}")

# ========================================================
# Configuration Comparison Plots
# ========================================================

def plot_flux_comparison(all_results, plots_dir):
    """Compare flux across configurations."""
    rows = []
    mode = domain_type = None

    for name, results in all_results.items():
        if results is None or 'flux_metrics' not in results:
            continue
        if mode is None:
            mode = get_mode(results)
            domain_type = get_domain_type(results)
        clean = name.replace('_', ' ').title()
        fm = results['flux_metrics']
        rows.append({'Configuration': clean, 'Flux': abs(fm['physical_flux']['bottom']['total']), 'Type': 'Physical'})
        if mode != 'no-uptake':
            rows.append({'Configuration': clean, 'Flux': abs(fm.get('uptake_flux', 0)), 'Type': 'Uptake'})

    if not rows:
        return

    df = pd.DataFrame(rows)
    with safe_plot(os.path.join(plots_dir, 'flux_comparison.png'), (14, 9)):
        ax = create_barplot_with_auto_resize(
            data=df, x='Configuration', y='Flux', hue='Type', palette=['lightcoral', 'lightseagreen'],
            title=f'Flux Comparison - {mode.replace("-", " ").title()}',
            xlabel='Configuration', ylabel='Flux Magnitude', scientific_y=True,
            label_mode='simple', rotation=45,
        )
        plt.xticks(rotation=45, ha='right')

def plot_mass_comparison(all_results, plots_dir):
    """Compare total mass across configurations."""
    rows = []
    for name, results in all_results.items():
        if results is None or 'mass_metrics' not in results:
            continue
        rows.append({'Configuration': name.replace('_', ' ').title(), 'Mass': results['mass_metrics']['total_mass']})

    if not rows:
        return

    df = pd.DataFrame(rows)
    with safe_plot(os.path.join(plots_dir, 'mass_comparison.png'), (12, 9)):
        create_barplot_with_auto_resize(
            data=df, x='Configuration', y='Mass', palette=['skyblue'],
            title='Total Mass Comparison', xlabel='Configuration', ylabel='Total Mass', scientific_y=True,
            label_mode='simple', rotation=45,
        )
        plt.xticks(rotation=45, ha='right')

def plot_concentration_comparison(all_results, plots_dir):
    """Compare average concentration across configurations."""
    rows = []
    domain_type = None

    for name, results in all_results.items():
        if results is None or 'mass_metrics' not in results:
            continue
        if domain_type is None:
            domain_type = get_domain_type(results)
        clean = name.replace('_', ' ').title()
        mm = results['mass_metrics']
        avg = (mm['average_concentration']['total'] if domain_type == 'sulcus' else mm['average_concentration'])
        rows.append({'Configuration': clean, 'Concentration': avg})

    if not rows:
        return

    df = pd.DataFrame(rows)
    with safe_plot(os.path.join(plots_dir, 'concentration_comparison.png'), (12, 9)):
        create_barplot_with_auto_resize(
            data=df, x='Configuration', y='Concentration', palette=['lightgreen'],
            title='Average Concentration Comparison', xlabel='Configuration', ylabel='Average Concentration',
            scientific_y=True, label_mode='simple', rotation=45,
        )
        plt.xticks(rotation=45, ha='right')

def plot_mu_eff_comparison(all_results, plots_dir):
    """Compare μ_eff across configurations, including both simulation methods."""
    rows = []
    mode = domain_type = None

    for name, results in all_results.items():
        if results is None:
            continue
        if mode is None:
            mode = get_mode(results)
            domain_type = get_domain_type(results)
        clean = name.replace('_', ' ').title()
        if 'mu_eff_comparison' in results:
            mu = results['mu_eff_comparison']
            mapping = {
                'Simulation (Full)': mu.get('mu_eff_simulation_full'),
                'Simulation (Segmented)': mu.get('mu_eff_simulation_segmented'),
                'Analytical': mu.get('mu_eff_analytical'),
                'Enhanced': mu.get('mu_eff_enhanced'),
            }
            for method, val in mapping.items():
                if val is not None:
                    rows.append({'Configuration': clean, 'μ_eff': val, 'Type': method})
        else:
            base_mu = getattr(results.get('params', {}), 'mu', 1.0)
            rows.append({'Configuration': clean, 'μ_eff': base_mu, 'Type': 'Baseline'})

    if not rows or mode == 'no-uptake':
        return

    df = pd.DataFrame(rows)
    with safe_plot(os.path.join(plots_dir, 'mu_eff_comparison.png'), (14, 9)):
        ax = create_barplot_with_auto_resize(
            data=df, x='Configuration', y='μ_eff',
            hue='Type' if df['Type'].nunique() > 1 else None,
            palette='Set2' if df['Type'].nunique() > 1 else ['steelblue'],
            title=r'$\mu_{\mathrm{eff}}$ Comparison Across Configurations',
            xlabel='Configuration', ylabel=r'Effective Uptake Rate ($\mu_{\mathrm{eff}}$)',
            scientific_y=False, label_mode='simple', rotation=0,
        )
        plt.xticks(rotation=45, ha='right')
        if df['Type'].nunique() > 1:
            ax.legend(title='Method', loc='upper right', fontsize=Config.FONT_SIZE_LEGEND)

def plot_mu_eff_ratios_comparison(all_results, plots_dir):
    """Compare μ_eff / μ ratios across configurations, including both simulation methods."""
    rows = []
    mode = None

    for name, results in all_results.items():
        if results is None:
            continue
        if mode is None:
            mode = get_mode(results)
        label = name.replace('_', ' ').title()
        mu = results.get('mu_eff_comparison', {})
        ratios = mu.get('mu_eff_ratios', {})
        for k, lbl in {
            'simulation_full_ratio': 'Simulation (Full)',
            'simulation_segmented_ratio': 'Simulation (Segmented)',
            'analytical_ratio': 'Analytical',
            'enhanced_ratio': 'Enhanced',
        }.items():
            v = ratios.get(k)
            if v is not None:
                rows.append({'Configuration': label, 'Method': lbl, 'Ratio': v})

    if not rows or mode == 'no-uptake':
        return

    df = pd.DataFrame(rows)
    df['Method'] = pd.Categorical(df['Method'], categories=['Simulation (Full)', 'Simulation (Segmented)', 'Analytical', 'Enhanced'], ordered=True)

    with safe_plot(os.path.join(plots_dir, 'mu_eff_ratios_comparison.png'), (14, 9)):
        ax = create_barplot_with_auto_resize(
            data=df, x='Configuration', y='Ratio', hue='Method', palette='Set2',
            title=r'$\mu_{\mathrm{eff}} / \mu$ Ratio Comparison Across Configurations',
            xlabel='Configuration', ylabel=r'Ratio ($\mu_{\mathrm{eff}} / \mu$)',
            scientific_y=False, label_mode='simple', rotation=0,
        )
        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Method', loc='upper right', fontsize=Config.FONT_SIZE_LEGEND)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Ideal (1.0)')

def plot_configuration_comparison(all_results, plots_dir):
    """Generate all configuration comparison plots."""
    print("📊 Generating configuration comparison plots...")
    os.makedirs(plots_dir, exist_ok=True)

    plot_flux_comparison(all_results, plots_dir)
    plot_mass_comparison(all_results, plots_dir)
    plot_concentration_comparison(all_results, plots_dir)
    plot_mu_eff_comparison(all_results, plots_dir)
    plot_mu_eff_ratios_comparison(all_results, plots_dir)
    save_summary(all_results, os.path.join(plots_dir, 'comparison_summary.json'))

    print(f"✅ Configuration comparison plots saved to {plots_dir}")

def plot_aspect_ratio_comparison(df_results, plots_dir):
    """Plot aspect ratio analysis with all three ratio types shown."""

    print("📊 Generating aspect ratio comparison plots...")

    # Main plot: All three ratios for each aspect ratio
    with safe_plot(os.path.join(plots_dir, 'aspect_ratio_all_methods.png'), figsize=(14, 8)):
        fig, ax = plt.subplots()

        # Aspect ratio styles and colors - LaTeX-safe labels
        aspect_styles = {
            'h_equals_w': {'base_color': 'purple', 'name': r'$h = w$ (square)'},
            'h_equals_2w': {'base_color': 'brown', 'name': r'$h = 2w$ (narrow \& deep)'},
            'h_equals_half_w': {'base_color': 'orange', 'name': r'$h = 0.5w$ (wide \& shallow)'}
        }

        # Line styles for different ratio types
        ratio_styles = {
            'ratio_sim': {'linestyle': '-', 'marker': 'o', 'alpha': 1.0, 'suffix': ' (Simulation)'},
            'ratio_analytical': {'linestyle': '--', 'marker': 's', 'alpha': 0.8, 'suffix': ' (Analytical)'},
            'ratio_enhanced': {'linestyle': ':', 'marker': '^', 'alpha': 0.8, 'suffix': ' (Enhanced)'}
        }

        # Plot each aspect ratio with all three methods
        for ar_type in df_results['aspect_ratio_type'].unique():
            ar_subset = df_results[df_results['aspect_ratio_type'] == ar_type].sort_values('depth')
            base_color = aspect_styles[ar_type]['base_color']
            base_name = aspect_styles[ar_type]['name']

            # Plot all three ratio types for this aspect ratio
            for ratio_col, style in ratio_styles.items():
                ax.plot(ar_subset['depth'], ar_subset[ratio_col],
                       color=base_color,
                       linestyle=style['linestyle'],
                       marker=style['marker'],
                       markersize=8,
                       alpha=style['alpha'],
                       label=base_name + style['suffix'],
                       linewidth=2.5)

        # Formatting
        ax.set_xlabel(r'Sulcus Depth $h$ (mm)', fontsize=Config.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel(r'$\mu_{\mathrm{eff}} / \mu$ Ratio', fontsize=Config.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title(r'Aspect Ratio Analysis: Simulation vs Models', fontsize=Config.FONT_SIZE_TITLE, fontweight='bold')

        # Reference line
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(0.1, 1.02, r'No enhancement', fontsize=Config.FONT_SIZE_VALUE, color='gray')

        # Grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, frameon=True, shadow=True, fontsize=Config.FONT_SIZE_LEGEND)

        # Set limits
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0.95)

        plt.tight_layout()

    # Separate subplots for each aspect ratio
    with safe_plot(os.path.join(plots_dir, 'aspect_ratio_subplots.png'), figsize=(15, 12)):
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

        for i, ar_type in enumerate(df_results['aspect_ratio_type'].unique()):
            ar_subset = df_results[df_results['aspect_ratio_type'] == ar_type].sort_values('depth')
            ax = axes[i]
            base_color = aspect_styles[ar_type]['base_color']
            base_name = aspect_styles[ar_type]['name']

            # Plot all three methods
            ax.plot(ar_subset['depth'], ar_subset['ratio_sim'],
                   color=base_color, linestyle='-', marker='o', markersize=8,
                   label='Simulation', linewidth=3, alpha=1.0)
            ax.plot(ar_subset['depth'], ar_subset['ratio_analytical'],
                   color=base_color, linestyle='--', marker='s', markersize=6,
                   label='Analytical', linewidth=2, alpha=0.7)
            ax.plot(ar_subset['depth'], ar_subset['ratio_enhanced'],
                   color=base_color, linestyle=':', marker='^', markersize=6,
                   label='Enhanced', linewidth=2, alpha=0.7)

            # Reference line
            ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

            # Formatting
            ax.set_title(base_name, fontsize=Config.FONT_SIZE_TITLE, fontweight='bold')
            ax.set_ylabel(r'$\mu_{\mathrm{eff}} / \mu$ Ratio', fontsize=Config.FONT_SIZE_LABEL)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            ax.set_ylim(bottom=0.95)

        # X-label only on bottom plot
        axes[-1].set_xlabel(r'Sulcus Depth $h$ (mm)', fontsize=Config.FONT_SIZE_LABEL, fontweight='bold')

        plt.tight_layout()

    # Error comparison plot
    with safe_plot(os.path.join(plots_dir, 'model_error_comparison.png'), figsize=(12, 8)):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        for ar_type in df_results['aspect_ratio_type'].unique():
            ar_subset = df_results[df_results['aspect_ratio_type'] == ar_type].sort_values('depth')
            base_color = aspect_styles[ar_type]['base_color']
            base_name = aspect_styles[ar_type]['name']

            # Absolute errors
            analytical_error = abs(ar_subset['mu_eff_sim'] - ar_subset['mu_eff_analytical'])
            enhanced_error = abs(ar_subset['mu_eff_sim'] - ar_subset['mu_eff_enhanced'])

            ax1.plot(ar_subset['depth'], analytical_error,
                    color=base_color, linestyle='--', marker='s',
                    label=base_name + ' (Analytical)', linewidth=2)
            ax1.plot(ar_subset['depth'], enhanced_error,
                    color=base_color, linestyle=':', marker='^',
                    label=base_name + ' (Enhanced)', linewidth=2)

            # Relative errors (%)
            analytical_rel_error = 100 * analytical_error / ar_subset['mu_eff_sim']
            enhanced_rel_error = 100 * enhanced_error / ar_subset['mu_eff_sim']

            ax2.plot(ar_subset['depth'], analytical_rel_error,
                    color=base_color, linestyle='--', marker='s',
                    label=base_name + ' (Analytical)', linewidth=2)
            ax2.plot(ar_subset['depth'], enhanced_rel_error,
                    color=base_color, linestyle=':', marker='^',
                    label=base_name + ' (Enhanced)', linewidth=2)

        ax1.set_ylabel(r'Absolute Error $|\mathrm{Simulation} - \mathrm{Model}|$', fontsize=Config.FONT_SIZE_LABEL)
        ax1.set_title(r'Model Accuracy Comparison', fontsize=Config.FONT_SIZE_TITLE, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', ncol=2)

        ax2.set_xlabel(r'Sulcus Depth $h$ (mm)', fontsize=Config.FONT_SIZE_LABEL, fontweight='bold')
        ax2.set_ylabel(r'Relative Error (\%)', fontsize=Config.FONT_SIZE_LABEL)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    print(f"✅ Aspect ratio comparison plots saved to {plots_dir}")

# ========================================================
# Summary + Entry Points
# ========================================================

def save_summary(all_results, filename):
    """Save JSON summary of results (μ_eff fields included when present)."""
    mode = None
    for r in all_results.values():
        if r is not None:
            mode = get_mode(r)
            break

    summary = {
        "simulation_info": {
            "mode": mode or "unknown",
            "timestamp": datetime.now().isoformat(),
            "total_configurations": len(all_results),
            "successful_runs": sum(1 for r in all_results.values() if r is not None),
        },
        "configurations": {},
    }

    for name, results in all_results.items():
        if results is None:
            summary["configurations"][name] = {"status": "failed"}
            continue

        cfg = {
            "status": "success",
            "total_mass": results['mass_metrics']['total_mass'],
            "bottom_flux": results['flux_metrics']['physical_flux']['bottom']['total'],
        }
        if mode != 'no-uptake':
            cfg["uptake_flux"] = results['flux_metrics'].get('uptake_flux', 0)
        if mode != 'no-uptake' and 'mu_eff_comparison' in results:
            mu = results['mu_eff_comparison']
            cfg.update({
                "mu_eff_simulation_full": mu.get('mu_eff_simulation_full'),
                "mu_eff_simulation_segmented": mu.get('mu_eff_simulation_segmented'),
                "mu_eff_analytical": mu.get('mu_eff_analytical'),
                "mu_eff_enhanced": mu.get('mu_eff_enhanced'),
                "mu_eff_ratios": mu.get('mu_eff_ratios', {}),
            })
        summary["configurations"][name] = cfg

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✓ Summary saved: {os.path.basename(filename)}")
    except Exception as e:
        print(f"❌ Error saving summary: {e}")

# Convenience wrappers - kept for backwards compatibility
def plot_all_metrics(results, plots_dir):
    """Generate all plots for a single simulation (main interface)."""
    plot_single_simulation(results, plots_dir)