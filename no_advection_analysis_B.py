##########################################################
# No Advection (Diffusion-Only)
##########################################################
"""
Runs diffusion-only simulations (pe = 0) for multiple geometries and uptake
values mu ∈ {0.1, 0.5, 1.0}. For each case it compares sulcus and rectangular
domains, computes concentration and flux ratios, saves a CSV summary, and
generates heatmaps. Allows either running simulations with
plots or replotting from an existing CSV.
"""

# ================
# Imports
# ================
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from parameters import Parameters, create_geometry_variations
from simulation import run_simulation
from plotting import safe_plot, set_style, Config, latexify_label, create_study_dirs

# ================
# Config
# ================
class NoAdvMuSweepConfig:
    # Dimensionless mu* factors (relative to Parameters.MU_DIM_NO_ADV baseline)
    MU_FACTORS = [0.1, 0.5, 1.0]

    # Output
    DEFAULT_OUTPUT_BASE = "Results/No Advection Simulations/mu Sweep"
    DEFAULT_CSV_NAME = "no_adv_mu_sweep_results.csv"

# ================
# Helpers
# ================
def _make_params_no_adv(mu_factor: float) -> Parameters:
    """
    Build Parameters for no-advection and scale mu_dim by 'mu_factor'
    using the class baseline Parameters.MU_DIM_NO_ADV (if present).
    """
    p = Parameters(mode='no-adv')
    baseline = getattr(Parameters, 'MU_DIM_NO_ADV', p.mu_dim)
    p.mu_dim = float(baseline) * float(mu_factor)
    p.validate()
    p.nondim()  # makes mu* = mu_dim * H / D
    return p

def _extract_flux(results: dict, domain_type: str):
    """Return signed total physical flux for y=0 (sulcus) or bottom (rectangle)."""
    fm = (results.get('flux_metrics') or {})
    if domain_type == 'sulcus':
        pf = ((fm.get('sulcus_specific') or {}).get('physical_flux') or {})
        for key in ('y0_flux', 'y0_combined'):
            if key in pf and isinstance(pf[key], dict):
                return pf[key].get('total', np.nan)
        return np.nan
    else:
        bot = ((fm.get('physical_flux') or {}).get('bottom', {}) or {})
        return bot.get('total', np.nan)

def _extract_avg_conc(results: dict, domain_type: str):
    """Return scalar average concentration for the whole domain."""
    mm = results.get('mass_metrics', {}) or {}
    avg = mm.get('average_concentration', None)
    if domain_type == 'sulcus':
        if isinstance(avg, dict):
            return avg.get('total', None)
        return None
    else:
        # rectangular case stores a scalar
        return avg if isinstance(avg, (int, float)) else None

# ================
# Core: run all geometries × mu
# ================

def run_no_adv_mu_sweep(output_base=None):
    """
    Returns DataFrame with columns:
      geometry, width_mm, depth_mm, aspect_ratio, mu_factor,
      avg_conc_sulc, avg_conc_rect,
      flux_sulc_y0, flux_rect_bottom,
      CR, flux_ratio, flux_error_pct
    """
    if output_base is None:
        output_base = NoAdvMuSweepConfig.DEFAULT_OUTPUT_BASE

    print("\n" + "="*64)
    print("NO ADVECTION — mu SWEEP OVER GEOMETRIES")
    print("="*64)

    t0 = time.time()
    study_dir, sim_dir = create_study_dirs("mu Sweep", output_base)

    # geometry set
    base = Parameters(mode='no-adv')
    configs = create_geometry_variations(base, max_width=1.0)
    print(f"• Geometries: {len(configs)}")
    print(f"• mu factors: {NoAdvMuSweepConfig.MU_FACTORS}")

    rows = []

    for mu in NoAdvMuSweepConfig.MU_FACTORS:
        print(f"\n--- mu* = {mu} ---")
        for gkey, gcfg in configs.items():
            try:
                # ----- Sulcus run -----
                ps = _make_params_no_adv(mu)
                ps.sulci_w_dim = gcfg['sulci_w_dim']
                ps.sulci_h_dim = gcfg['sulci_h_dim']
                ps.validate(); ps.nondim()

                name_s = f"{gkey}_mu{str(mu).replace('.','p')}"
                sulc = run_simulation(
                    mode='no-adv',
                    study_type="mu Sweep",
                    config_name=f"Sulcus_{name_s}",
                    domain_type='sulcus',
                    params=ps
                )

                # ----- Rectangular run -----
                pr = _make_params_no_adv(mu)
                pr.sulci_w_dim = gcfg['sulci_w_dim']  # harmless for rectangle mesh generator
                pr.sulci_h_dim = gcfg['sulci_h_dim']
                pr.validate(); pr.nondim()

                rect = run_simulation(
                    mode='no-adv',
                    study_type="mu Sweep",
                    config_name=f"Rect_{name_s}",
                    domain_type='rectangular',
                    params=pr
                )

                # ----- Metrics -----
                conc_s = _extract_avg_conc(sulc, 'sulcus')
                conc_r = _extract_avg_conc(rect, 'rectangular')

                flux_s = _extract_flux(sulc, 'sulcus')          # y=0
                flux_r = _extract_flux(rect, 'rectangular')     # bottom

                CR = (conc_s / conc_r) if (conc_s is not None and conc_r not in (None, 0)) else np.nan
                if flux_s is None or not np.isfinite(flux_s) or np.isclose(flux_s, 0.0):
                    flux_ratio = np.nan
                    flux_err = np.nan
                else:
                    flux_ratio = flux_r / flux_s
                    # % error vs sulcus reference (signed, magnitude wrt |Φ_s|)
                    denom = abs(flux_s) if not np.isclose(abs(flux_s), 0.0) else 1.0
                    flux_err = 100.0 * (flux_r - flux_s) / denom

                rows.append({
                    'geometry': gkey,
                    'width_mm': gcfg['sulci_w_dim'],
                    'depth_mm': gcfg['sulci_h_dim'],
                    'aspect_ratio': gcfg.get('aspect_ratio'),
                    'mu_factor': mu,
                    'avg_conc_sulc': conc_s,
                    'avg_conc_rect': conc_r,
                    'flux_sulc_y0': flux_s,
                    'flux_rect_bottom': flux_r,
                    'CR': CR,
                    'flux_ratio': flux_ratio,
                    'flux_error_pct': flux_err
                })

                # Console summary
                cr_str = f"{CR:.3f}" if np.isfinite(CR) else "nan"
                fr_str = f"{flux_ratio:.3f}" if np.isfinite(flux_ratio) else "nan"
                print(f"  ✓ {gkey}: CR={cr_str}  Flux ratio={fr_str}")

            except Exception as e:
                print(f"  ✗ {gkey} failed @ mu*={mu}: {e}")

    df = pd.DataFrame(rows).sort_values(['mu_factor', 'geometry']).reset_index(drop=True)

    # Save CSV + metadata
    csv_path = os.path.join(study_dir, NoAdvMuSweepConfig.DEFAULT_CSV_NAME)
    df.to_csv(csv_path, index=False)

    # Record a few baseline values from Parameters for reproducibility
    p0 = Parameters(mode='no-adv'); p0.validate(); p0.nondim()
    meta = {
        'study_type': 'No Advection — mu Sweep',
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
        'mu_factors': NoAdvMuSweepConfig.MU_FACTORS,
        'baselines': {
            'MU_DIM_NO_ADV': getattr(Parameters, 'MU_DIM_NO_ADV', None),
            'D_dim': p0.D_dim,
            'H_dim': p0.H_dim,
            'L_dim': p0.L_dim,
        },
        'parameters_defaults_used': True
    }
    with open(os.path.join(study_dir, "study_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved CSV: {csv_path}")
    print(f"⏱ Done in {time.time()-t0:.1f}s")

    # Plots
    plots_dir = os.path.join(study_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    create_heatmaps(df, plots_dir)

    return df

# ========================================================
# Heatmap helper tailored to this module
# ========================================================

def choose_colormap(vals: pd.Series):
    """Pick colormap and colour limits based on sign of values."""
    vals_clean = vals.dropna()
    if vals_clean.empty:
        return 'viridis', 0.0, 1.0

    vmin, vmax = float(vals_clean.min()), float(vals_clean.max())

    if vmin < 0 and vmax > 0:
        # Mixed sign -> diverging around 0
        limit = max(abs(vmin), abs(vmax))
        cmap = 'RdBu_r'
        return cmap, -limit, limit
    elif vmin >= 0:
        # All non-negative -> sequential white→red, anchored at 0
        cmap = 'Reds'
        return cmap, 0.0, vmax
    else:
        # All non-positive -> sequential white→blue, anchored at 0
        cmap = 'Blues_r'
        return cmap, vmin, 0.0

def create_heatmap(df: pd.DataFrame,col_name: str,title: str,cbar_label: str,
    filename_prefix: str,plots_dir: str,show_deviation: bool = False,reference_value: float = 1.0,
    symmetric: bool = False, annot: bool = True,fmt: str = ".3f",):
    """
    Create scatter-style geometry 'heatmaps' for this module.

    Expects columns:
      - 'mu_factor' (groups; one figure per mu)
      - 'width_mm', 'depth_mm' (x/y coordinates)
      - <col_name> as numeric metric to colour points
    """
    required = {'mu_factor', 'width_mm', 'depth_mm', col_name}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[create_heatmap] Missing columns: {missing}")
        return

    os.makedirs(plots_dir, exist_ok=True)
    set_style()

    mu_values = sorted(pd.to_numeric(df['mu_factor'], errors='coerce').dropna().unique())
    if not mu_values:
        print("[create_heatmap] No mu_factor values found.")
        return

    for mu in mu_values:
        dmu = df[df['mu_factor'] == mu].copy()
        if dmu.empty:
            continue

        vals = pd.to_numeric(dmu[col_name], errors='coerce')
        plot_vals = (vals - reference_value) if show_deviation else vals

        # Choose colour map limits
        cmap, vmin_plot, vmax_plot = choose_colormap(plot_vals)

        fig_w, fig_h = 6.5, 4.2
        fname = f"{filename_prefix}_mu_{str(mu).replace('.','p')}.png"
        out_path = os.path.join(plots_dir, fname)

        with safe_plot(out_path, (fig_w, fig_h)):
            fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

            sc = ax.scatter(
                dmu['width_mm'], dmu['depth_mm'],
                c=plot_vals, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                s=40, alpha=0.95, edgecolors='black', linewidth=0.5
            )

            ax.set_title(latexify_label(f"{title} ($\\mu={mu}$)"), fontweight='bold')
            ax.set_xlabel(latexify_label("Sulcus Width (mm)"), fontweight='bold')
            ax.set_ylabel(latexify_label("Sulcus Depth (mm)"), fontweight='bold')
            ax.grid(True, alpha=0.3)

            cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.056, pad=0.08)
            cbar.set_label(latexify_label(cbar_label), fontsize=Config.FONT_SIZE_LABEL, fontweight='bold')
            if show_deviation and (vmin_plot <= 0.0 <= vmax_plot):
                try:
                    cbar.ax.axhline(0.0, color='black', linestyle='--', linewidth=0.6)
                except Exception:
                    pass

            # Numeric annotations (optional)
            if annot:
                for _, r in dmu.iterrows():
                    try:
                        val = r[col_name]
                        txt = f"{val:{fmt}}" if np.isfinite(val) else ""
                    except Exception:
                        txt = ""
                    ax.text(r['width_mm'], r['depth_mm'] + 0.02, txt, ha='center', va='bottom',
                            fontsize=8, color='black',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

            plt.tight_layout()

        print(f"Created: {out_path}")

def create_mu_sweep_heatmaps(df: pd.DataFrame, plots_dir: str):
    """Produce CR and Flux Ratio panels (one figure per mu)."""
    create_heatmap(
        df, col_name='CR',
        title=r"Concentration Ratio $CR=\bar c_S/\bar c_R$",
        cbar_label="CR",
        filename_prefix="CR_panels",
        plots_dir=plots_dir,
        show_deviation=True, reference_value=1.0, symmetric=False,
        annot=True, fmt=".3f"
    )
    create_heatmap(
        df, col_name='flux_ratio',
        title=r"Net Flux Ratio (rect/sulc) at $y=0$/bottom",
        cbar_label="Flux Ratio",
        filename_prefix="FluxRatio_panels",
        plots_dir=plots_dir,
        show_deviation=False, reference_value=1.0, symmetric=False,
        annot=True, fmt=".2f"
    )

def create_heatmaps(df: pd.DataFrame, plots_dir: str):
    """Compatibility wrapper (called by run_no_adv_mu_sweep)."""
    print("\n📊 Creating heatmaps (no advection, mu sweep)...")
    create_mu_sweep_heatmaps(df, plots_dir)
    print(f"✓ Heatmaps saved to {plots_dir}")

# ========================================================
# Replotting from CSV
# ========================================================

def replot_from_csv(csv_path=None, output_base=None):
    """
    Load an existing CSV, compute any missing derived columns (CR, flux_ratio, flux_error_pct),
    and regenerate the heatmaps into <study_dir>/Plots.
    """
    if output_base is None:
        output_base = NoAdvMuSweepConfig.DEFAULT_OUTPUT_BASE
    if csv_path is None:
        csv_path = os.path.join(output_base, NoAdvMuSweepConfig.DEFAULT_CSV_NAME)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"\n↻ Replotting from CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Compute derived columns if missing
    if 'CR' not in df.columns:
        conc_s = pd.to_numeric(df.get('avg_conc_sulc'), errors='coerce')
        conc_r = pd.to_numeric(df.get('avg_conc_rect'), errors='coerce')
        df['CR'] = conc_s / conc_r

    if 'flux_ratio' not in df.columns:
        fs = pd.to_numeric(df.get('flux_sulc_y0'), errors='coerce')
        fr = pd.to_numeric(df.get('flux_rect_bottom'), errors='coerce')
        df['flux_ratio'] = fr / fs

    if 'flux_error_pct' not in df.columns:
        fs = pd.to_numeric(df.get('flux_sulc_y0'), errors='coerce')
        fr = pd.to_numeric(df.get('flux_rect_bottom'), errors='coerce')
        denom = fs.where(~np.isclose(fs.abs(), 0.0), 1.0)
        df['flux_error_pct'] = 100.0 * (fr - fs) / denom.abs()

    # Study dir and plots dir adjacent to CSV
    study_dir = os.path.dirname(csv_path)
    plots_dir = os.path.join(study_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    create_heatmaps(df, plots_dir)
    print("Replot complete.")
    return df

# ================
# Menu
# ================

def _interactive_menu():
    print("\nSelect an option:")
    print("  1) Run simulations and plot")
    print("  2) Replot from existing CSV")
    choice = input("Enter 1 or 2: ").strip()
    return choice

# ================
# Entry point
# ================

if __name__ == "__main__":
    try:
        choice = _interactive_menu()
        if choice == "1":
            run_no_adv_mu_sweep()
            print("\n✅ No-advection mu-sweep complete.")
        elif choice == "2":
            default_csv = os.path.join(
                NoAdvMuSweepConfig.DEFAULT_OUTPUT_BASE, NoAdvMuSweepConfig.DEFAULT_CSV_NAME
            )
            print(f"Default CSV: {default_csv}")
            csv_in = input("Path to CSV [press Enter to use default]: ").strip() or default_csv
            replot_from_csv(csv_in)
            print("\n✅ Replot complete.")
        else:
            print("No action taken (please choose 1 or 2).")
    except Exception as e:
        print(f"\n Failed: {e}")
        import traceback; traceback.print_exc()
