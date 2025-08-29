##########################################################
# Advection–Diffusion Module
##########################################################
"""
Validates a rectangular surrogate that uses a *step* $\mu$(x) over the mouth
against the sulcus simulation in the advection–diffusion regime.

Sweeps:
  Pe ∈ {0.1, 0.5, 1.0}
  $\mu$_factor ∈ {0.1, 1.0, 10}   (scales MU_DIM_BASE)

For each (Pe, $\mu$_factor):
  1) Run sulcus simulation (reference), record $\mu$_eff^{open}, baseline $\mu$
  2) Run rectangular surrogate with StepUptakeOpen:
        - baseline (outside mouth) $\mu$ = $\mu$_factor (non-dimensional)
        - over-mouth target = $\mu$_eff^{open}
  3) Save CSV, then plot:
        - 3x3 spatial grid of $\mu$(x) (step $\mu$(x) and baseline $\mu$ only)
        - 3x3 heatmap of relative flux error (%)
        - 3x3 heatmap of CR = c̄_S / c̄_R
"""

# ========================================================
# Imports
# ========================================================

import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Core project modules
from parameters import Parameters, StepUptakeOpen
from simulation import run_simulation
from plotting import safe_plot, set_style, Config

# ========================================================
# Configuration
# ========================================================

class AdvDiffValidationConfig:
    """Configuration for advection–diffusion validation study (step $\mu$ only)."""

    PE_VALUES = [0.1, 1.0, 10]
    MU_FACTORS = [0.1, 1.0, 10]

    REFERENCE_GEOMETRY = {
        'L_dim': 10.0,      # mm
        'H_dim': 1.0,       # mm
        'sulci_w_dim': 0.5, # mm
        'sulci_h_dim': 1.0, # mm
        'mesh_size_dim': 0.02,
        'refinement_factor': 1
    }

    D_DIM = 0.0003        # mm^2/s
    MU_DIM_BASE = 0.0003  # chosen so that $\mu$=1 after nondimensionalisation

    STEP_PARAMS = {
        'L_c': None,        # defaults to 10% of sulcus width if None
        'Gamma': 5.0,
        'degree': 2
    }

# ========================================================
# Helpers (parameters & metrics)
# ========================================================

def create_base_parameters(Pe_target, mu_factor):
    """Build Parameters for a target Pe and baseline $\mu$ factor."""
    cfg = AdvDiffValidationConfig
    U_ref_dim = Pe_target * cfg.D_DIM / cfg.REFERENCE_GEOMETRY['H_dim']

    params = Parameters(
        mode='adv-diff',
        U_ref_dim=U_ref_dim,
        D_dim=cfg.D_DIM,
        **cfg.REFERENCE_GEOMETRY
    )
    params.mu_dim = cfg.MU_DIM_BASE * float(mu_factor)  # makes non-dim $\mu$ = mu_factor
    return params

def extract_flux_data(results, domain_type):
    """Pull signed flux components."""
    flux_metrics = results.get('flux_metrics', {}) or {}
    if domain_type == 'sulcus':
        y0 = ((flux_metrics.get('sulcus_specific') or {})
              .get('physical_flux') or {}).get('y0_flux', {}) or {}
        return {
            'total_flux': y0.get('total', None),
            'diffusive_flux': y0.get('diffusive', None),
            'advective_flux': y0.get('advective', None),
            'uptake_flux': flux_metrics.get('uptake_flux', None)
        }
    else:
        bottom = (flux_metrics.get('physical_flux') or {}).get('bottom', {}) or {}
        return {
            'total_flux': bottom.get('total', None),
            'diffusive_flux': bottom.get('diffusive', None),
            'advective_flux': bottom.get('advective', None),
            'uptake_flux': flux_metrics.get('uptake_flux', None)
        }

# ========================================================
# Core runs
# ========================================================

def run_sulcus_reference(Pe_value, mu_factor):
    """Run the sulcus simulation and extract $\mu$_eff values for (Pe, $\mu$)."""
    print(f"\n{'='*50}\nSULCUS REFERENCE  (Pe={Pe_value}, $\mu$={mu_factor})\n{'='*50}")
    params = create_base_parameters(Pe_value, mu_factor)
    params.validate()
    params.nondim()

    print(f"Pe={params.Pe:.2f}  Re={params.Re:.3f} | L={params.L:.1f}  H={params.H:.1f}")
    print(f"Sulcus: w={params.sulci_w:.2f}  h={params.sulci_h:.2f}")
    print(f"Baseline $\mu$ (non-dim) = {params.mu:.4f}")

    config_name = f"Sulcus_Pe_{Pe_value:.1f}_mu_{mu_factor:.1f}".replace('.', 'p')
    results = run_simulation(
        mode='adv-diff',
        study_type='AdvDiff Step Validation',
        config_name=config_name,
        domain_type='sulcus',
        params=params
    )

    mu_eff = results.get('mu_eff_comparison', {}) or {}
    mu_eff_arc  = mu_eff.get('mu_eff_arc')
    mu_eff_sim  = mu_eff.get('mu_eff_sim')
    mu_eff_open = mu_eff.get('mu_eff_open')

    def _fmt(x): return f"{x:.6f}" if x is not None else "NA"
    print("$\mu$_eff values:",
          f"arc={_fmt(mu_eff_arc)}  sim={_fmt(mu_eff_sim)}  open={_fmt(mu_eff_open)}")

    return results, mu_eff_arc, mu_eff_sim, mu_eff_open

def run_rect_step_surrogate(Pe_value, mu_factor, mu_eff_open, sulcus_params):
    """Run the rectangular surrogate with *step* $\mu$(x) only."""
    print(f"\nRectangular step surrogate  (Pe={Pe_value}, $\mu$={mu_factor})")

    params = create_base_parameters(Pe_value, mu_factor)
    params.validate()
    params.nondim()

    # Mouth geometry (non-dimensional)
    sulcus_left_x  = params.L/2 - params.sulci_w/2
    sulcus_right_x = params.L/2 + params.sulci_w/2
    L_c = AdvDiffValidationConfig.STEP_PARAMS['L_c'] or (0.1 * params.sulci_w)

    mu_step = StepUptakeOpen(
        mu_base=float(mu_factor),          # outside-mouth baseline (non-dim)
        mu_eff_target=float(mu_eff_open),  # mouth target (non-dim)
        sulcus_left_x=sulcus_left_x,
        sulcus_right_x=sulcus_right_x,
        L_c=L_c,
        Gamma=AdvDiffValidationConfig.STEP_PARAMS['Gamma'],
        degree=AdvDiffValidationConfig.STEP_PARAMS['degree']
    )
    params.mu = mu_step
    params.mu_dim = mu_step  # tracked expression

    config_name = f"Rect_step_open_Pe_{Pe_value:.1f}_mu_{mu_factor:.1f}".replace('.', 'p')
    results = run_simulation(
        mode='adv-diff',
        study_type='AdvDiff Step Validation',
        config_name=config_name,
        domain_type='rectangular',
        params=params,
        mu_variable=True  # IMPORTANT: ensure run_simulation forwards this flag
    )
    return results

# ========================================================
# Main driver
# ========================================================

def run_advdiff_step_validation(output_base_dir="Results/AdvDiff Validation (Pe x mu) - Step Only"):
    """
    Run the Pe x $\mu$ study using only the *step $\mu$(x)* rectangular surrogate.
    Returns the assembled results DataFrame and writes a CSV.
    """
    print(f"\n{'='*64}\nADVECTION–DIFFUSION VALIDATION (Step $\mu$ only)\n{'='*64}")
    t0 = time.time()

    # Folders
    os.makedirs(output_base_dir, exist_ok=True)
    results_dir = os.path.join(output_base_dir, "Results Data")
    plots_dir   = os.path.join(output_base_dir, "Analysis Plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    rows = []

    for Pe in AdvDiffValidationConfig.PE_VALUES:
        print(f"\n{'='*40}\nPROCESSING Pe = {Pe}\n{'='*40}")
        for mu_factor in AdvDiffValidationConfig.MU_FACTORS:

            print(f"\n--- $\mu$ factor = {mu_factor} ---")

            # 1) Sulcus reference
            sulc_res, mu_eff_arc, mu_eff_sim, mu_eff_open = run_sulcus_reference(Pe, mu_factor)
            sulc_flux = extract_flux_data(sulc_res, 'sulcus')

            mmS = sulc_res.get('mass_metrics', {})
            avg_sulcus = mmS.get('average_concentration', {}).get('total')

            # sulcus row
            rows.append({
                'Pe': Pe,
                'mu_factor': mu_factor,
                'domain_type': 'sulcus',
                'surrogate_type': 'reference',
                'total_flux': sulc_flux['total_flux'],
                'diffusive_flux': sulc_flux['diffusive_flux'],
                'advective_flux': sulc_flux['advective_flux'],
                'uptake_flux': sulc_flux['uptake_flux'],
                'mu_eff_arc': mu_eff_arc,
                'mu_eff_sim': mu_eff_sim,
                'mu_eff_open': mu_eff_open,
                'avg_conc': avg_sulcus,
                'CR': np.nan,
                # for spatial $\mu$(x) plotting:
                'Mu_base_nondim': sulc_res['params'].mu,
                'Domain_Length_mm': sulc_res['params'].L_dim,
                'Sulcus_Width_mm': sulc_res['params'].sulci_w_dim,
            })

            if mu_eff_open is None:
                print("⚠ No $\mu$_eff_open computed; skipping rectangular surrogate.")
                continue

            # 2) Rectangular step surrogate
            rect_res = run_rect_step_surrogate(Pe, mu_factor, mu_eff_open, sulc_res['params'])
            rect_flux = extract_flux_data(rect_res, 'rectangular')

            mmR = rect_res.get('mass_metrics', {}) or {}
            avg_rect = mmR.get('average_concentration')

            rows.append({
                'Pe': Pe,
                'mu_factor': mu_factor,
                'domain_type': 'rectangular',
                'surrogate_type': 'step_open',
                'total_flux': rect_flux['total_flux'],
                'diffusive_flux': rect_flux['diffusive_flux'],
                'advective_flux': rect_flux['advective_flux'],
                'uptake_flux': rect_flux['uptake_flux'],
                'mu_eff_arc': mu_eff_arc,
                'mu_eff_sim': mu_eff_sim,
                'mu_eff_open': mu_eff_open,
                'avg_conc': avg_rect,
                'CR': (avg_sulcus / avg_rect) if (avg_sulcus is not None and avg_rect not in (None, 0.0)) else np.nan
            })

    # Assemble DataFrame and compute errors
    df = pd.DataFrame(rows).sort_values(['Pe','mu_factor','domain_type']).reset_index(drop=True)

    # Relative flux error (%) for step surrogate vs sulcus per (Pe, $\mu$)
    df['flux_error_pct'] = np.nan
    df['flux_ratio'] = np.nan

    for Pe in AdvDiffValidationConfig.PE_VALUES:
        for mu in AdvDiffValidationConfig.MU_FACTORS:
            ref_mask = (df['Pe']==Pe) & (df['mu_factor']==mu) & (df['domain_type']=='sulcus')
            rec_mask = (df['Pe']==Pe) & (df['mu_factor']==mu) & (df['domain_type']=='rectangular') & (df['surrogate_type']=='step_open')
            if not ref_mask.any() or not rec_mask.any():
                continue
            ref_flux = df.loc[ref_mask, 'total_flux'].iloc[0]
            df.loc[rec_mask, 'flux_ratio'] = df.loc[rec_mask, 'total_flux'] / (ref_flux if ref_flux != 0 else 1.0)
            df.loc[rec_mask, 'flux_error_pct'] = 100.0 * (df.loc[rec_mask, 'total_flux'] - ref_flux) / (abs(ref_flux) if ref_flux != 0 else 1.0)

    # Save CSV + metadata
    csv_path = os.path.join(results_dir, "advdiff_validation_step_pe_x_mu.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")

    metadata = {
        'study_type': 'AdvDiff Validation (Pe x $\mu$) - Step $\mu$ only',
        'timestamp': datetime.now().isoformat(),
        'Pe_values': AdvDiffValidationConfig.PE_VALUES,
        'mu_factors': AdvDiffValidationConfig.MU_FACTORS,
        'reference_geometry': AdvDiffValidationConfig.REFERENCE_GEOMETRY,
        'parameters': {
            'D_dim': AdvDiffValidationConfig.D_DIM,
            'mu_dim_base': AdvDiffValidationConfig.MU_DIM_BASE
        }
    }
    with open(os.path.join(results_dir, "study_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    # Plots
    create_validation_plots(df, plots_dir)

    print(f"\n{'='*64}\nCOMPLETE in {time.time()-t0:.1f}s\nResults dir: {output_base_dir}\nCSV: {csv_path}")
    return df

# ========================================================
# Plotting
# ========================================================

def create_mu_eff_spatial_plots_grid(df, plots_dir, zoom='mouth', x_margin_mm=0.5):
    """
    3x3 grid (rows=mu_factor, cols=Pe) showing:
      - step mu(x) along y=0,
      - baseline mu (outside mouth),
    plus shaded mouth region.

    Updates:
      X-tick labels are shown on all subplots (not just the bottom row).
      Y-limits are set per row using that row's min/max values across columns.
    """
    set_style()
    pe_vals = sorted(df['Pe'].unique().tolist())
    mu_vals = sorted(df['mu_factor'].unique().tolist())

    # Use geometry from sulcus rows
    sulc = df[(df['domain_type'] == 'sulcus') & (df['surrogate_type'] == 'reference')]
    if sulc.empty:
        print('No sulcus reference rows found; skipping mu(x) grid.')
        return

    row0 = sulc.iloc[0]
    L_dim = float(row0['Domain_Length_mm'])
    H_dim = 1.0
    L_ref = H_dim
    L = L_dim / L_ref

    w_dim = float(row0['Sulcus_Width_mm'])
    left_dim = 0.5 * (L_dim - w_dim)
    right_dim = left_dim + w_dim

    left = left_dim / L_ref
    right = right_dim / L_ref
    w = w_dim / L_ref

    # x sampling
    x_nd = np.linspace(0.0, L, 1500)

    # zoom window
    if zoom == 'mouth':
        x_min_mm = max(0.0, left_dim - x_margin_mm)
        x_max_mm = min(L_dim, right_dim + x_margin_mm)
    else:
        x_min_mm, x_max_mm = 0.0, L_dim

    # figure
    n_rows, n_cols = len(mu_vals), len(pe_vals)
    fig_w = 2.4 * n_cols
    fig_h = 2.0 * n_rows + 0.6
    out_path = os.path.join(plots_dir, 'mu_eff_spatial_grid.png')

    with safe_plot(out_path, figsize=(fig_w, fig_h)):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                 squeeze=False, sharex=True, sharey=False)

        fig.suptitle(r'$\mu_{\mathrm{eff}}$ Spatial Distribution: Step profile and Baseline',
                     fontsize=Config.FONT_SIZE_TITLE + 2, fontweight='bold', y=1.02)

        handles_global, labels_global = None, None

        # Track per-row limits and axes to set y-lims after each row is filled
        row_axes = [[] for _ in range(n_rows)]
        row_ymins = [np.inf] * n_rows
        row_ymaxs = [-np.inf] * n_rows

        for r, mu in enumerate(mu_vals):
            for c, Pe in enumerate(pe_vals):
                ax = axes[r, c]
                S = sulc[(np.isclose(sulc['Pe'], Pe)) & (np.isclose(sulc['mu_factor'], mu))]
                if S.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    # Still ensure cosmetics and ticks
                    ax.set_xlim(x_min_mm, x_max_mm)
                    ax.tick_params(axis='x', which='both', labelbottom=True)  # force x-tick labels
                    ax.grid(True, alpha=0.3)
                    row_axes[r].append(ax)
                    continue

                srow = S.iloc[0]
                mu_base = float(srow['Mu_base_nondim'])
                mu_eff_open = float(srow['mu_eff_open'])

                # build step profile for this regime
                step_fn = StepUptakeOpen(
                    mu_base=mu_base,
                    mu_eff_target=mu_eff_open,
                    sulcus_left_x=left,
                    sulcus_right_x=right,
                    L_c=0.1 * w,
                    Gamma=5.0,
                    degree=2
                )
                vals = np.empty_like(x_nd)
                buf = [0.0]
                for i, x in enumerate(x_nd):
                    step_fn.eval(buf, [float(x), 0.0])
                    vals[i] = buf[0]

                # draw: step and baseline only
                ax.plot(x_nd * L_ref, vals, linewidth=1.4, alpha=0.95, label=r'Step $\mu(x)$')
                ax.plot([x_nd[0] * L_ref, x_nd[-1] * L_ref], [mu_base, mu_base],
                        linestyle=':', linewidth=1.6, alpha=0.95, label='Baseline $\mu$')

                ax.axvspan(left_dim, right_dim, alpha=0.15, color='gray')

                if c == 0:
                    ax.set_ylabel(rf'$\mu={mu:.1f}$', fontweight='bold')
                if r == 0:
                    ax.set_title(rf'Pe = {Pe}', fontsize=Config.FONT_SIZE_TITLE, fontweight='bold')

                ax.grid(True, alpha=0.3)
                ax.set_xlim(x_min_mm, x_max_mm)
                ax.set_xlabel('x-coordinate (mm)', fontweight='bold')

                # force x-tick labels to be visible on all rows
                ax.tick_params(axis='x', which='both', labelbottom=True)

                # update row-wise y-limits
                y_all = np.r_[vals, [mu_base]]
                row_ymins[r] = min(row_ymins[r], float(np.min(y_all)))
                row_ymaxs[r] = max(row_ymaxs[r], float(np.max(y_all)))

                # capture legend handles from any populated axes
                handles_global, labels_global = ax.get_legend_handles_labels()

                row_axes[r].append(ax)

            # after finishing this row, set uniform y-lims across its columns
            y_min, y_max = row_ymins[r], row_ymaxs[r]
            span = max(1e-12, y_max - y_min)
            pad = 0.05 * span
            for ax in row_axes[r]:
                ax.set_ylim(y_min - pad, y_max + pad)

        if handles_global:
            fig.legend(handles_global, labels_global, loc='lower center',
                       ncol=len(labels_global), fontsize=Config.FONT_SIZE_LEGEND+2,
                       frameon=True, fancybox=True, framealpha=0.55, facecolor='lightgray', bbox_to_anchor=(0.5, -0.05))
            plt.subplots_adjust(bottom=0.14, left=0.08, right=0.98, top=0.92,
                                wspace=0.35, hspace=0.45)

    print(f"Saved: {out_path}")

def _side_by_side_heatmaps(df, plots_dir):
    """
    Side-by-side square heatmaps with Matplotlib only:
      - Left: Relative flux error (%) for step μ(x)
      - Right: Concentration ratio deviation (c̄_S/c̄_R - 1)
    """
    os.makedirs(plots_dir, exist_ok=True)

    # --- Pivot tables
    d_rect = df[(df['domain_type']=='rectangular') & (df['surrogate_type']=='step_open')]
    if d_rect.empty:
        print("No rectangular step rows; skipping side-by-side heatmaps.")
        return

    pivot_err = (
        d_rect.pivot_table(index='mu_factor', columns='Pe',
                           values='flux_error_pct', aggfunc='mean')
             .sort_index()
    )

    S = df[(df['domain_type']=='sulcus') & (df['surrogate_type']=='reference')]
    if S.empty:
        print("Missing sulcus rows; skipping CR deviation heatmap.")
        return

    S = S[['Pe','mu_factor','avg_conc']].rename(columns={'avg_conc':'avg_conc_sulcus'})
    d = d_rect.merge(S, on=['Pe','mu_factor'], how='left')
    d['CR_dev'] = (d['avg_conc_sulcus'] / d['avg_conc']) - 1.0

    pivot_cr = (
        d.pivot_table(index='mu_factor', columns='Pe',
                      values='CR_dev', aggfunc='mean')
         .sort_index()
    )

    # --- Color limits symmetric around 0 (with guard)
    def sym_lim(piv, eps=1e-12):
        arr = np.asarray(piv.to_numpy(), dtype=float)
        m = np.nanmax(np.abs(arr)) if arr.size else 0.0
        m = max(m, eps)
        return -m, m

    vmin_err, vmax_err = sym_lim(pivot_err)
    vmin_cr,  vmax_cr  = sym_lim(pivot_cr)

    fig_w, fig_h = 8.4, 4.6
    out = os.path.join(plots_dir, "comparison_heatmaps_step_open.png")

    with safe_plot(out, figsize=(fig_w, fig_h)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

        # ---------- Left: flux error heatmap ----------
        arr_err = pivot_err.to_numpy()
        im1 = ax1.imshow(arr_err, cmap="RdBu_r",
                         vmin=vmin_err, vmax=vmax_err,
                         aspect="equal", origin="lower")

        # grid
        ax1.set_xticks(np.arange(-.5, arr_err.shape[1], 1), minor=True)
        ax1.set_yticks(np.arange(-.5, arr_err.shape[0], 1), minor=True)
        ax1.grid(which="minor", color="lightgrey", linestyle='-', linewidth=0.5)
        ax1.tick_params(which="minor", bottom=False, left=False)

        ax1.set_title(r"Relative $y=0$ Flux Error for Step $\mu(x)$",
                      fontsize=Config.FONT_SIZE_TITLE, fontweight="bold")
        ax1.set_xlabel("Pe"); ax1.set_ylabel(r"$\mu$")

        ax1.set_xticks(np.arange(arr_err.shape[1]))
        ax1.set_xticklabels(list(pivot_err.columns))
        ax1.set_yticks(np.arange(arr_err.shape[0]))
        ax1.set_yticklabels(list(pivot_err.index))

        # axes bounds align to cells
        ax1.set_xlim(-0.5, arr_err.shape[1]-0.5)
        ax1.set_ylim(-0.5, arr_err.shape[0]-0.5)

        # annotate (skip NaNs), with soft grey box
        for i in range(arr_err.shape[0]):
            for j in range(arr_err.shape[1]):
                val = arr_err[i, j]
                if np.isnan(val):
                    continue
                ax1.text(j, i, f"{val:.2f}%",
                         ha='center', va='center', color='black',
                         bbox=dict(boxstyle='round,pad=0.25',
                                   facecolor='lightgrey', alpha=0.6, edgecolor='none'))

        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # ---------- Right: CR deviation heatmap ----------
        arr_cr = pivot_cr.to_numpy()
        im2 = ax2.imshow(arr_cr, cmap="RdBu_r",
                         vmin=vmin_cr, vmax=vmax_cr,
                         aspect="equal", origin="lower")

        ax2.set_xticks(np.arange(-.5, arr_cr.shape[1], 1), minor=True)
        ax2.set_yticks(np.arange(-.5, arr_cr.shape[0], 1), minor=True)
        ax2.grid(which="minor", color="lightgrey", linestyle='-', linewidth=0.5)
        ax2.tick_params(which="minor", bottom=False, left=False)

        ax2.set_title(r"Concentration Ratio Deviation $(\bar c_S/\bar c_R - 1)$",
                      fontsize=Config.FONT_SIZE_TITLE, fontweight="bold")
        ax2.set_xlabel("Pe"); ax2.set_ylabel(r"$\mu$")

        ax2.set_xticks(np.arange(arr_cr.shape[1]))
        ax2.set_xticklabels(list(pivot_cr.columns))
        ax2.set_yticks(np.arange(arr_cr.shape[0]))
        ax2.set_yticklabels(list(pivot_cr.index))

        ax2.set_xlim(-0.5, arr_cr.shape[1]-0.5)
        ax2.set_ylim(-0.5, arr_cr.shape[0]-0.5)

        for i in range(arr_cr.shape[0]):
            for j in range(arr_cr.shape[1]):
                val = arr_cr[i, j]
                if np.isnan(val):
                    continue
                # show as percent with sign
                ax2.text(j, i, f"{val*100:+.2f}%",
                         ha='center', va='center', color='black',
                         bbox=dict(boxstyle='round,pad=0.25',
                                   facecolor='lightgrey', alpha=0.6, edgecolor='none'))

        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax1.grid(False)
        ax2.grid(False)

        fig.suptitle(r"Flux Error and Concentration Ratio Comparison",
                    fontsize=Config.FONT_SIZE_TITLE+2, fontweight="bold", y=0.96)

        plt.subplots_adjust(wspace=0.35, left=0.08, right=0.98, top=0.90, bottom=0.12)


    print(f"Saved: {out}")

def create_validation_plots(df, plots_dir):
    """
    Generate (step mu only)
      - 3x3 $\mu$(x) spatial grid,
      - 3x3 flux-error heatmap,
      - 3x3 concentration-ratio heatmap.
    """
    print("\n📊 Creating validation plots...")
    os.makedirs(plots_dir, exist_ok=True)
    create_mu_eff_spatial_plots_grid(df, plots_dir, zoom='mouth', x_margin_mm=0.5)
    _side_by_side_heatmaps(df, plots_dir)
    print(f"✓ Validation plots saved to {plots_dir}")

# ========================================================
# Re-plotting from CSV (no simulations)
# ========================================================

def replot_from_csv():
    """Reload results CSV and regenerate all validation plots (no simulations)."""
    output_base_dir = "Results/AdvDiff Validation (Pe x mu) - Step Only"
    results_dir = os.path.join(output_base_dir, "Results Data")
    plots_dir   = os.path.join(output_base_dir, "Analysis Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Use the exact path we saved to in run_advdiff_step_validation()
    csv_path = os.path.join(results_dir, "advdiff_validation_step_pe_x_mu.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Results CSV not found at:\n  {csv_path}\n"
            "Run option 1 first (simulations + plots), or place the CSV there."
        )

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Recompute flux_error_pct/flux_ratio if missing (robustness)
    if ('flux_error_pct' not in df.columns) or df['flux_error_pct'].isna().all():
        df['flux_error_pct'] = np.nan
        df['flux_ratio'] = np.nan
        for Pe in sorted(df['Pe'].unique()):
            for mu in sorted(df['mu_factor'].unique()):
                ref_mask = (df['Pe']==Pe) & (df['mu_factor']==mu) & (df['domain_type']=='sulcus')
                rec_mask = (df['Pe']==Pe) & (df['mu_factor']==mu) & \
                           (df['domain_type']=='rectangular') & (df['surrogate_type']=='step_open')
                if not ref_mask.any() or not rec_mask.any():
                    continue
                ref_flux = df.loc[ref_mask, 'total_flux'].iloc[0]
                denom = (abs(ref_flux) if ref_flux != 0 else 1.0)
                df.loc[rec_mask, 'flux_ratio'] = df.loc[rec_mask, 'total_flux'] / (ref_flux if ref_flux != 0 else 1.0)
                df.loc[rec_mask, 'flux_error_pct'] = 100.0 * (df.loc[rec_mask, 'total_flux'] - ref_flux) / denom

    create_validation_plots(df, plots_dir)
    print(f"✓ Re-plotted from CSV to: {plots_dir}")

# ========================================================
# Entry point
# ========================================================

if __name__ == "__main__":
    print("="*64)
    print("ADVECTION–DIFFUSION VALIDATION (Pe x $\mu$) - Step $\mu$(x) only")
    print("="*64)

    try:
        print("\nChoose an option:")
        print("  1) Run full validation (simulations + plots)")
        print("  2) Re-plot from existing CSV")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "2":
            replot_from_csv()

        elif choice == "1":
            results_df = run_advdiff_step_validation()

            # Optional brief console summary
            print("\n" + "="*40)
            print("SUMMARY STATISTICS (step $\mu$ only)")
            print("="*40)
            for mu in AdvDiffValidationConfig.MU_FACTORS:
                print(f"\n$\mu$ factor = {mu}:")
                for Pe in AdvDiffValidationConfig.PE_VALUES:
                    ref = results_df[(results_df['Pe']==Pe) & (results_df['mu_factor']==mu) &
                                     (results_df['domain_type']=='sulcus')]
                    rect = results_df[(results_df['Pe']==Pe) & (results_df['mu_factor']==mu) &
                                      (results_df['domain_type']=='rectangular') &
                                      (results_df['surrogate_type']=='step_open')]
                    if ref.empty or rect.empty:
                        continue
                    ref_flux = ref['total_flux'].iloc[0]
                    err = rect['flux_error_pct'].iloc[0]
                    CR = (ref['avg_conc'].iloc[0] / rect['avg_conc'].iloc[0]
                          if rect['avg_conc'].iloc[0] else np.nan)
                    print(f"  Pe={Pe:>4}: Flux ref={ref_flux:.6e} | step_open error={err:+6.2f}% | CR={CR:.3f}")

            print("\n✅ Step $\mu$(x) validation completed successfully!")

        else:
            print("Invalid choice. Exiting.")

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


