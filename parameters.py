##########################################################
# Parameters Module
##########################################################
"""
Default parameters and domain geometry configurations
Supports both advection-diffusion and pure diffusion problems.
"""
# ========================================================
# region Inputs
# ========================================================

import warnings
import numpy as np

from typing import Dict, List
from dolfin import UserExpression

#endregion

# ========================================================
# region Step Function Expression Classes
# ========================================================

class StepUptakeFunction(UserExpression):
    """
    S2: Smoothed step function with smooth transitions at sulcus boundaries.

    Average-preserving mode:
      - If `mu_eff_target` and `L` are provided, the class rescales the opening
        value internally so that the length-average over [0, L] equals mu_eff_target.
      - Otherwise, if `mu_open` is provided, the class behaves like your original.

    Args (ND coordinates):
        mu_base:          baseline μ outside the opening (float or Constant-like)
        sulcus_left_x:    left x of opening
        sulcus_right_x:   right x of opening
        L:                total bottom length (required for average-preserving mode)
        mu_open:          opening μ (used if mu_eff_target is None)
        mu_eff_target:    target global mean μ across [0, L] (activates avg-preserving)
        L_c:              smoothing half-width near edges (default 10% of opening width)
        Gamma:            sigmoid steepness for smoothing
        quad_pts:         quadrature points for ⟨α⟩ computation
        degree:           FEniCS expression degree
    """

    def __init__(self,
                 mu_base,
                 sulcus_left_x,
                 sulcus_right_x,
                 L=None,
                 mu_open=None,
                 mu_eff_target=None,
                 L_c=None,
                 Gamma=5.0,
                 quad_pts=256,
                 **kwargs):

        super().__init__(**kwargs)

        # Store geometry
        self.xL = float(sulcus_left_x)
        self.xR = float(sulcus_right_x)
        self.w  = float(self.xR - self.xL)
        if self.w <= 0:
            raise ValueError(f"sulcus_right_x must be > sulcus_left_x (got w={self.w})")

        # Scalars
        self.mu_base = float(mu_base)
        self.Gamma   = float(Gamma)
        self.quad_pts = int(quad_pts)

        # Smoothing length (cap to < half-width)
        self.L_c = float(L_c) if L_c is not None else 0.1 * self.w
        self.L_c = max(0.0, min(self.L_c, 0.49 * self.w))

        # Average-preserving or legacy?
        self.mu_eff_target = None if mu_eff_target is None else float(mu_eff_target)
        self.L = None if L is None else float(L)

        if self.mu_eff_target is not None:
            if self.L is None or self.L <= 0 or self.w > self.L:
                raise ValueError(f"Average-preserving mode needs valid L: L={self.L}, w={self.w}")

            # Compute mean smoothing weight ⟨α⟩ over opening
            alpha_bar = self._compute_alpha_mean()

            # Solve for opening value that preserves global average:
            # global mean = (w/L)*[ μ_base*(1-⟨α⟩) + μ_open_eff*⟨α⟩ ] + ((L-w)/L)*μ_base = μ_eff_target
            if alpha_bar <= 1e-14:
                # Degenerate: fall back to sharp-step formula
                mu_open_eff = (self.L / self.w) * self.mu_eff_target - ((self.L - self.w) / self.w) * self.mu_base
            else:
                mu_open_eff = (
                    self.mu_base +
                    (self.L / (self.w * alpha_bar)) *
                    (self.mu_eff_target - self.mu_base * (1.0 - self.w / self.L))
                )

            # Clamp to non-negative for safety
            self.mu_open = max(0.0, float(mu_open_eff))

        else:
            # Legacy behaviour: use provided mu_open directly
            if mu_open is None:
                raise ValueError("Provide either mu_eff_target (with L) or mu_open.")
            self.mu_open = float(mu_open)

    # --------------------------
    # Internals for smoothing
    # --------------------------
    def _alpha_at(self, x):
        """Smoothing weight α(x) ∈ [0,1] inside the opening; 0 outside."""
        if x < self.xL or x > self.xR:
            return 0.0
        if self.L_c <= 0.0:
            return 1.0
        d = min(x - self.xL, self.xR - x)
        if d >= self.L_c:
            return 1.0
        z = d / self.L_c
        return 1.0 / (1.0 + np.exp(-self.Gamma * (z - 0.5)))

    def _compute_alpha_mean(self):
        """⟨α⟩ over [xL, xR] via Gauss–Legendre quadrature."""
        xi, wi = np.polynomial.legendre.leggauss(self.quad_pts)
        xs = 0.5*(self.xR - self.xL)*xi + 0.5*(self.xR + self.xL)
        ws = 0.5*(self.xR - self.xL)*wi
        vals = np.array([self._alpha_at(x) for x in xs], dtype=float)
        integral = float(np.dot(vals, ws))
        return integral / self.w

    # --------------------------
    # FEniCS API
    # --------------------------
    def eval(self, values, x):
        """
        Evaluate μ(x):
          Outside opening: μ_base
          Inside opening:  (1-α) μ_base + α μ_open
        """
        xx = x[0]
        if xx < self.xL or xx > self.xR:
            values[0] = self.mu_base
        else:
            alpha = self._alpha_at(xx)
            values[0] = (1.0 - alpha) * self.mu_base + alpha * self.mu_open

    def value_shape(self):
        return ()

import numpy as np
from dolfin import UserExpression

class StepUptakeOpen(UserExpression):
    """
    Smoothed step Robin coefficient μ(x) on y=0 with a single sulcus opening.

    Behaviour:
      - Outside the opening [xL, xR]: μ(x) = mu_base
      - Inside the opening:           μ(x) = mu_open
      - Optional smooth transition near the edges using a logistic ramp.

    Args (all non-dimensional):
        mu_base         : baseline μ on the flat wall (float-like)
        mu_eff_target   : opening μ to apply over the mouth (used directly)
        sulcus_left_x   : left x-coordinate of the mouth (xL)
        sulcus_right_x  : right x-coordinate of the mouth (xR)
        L_c             : smoothing half-width near each edge (default 10% of mouth width).
                          Set L_c=0 for a sharp step.
        Gamma           : steepness of the logistic ramp (larger = steeper)
        degree          : FEniCS expression polynomial degree
    """

    def __init__(self,
                 mu_base,
                 mu_eff_target,
                 sulcus_left_x,
                 sulcus_right_x,
                 L_c=None,
                 Gamma=5.0,
                 **kwargs):
        super().__init__(**kwargs)

        # Geometry
        self.xL = float(sulcus_left_x)
        self.xR = float(sulcus_right_x)
        self.w  = float(self.xR - self.xL)
        if self.w <= 0:
            raise ValueError(f"sulcus_right_x must be > sulcus_left_x (got w={self.w})")

        # Parameters
        self.mu_base = float(mu_base)
        self.mu_open = float(mu_eff_target)  # used directly as the mouth value
        self.Gamma   = float(Gamma)

        # Smoothing width (cap below half-width)
        if L_c is None:
            L_c = 0.1 * self.w
        self.L_c = max(0.0, min(float(L_c), 0.49 * self.w))

    # ---------- Internal: edge smoothing weight ----------
    def _alpha_at(self, x):
        """
        α(x) ∈ [0,1] inside the mouth; 0 outside.
        α=1 in the interior of the mouth; ramps from 0→1 over L_c near edges.
        """
        if x < self.xL or x > self.xR:
            return 0.0
        if self.L_c <= 0.0:
            return 1.0
        d = min(x - self.xL, self.xR - x)   # distance to nearest mouth edge
        if d >= self.L_c:
            return 1.0
        z = d / self.L_c                     # 0 at the edge, 1 at the end of ramp
        # Logistic centred at z=0.5; maps [0,1] → (0,1)
        return 1.0 / (1.0 + np.exp(-self.Gamma * (z - 0.5)))

    # ---------- FEniCS API ----------
    def eval(self, values, x):
        xx = x[0]
        if xx < self.xL or xx > self.xR:
            values[0] = self.mu_base
        else:
            alpha = self._alpha_at(xx)
            # Blend to avoid discontinuity at edges
            values[0] = (1.0 - alpha) * self.mu_base + alpha * self.mu_open

    def value_shape(self):
        return ()

# endregion

# ========================================================
# region Parameters Class
# ========================================================

class Parameters:
    """Holds all user-defined parameters, validates these and computes non-dimensionalised values."""

    # Class constants for mu values
    MU_DIM_ADV_DIFF = 0.0003  # Gives μ = 1
    MU_DIM_NO_ADV = 0.0003   # Gives μ = 1
    MU_DIM_NO_UPTAKE = 0     # No uptake

    # Valid modes
    VALID_MODES = {'adv-diff', 'no-adv', 'no-uptake'}

    # Fluid properties (dimensionless)
    VISCOSITY = 1.0
    RHO = 1.0

    def __init__(self, mode='adv-diff',
                 L_dim=10.0,
                 H_dim=1.0,
                 sulci_n=1,
                 sulci_w_dim=0.5,
                 sulci_h_dim=1.0,
                 mesh_size_dim=0.02,
                 refinement_factor=1,
                 U_ref_dim=0.012,
                 D_dim=0.0003):

        """Initialise parameters. Call validate() and nondim() afterwards."""
        # 0.015
        # Validate mode first
        if mode not in self.VALID_MODES:
            raise ValueError(f"Mode must be one of {self.VALID_MODES}, got '{mode}'")

        # Store dimensional parameters
        self.mode = mode
        self.L_dim = L_dim
        self.H_dim = H_dim
        self.sulci_n = sulci_n
        self.sulci_w_dim = sulci_w_dim
        self.sulci_h_dim = sulci_h_dim
        self.mesh_size_dim = mesh_size_dim
        self.refinement_factor = refinement_factor
        self.U_ref_dim = U_ref_dim
        self.D_dim = D_dim

        # Set mu_dim based on mode
        mode_mu_map = {
            'adv-diff': self.MU_DIM_ADV_DIFF,
            'no-adv': self.MU_DIM_NO_ADV,
            'no-uptake': self.MU_DIM_NO_UPTAKE
        }
        self.mu_dim = mode_mu_map[mode]

    def validate(self):
        """Validate input parameters for the simulation"""

        # Domain validation
        self._validate_positive(self.L_dim, 'Domain length')
        self._validate_positive(self.H_dim, 'Domain height')

        # Sulcus parameters must be non-negative
        self._validate_non_negative(self.sulci_n, 'Number of sulci')
        self._validate_non_negative(self.sulci_h_dim, 'Sulcus height')
        self._validate_non_negative(self.sulci_w_dim, 'Sulci width')

        # If sulcus exists, dimensions must be positive
        if self.sulci_n > 0:
            self._validate_positive(self.sulci_h_dim, 'Sulcus height (when sulci defined)')
            self._validate_positive(self.sulci_w_dim, 'Sulcus width (when sulci defined)')
            if self.sulci_w_dim * self.sulci_n >= self.L_dim:
                raise ValueError("Total sulcus width must be less than domain length.")

        # Mesh validation
        self._validate_positive(self.mesh_size_dim, 'Mesh size')
        if not isinstance(self.refinement_factor, int) or self.refinement_factor < 1:
            raise ValueError("Refinement factor must be an integer ≥ 1.")

        # Mesh size warnings
        min_dim = min(self.L_dim, self.H_dim)
        if self.mesh_size_dim > min_dim / 10:
            warnings.warn(f"Mesh size ({self.mesh_size_dim}) is large relative to domain.")
        if self.mesh_size_dim < min_dim / 1000:
            warnings.warn(f"Mesh size ({self.mesh_size_dim}) is very small - may be slow.")

        # PDE parameter validation
        if self.mode in ['adv-diff', 'no-uptake']:
            self._validate_non_negative(self.U_ref_dim, 'Reference velocity')

        self._validate_non_negative(self.D_dim, 'Diffusion coefficient')
        if self.mode == 'no-adv' and self.D_dim <= 0:
            raise ValueError("Diffusion coefficient must be > 0 for diffusion-only mode.")

        # Uptake validation
        if self.mode == 'no-uptake' and self.mu_dim != 0:
            warnings.warn("Setting mu to 0 for no-uptake mode.")
            self.mu_dim = 0
        elif self.mode != 'no-uptake':
            self._validate_non_negative(self.mu_dim, 'Uptake parameter')

    def _validate_positive(self, value, name):
        """Helper to validate positive values."""
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")

    def _validate_non_negative(self, value, name):
        """Helper to validate non-negative values."""
        if value < 0:
            raise ValueError(f"{name} cannot be negative, got {value}")

    def nondim(self):
        """Convert to dimensionless parameters."""

        # Using domain height as the length scale
        self.L_ref = self.H_dim

        # Geometry in dimensionless units
        self.L = self.L_dim / self.L_ref
        self.H = self.H_dim / self.L_ref
        self.sulci_h = self.sulci_h_dim / self.L_ref
        self.sulci_w = self.sulci_w_dim / self.L_ref
        self.mesh_size = self.mesh_size_dim / self.L_ref

        if self.mode in ['adv-diff', 'no-uptake']:
            # Advection-diffusion case
            self.Pe = (self.U_ref_dim * self.H_dim) / self.D_dim
            self.D = 1.0 / self.Pe
            self.Re = (self.RHO * self.U_ref_dim * self.L_ref) / self.VISCOSITY
            self.mu = self.mu_dim * self.H_dim / self.D_dim
            self.U_ref = 1.0
        else:
            # Diffusion-only case
            self.D = 1.0
            self.mu = self.mu_dim * self.H_dim / self.D_dim
            self.U_ref = 0.0
            self.Pe = None
            self.Re = None

    def __str__(self):
        """String representation for logging and debugging."""
        lines = [f"Simulation Parameters ({self.mode.title()} Mode):"]
        lines.append(f"  Domain: L={self.L_dim}×H={self.H_dim}mm")
        lines.append(f"  Mesh: size={self.mesh_size_dim}mm, refinement={self.refinement_factor}×")
        lines.append(f"  Sulci: n={self.sulci_n}, {self.sulci_w_dim}×{self.sulci_h_dim}mm")

        if self.mode in ['adv-diff', 'no-uptake']:
            lines.append(f"  Flow: U={self.U_ref_dim}mm/s")
            lines.append(f"  Transport: D={self.D_dim}mm²/s, μ={self.mu_dim:.3f}")
            # Only show non-dim if they exist
            if hasattr(self, 'Pe'):
                lines.append(f"  Non-dim: D*={self.D:.3f}, μ*={self.mu:.3f}, Pe={self.Pe:.1f}, Re={self.Re:.3f}")
        else:
            lines.append(f"  Diffusion: D={self.D_dim}mm²/s, μ={self.mu_dim:.3f}")
            if hasattr(self, 'D'):
                lines.append(f"  Non-dim: D*={self.D:.3f}, μ*={self.mu:.3f}, L*={self.L:.1f}")

        return '\n'.join(lines)

    def to_dict(self):
        """Convert parameters to dictionary format."""
        result = {
            'mode': self.mode,
            'dimensional': {
                'L_dim': self.L_dim,
                'H_dim': self.H_dim,
                'sulci_n': self.sulci_n,
                'sulci_h_dim': self.sulci_h_dim,
                'sulci_w_dim': self.sulci_w_dim,
                'mesh_size_dim': self.mesh_size_dim,
                'refinement_factor': self.refinement_factor,
                'U_ref_dim': self.U_ref_dim,
                'D_dim': self.D_dim
            }
        }

        # Handle mu_dim - check if it's a StepUptakeFunction or scalar
        if isinstance(self.mu_dim, StepUptakeFunction):
            result['dimensional']['mu_dim'] = {
                'type': 'StepUptakeFunction',
                'mu_base': self.mu_dim.mu_base,
                'mu_open': self.mu_dim.mu_open,
                'sulcus_left_x': self.mu_dim.sulcus_left_x,
                'sulcus_right_x': self.mu_dim.sulcus_right_x,
                'L_c': self.mu_dim.L_c,
                'Gamma': self.mu_dim.Gamma
            }
        else:
            result['dimensional']['mu_dim'] = self.mu_dim

        # Add non-dimensional data if it exists
        if hasattr(self, 'L_ref'):
            result['non_dimensional'] = {
                'L_ref': self.L_ref,
                'L': self.L, 'H': self.H,
                'sulci_h': self.sulci_h, 'sulci_w': self.sulci_w,
                'mesh_size': self.mesh_size,
                'U_ref': self.U_ref, 'D': self.D
            }

            # Handle mu - check if it's a StepUptakeFunction or scalar
            if isinstance(self.mu, StepUptakeFunction):
                result['non_dimensional']['mu'] = {
                    'type': 'StepUptakeFunction',
                    'mu_base': self.mu.mu_base,
                    'mu_open': self.mu.mu_open,
                    'sulcus_left_x': self.mu.sulcus_left_x,
                    'sulcus_right_x': self.mu.sulcus_right_x,
                    'L_c': self.mu.L_c,
                    'Gamma': self.mu.Gamma
                }
            else:
                result['non_dimensional']['mu'] = self.mu

        # Add computed metrics if they exist
        result['computed_metrics'] = {}
        if hasattr(self, 'Pe') and self.Pe is not None:
            result['computed_metrics']['Pe'] = self.Pe
        if hasattr(self, 'Re') and self.Re is not None:
            result['computed_metrics']['Re'] = self.Re

        return result

    @classmethod
    def from_dict(cls, params_dict):
        """Create Parameters instance from dictionary."""
        dim_params = params_dict.get('dimensional', {})
        mode = params_dict.get('mode', 'adv-diff')

        # Remove mu_dim from dim_params since it's set by mode
        init_params = {k: v for k, v in dim_params.items() if k != 'mu_dim'}
        init_params['mode'] = mode

        return cls(**init_params)

    def get_mesh_generator_params(self):
        """Get non-dimensional parameters for mesh generation."""
        return {
            'width': self.L,
            'height': self.H,
            'sulcus_depth': self.sulci_h if self.sulci_n > 0 else 0,
            'sulcus_width': self.sulci_w if self.sulci_n > 0 else 0,
            'mesh_size': self.mesh_size,
            'refinement_factor': self.refinement_factor,
            'output_dir': None
        }

#endregion

# ========================================================
# region Geometry Configuration Helpers
# ========================================================

def create_geometry_variations(base_params, max_width=1.0, small_thresh=0.10, include_small=False):
    """Create systematic geometry variations for analysis.
    Adds explicit 'small sulci' (w, h <= small_thresh * H_dim) and tags each case with is_small.
    """
    base_config = {
        'L_dim': base_params.L_dim,
        'H_dim': base_params.H_dim,
        'mode': base_params.mode,
    }
    H = float(base_params.H_dim)
    L = float(base_params.L_dim)

    def classify_small(w_mm, h_mm):
        w_over_H = w_mm / H
        h_over_H = h_mm / H
        is_small = (max(w_over_H, h_over_H) <= small_thresh)
        reason = (
            f"max(w/H, h/H) = {max(w_over_H, h_over_H):.3f} "
            f"{'<= ' if is_small else '> '} {small_thresh:.2f}"
        )
        return is_small, w_over_H, h_over_H, reason

    # Systematic variations (existing grid)
    variations = [
        # VERY WIDE (AR ≤ 0.5)
        (1.0, 0.2, 'very_wide_tiny',   'Very wide, tiny depth (AR=0.2)', 'very_wide'),
        (1.0, 0.3, 'very_wide_medium', 'Very wide, medium depth (AR=0.3)', 'very_wide'),
        (1.0, 0.5, 'very_wide_large',  'Very wide, large depth (AR=0.5)', 'very_wide'),

        # MODERATELY WIDE (0.5 < AR ≤ 1.0)
        (0.5, 0.3, 'mod_wide_small',   'Moderately wide, small (AR=0.6)', 'mod_wide'),
        (0.8, 0.6, 'mod_wide_medium',  'Moderately wide, medium (AR=0.75)', 'mod_wide'),
        (1.0, 0.9, 'mod_wide_large',   'Moderately wide, large (AR=0.9)', 'mod_wide'),

        # SQUARE (AR ≈ 1.0)
        (0.2, 0.2, 'square_small',     'Small square sulcus (AR=1.0)', 'square'),
        (0.5, 0.5, 'square_medium',    'Medium square sulcus (AR=1.0)', 'square'),
        (0.7, 0.7, 'square_large',     'Large square sulcus (AR=1.0)', 'square'),

        # MODERATELY DEEP (1.0 < AR ≤ 2.0)
        (0.5, 0.8, 'mod_deep_small',   'Moderately deep, small width (AR=1.6)', 'mod_deep'),
        (0.5, 1.0, 'reference',        'Reference case (AR=2.0)', 'mod_deep'),
        (1.0, 1.5, 'mod_deep_large',   'Moderately deep, large width (AR=1.5)', 'mod_deep'),

        # DEEP (2.0 < AR ≤ 5.0)
        (0.3, 1.0, 'deep_small',       'Deep, small width (AR=3.3)', 'deep'),
        (0.5, 1.5, 'deep_medium',      'Deep, medium width (AR=3.0)', 'deep'),
        (0.4, 2.0, 'deep_large',       'Deep, large depth (AR=5.0)', 'deep'),

        # VERY DEEP (AR > 5.0)
        (0.25, 1.5, 'very_deep_small', 'Very deep, small (AR=6.0)', 'very_deep'),
        (0.15, 1.8, 'very_deep_large', 'Very deep, large (AR=12.0)', 'very_deep'),
        (0.1, 2.0, 'very_deep_extreme','Very deep, extreme (AR=20.0)', 'very_deep'),

        # SPECIAL CASES
        (1.0, 0.05, 'micro_depth_wide','Micro depth, wide (AR=0.05)', 'special'),
        (0.05, 1.0, 'micro_width_deep','Micro width, deep (AR=20.0)', 'special'),
        (1.0, 2.0, 'largest','Largest sulcus, deep (AR=2.0)', 'special'),
        (0.01, 0.01,'micro_square',    'Micro square sulcus (AR=1.0)', 'special'),
        (1.0, 1.0,  'macro_square',    'Macro square sulcus (AR=1.0)', 'special'),
    ]

    # NEW: explicit "small sulci" panel (all ≤ 0.1H with H=1 mm)
    small_panel = [
        (0.03, 0.03, 'small_sq_030',   'Small square (0.03 mm)', 'small'),
        (0.05, 0.05, 'small_sq_050',   'Small square (0.05 mm)', 'small'),
        (0.08, 0.08, 'small_sq_080',   'Small square (0.08 mm)', 'small'),
        (0.10, 0.10, 'small_sq_100',   'Small square (0.10 mm)', 'small'),
        (0.10, 0.05, 'small_wide_100x050','Small wide, shallow', 'small'),
        (0.05, 0.10, 'small_deep_050x100','Small narrow, deeper','small'),
    ]

    if include_small:
        variations.extend(small_panel)

    configs = {}
    for width, depth, key, desc_template, ar_category in variations:
        actual_width = min(width, max_width)
        aspect_ratio = depth / actual_width if actual_width > 0 else float('inf')

        # Classify smallness vs H
        is_small, w_over_H, h_over_H, reason = classify_small(actual_width, depth)

        # Useful ratios (keep old ones, add new ones)
        width_ratio_L = actual_width / L
        depth_ratio_H = depth / H

        description = f"{desc_template} ({actual_width:.2f}x{depth:.2f} mm, AR={aspect_ratio:.2f})"

        configs[key] = {
            **base_config,
            'sulci_w_dim': actual_width,
            'sulci_h_dim': depth,
            'name': description,
            'aspect_ratio': aspect_ratio,
            'aspect_ratio_category': ar_category,
            # ratios
            'width_ratio_L': width_ratio_L,
            'width_over_H': w_over_H,
            'depth_over_H': h_over_H,
            'depth_ratio': depth_ratio_H,
            # smallness flags
            'is_small': is_small,
            'smallness_reason': reason,
            'small_threshold': small_thresh,
        }

    return configs

def create_width_variations(base_params, widths, fixed_depth=None):
    """Create configurations with varying sulcus width and fixed depth."""
    if fixed_depth is None:
        fixed_depth = base_params.sulci_h_dim

    configs = {}
    base_config = {
        'L_dim': base_params.L_dim,
        'H_dim': base_params.H_dim,
        'sulci_n': base_params.sulci_n,
        'mesh_size_dim': base_params.mesh_size_dim,
        'refinement_factor': base_params.refinement_factor,
        'U_ref_dim': base_params.U_ref_dim,
        'D_dim': base_params.D_dim,
        'mode': base_params.mode
    }

    for width in widths:
        key = f'width_{width:.2f}mm'.replace('.', 'p')
        configs[key] = {
            **base_config,
            'sulci_w_dim': width,
            'sulci_h_dim': fixed_depth,
            'name': f'Width variation ({width}×{fixed_depth}mm)'
        }

    return configs

def create_depth_variations(base_params, depths, fixed_width=None):
    """Create configurations with varying sulcus depth and fixed width."""
    if fixed_width is None:
        fixed_width = base_params.sulci_w_dim

    configs = {}
    base_config = {
        'L_dim': base_params.L_dim,
        'H_dim': base_params.H_dim,
        'sulci_n': base_params.sulci_n,
        'mesh_size_dim': base_params.mesh_size_dim,
        'refinement_factor': base_params.refinement_factor,
        'U_ref_dim': base_params.U_ref_dim,
        'D_dim': base_params.D_dim,
        'mode': base_params.mode
    }

    for depth in depths:
        key = f'depth_{depth:.2f}mm'.replace('.', 'p')
        configs[key] = {
            **base_config,
            'sulci_w_dim': fixed_width,
            'sulci_h_dim': depth,
            'name': f'Depth variation ({fixed_width}×{depth}mm)'
        }

    return configs

#endregion

