##########################################################
# Analysis Module
##########################################################
"""
Domain physics analysis functions
"""

# ========================================================
# region Imports
# ========================================================

from mesh import *
import numpy as np
from scipy.integrate import quad
from dolfin import *

MARKERS = {
    'left': 1, 'right': 2, 'top': 3, 'bottom': 4,
    'bottom_left': 5, 'sulcus': 6, 'bottom_right': 7, 'sulcus_opening': 8,
    'y0_line': 10
}

#endregion

# ========================================================
# region Math Expressions for Flux and B.C
# ========================================================

def physical_flux_expression(c, u, mesh_results, D_val):
    """Create physical flux expressions using mesh results."""
    mesh = mesh_results['mesh']
    n = FacetNormal(mesh)
    return {
        'diffusive': -Constant(D_val) * dot(grad(c), n),
        'advective': dot(u, n) * c
    }

def uptake_flux_expression(c, mu_val):
    """
    Return mu * c where mu may be a scalar or a FEniCS expression.
    """
    if np.isscalar(mu_val):
        return Constant(float(mu_val)) * c
    if isinstance(mu_val, (Constant, Function, Expression, UserExpression)) or isinstance(mu_val, ufl.core.expr.Expr):
        return mu_val * c
    raise TypeError(f"Unsupported mu_val type: {type(mu_val)}")

#endregion

# ========================================================
# region Physical Flux Calculations
# ========================================================
# Calculated across all individual boundaries

def compute_physical_flux_boundary(c, u, mesh_results, measures, boundary_marker, D_val):
    """Compute physical flux (diffusive + advective) across a boundary."""

    flux_exprs = physical_flux_expression(c, u, mesh_results, D_val)
    ds_bc = measures['ds_bc']

    diffusive_flux = assemble(flux_exprs['diffusive'] * ds_bc(boundary_marker))
    advective_flux = assemble(flux_exprs['advective'] * ds_bc(boundary_marker))

    return {
        'diffusive': float(diffusive_flux),
        'advective': float(advective_flux),
        'total': float(diffusive_flux + advective_flux)
    }

def compute_sulcus_segment_fluxes(c, u, mesh_results, measures, D_val):
    mesh = mesh_results['mesh']

    # --- 1) external pieces on FULL mesh
    flux_exprs = physical_flux_expression(c, u, mesh_results, D_val)  # uses FacetNormal(mesh)
    diffusive_expr = flux_exprs['diffusive']
    advective_expr = flux_exprs['advective']

    ds_bottom = measures['ds_bottom']
    fluxes = {}
    for name, marker in {
        'bottom_left':  MARKERS['bottom_left'],
        'sulcus':       MARKERS['sulcus'],
        'bottom_right': MARKERS['bottom_right'],
    }.items():
        d = assemble(diffusive_expr * ds_bottom(marker))
        a = assemble(advective_expr * ds_bottom(marker))
        fluxes[name] = {'diffusive': float(d), 'advective': float(a), 'total': float(d+a)}

    # --- 2) submeshes (1=sulcus, 2=rectangle)
    domain_markers = mesh_results.get('domain_markers', None)
    if domain_markers is None:
        raise RuntimeError("domain_markers missing in mesh_results; required for submesh fluxes.")
    sulc_mesh = SubMesh(mesh, domain_markers, 1)
    rect_mesh = SubMesh(mesh, domain_markers, 2)  # 1/2 as set in mesh.py.

    # --- 3) mark y≈0 on each submesh (opening on sulcus, full y=0 on rectangle)
    tol = MeshGenerator.TOLERANCE
    OPEN_ID = MARKERS['sulcus_opening']
    Y0_ID   = MARKERS['y0_line']

    def mark_y0(submesh, label_id):
        facets = MeshFunction('size_t', submesh, submesh.topology().dim()-1, 0)
        class Y0(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 0.0, tol)
        Y0().mark(facets, label_id)
        return facets, Measure('ds', domain=submesh, subdomain_data=facets)

    sulc_facets, ds_sulc = mark_y0(sulc_mesh, OPEN_ID)  # opening
    rect_facets, ds_rect = mark_y0(rect_mesh, Y0_ID)    # full y=0

    # --- 4) pull fields to submeshes
    deg_c = c.function_space().ufl_element().degree()
    try:
        deg_u = u.function_space().ufl_element().sub_elements()[0].degree()
    except Exception:
        deg_u = u.function_space().ufl_element().degree()

    Vc_sulc = FunctionSpace(sulc_mesh, 'CG', deg_c)
    Vu_sulc = VectorFunctionSpace(sulc_mesh, 'CG', deg_u)
    Vc_rect = FunctionSpace(rect_mesh, 'CG', deg_c)
    Vu_rect = VectorFunctionSpace(rect_mesh, 'CG', deg_u)

    c_sulc, u_sulc = Function(Vc_sulc), Function(Vu_sulc)
    c_rect, u_rect = Function(Vc_rect), Function(Vu_rect)

    # allow tiny outside evals + use project
    c.set_allow_extrapolation(True)
    u.set_allow_extrapolation(True)

    c_sulc.assign(project(c, Vc_sulc))
    u_sulc.assign(project(u, Vu_sulc))
    c_rect.assign(project(c, Vc_rect))
    u_rect.assign(project(u, Vu_rect))

    # --- 5) submesh-specific flux expressions (use each submesh's normal)
    def submesh_flux_exprs(c_sub, u_sub, submesh):
        n_sub = FacetNormal(submesh)
        return (
            -Constant(D_val) * dot(grad(c_sub), n_sub),  # diffusive
            dot(u_sub, n_sub) * c_sub                    # advective
        )
    diff_sulc, adv_sulc = submesh_flux_exprs(c_sulc, u_sulc, sulc_mesh)
    diff_rect, adv_rect = submesh_flux_exprs(c_rect, u_rect, rect_mesh)

    # --- 6) opening flux: report INTO the cavity (flip sign)
    J_open_diff = -assemble(diff_sulc * ds_sulc(OPEN_ID))
    J_open_adv  = -assemble(adv_sulc  * ds_sulc(OPEN_ID))
    fluxes['sulcus_opening'] = {
        'diffusive': float(J_open_diff),
        'advective': float(J_open_adv),
        'total':     float(J_open_diff + J_open_adv),
    }

    # --- 7) full y=0 (OUTWARD of rectangle)
    J_y0_diff = assemble(diff_rect * ds_rect(Y0_ID))
    J_y0_adv  = assemble(adv_rect  * ds_rect(Y0_ID))
    fluxes['y0_flux'] = {
        'diffusive': float(J_y0_diff),
        'advective': float(J_y0_adv),
        'total':     float(J_y0_diff + J_y0_adv),
    }

    # --- 8) combined identities
    def sum_fields(keys):
        out = {}
        names = set().union(*(fluxes[k].keys() for k in keys))
        for nm in names:
            out[nm] = float(sum(fluxes[k][nm] for k in keys if nm in fluxes[k]))
        return out

    fluxes['bottom_combined'] = sum_fields(['bottom_left', 'sulcus', 'bottom_right'])

    # y0 = left + right + opening  (all fluxes in same direction)
    fluxes['y0_combined'] = sum_fields(['bottom_left', 'bottom_right'])
    for key in ['diffusive', 'advective', 'total']:
        fluxes['y0_combined'][key] += fluxes['sulcus_opening'][key]  # Add, don't subtract

    return fluxes

def compute_sulcus_segment_fluxes(c, u, mesh_results, measures, D_val):
    """
    Flux bookkeeping for the full sulcus mesh.

    ouutputs:
      - bottom_left, sulcus, bottom_right : external bottom pieces
      - sulcus_opening : interior opening (rectangle → sulcus, via DG0 selector)
      - y0_flux        : one-shot flux across full y≈0 line
      - y0_combined    : left + opening + right
      - bottom_combined: left + sulcus floor + right
    """
    mesh = mesh_results['mesh']
    n = FacetNormal(mesh)

    # --- external flux expressions
    diff_ext = -Constant(D_val) * dot(grad(c), n)
    adv_ext  =  dot(u, n) * c

    ds_bottom = measures['ds_bottom']
    ds_y0     = measures['ds_y0']
    dS_y0     = measures['dS_y0']
    Y0_ID     = MARKERS['y0_line']

    fluxes = {}

    # --- 1) external bottom segments
    for name, marker in {
        'bottom_left':  MARKERS['bottom_left'],
        'sulcus':       MARKERS['sulcus'],
        'bottom_right': MARKERS['bottom_right'],
    }.items():
        d = assemble(diff_ext * ds_bottom(marker))
        a = assemble(adv_ext  * ds_bottom(marker))
        fluxes[name] = {'diffusive': float(d), 'advective': float(a), 'total': float(d+a)}

    # --- 2) DG0 selector for rectangle cells (domain_markers: 1=sulcus, 2=rectangle)
    V0 = FunctionSpace(mesh, "DG", 0)
    chi_rect = Function(V0)
    dm = V0.dofmap()
    vec = chi_rect.vector()
    vec.zero()
    for cell in cells(mesh):
        dof = dm.cell_dofs(cell.index())[0]
        if mesh_results['domain_markers'][cell] == 2:  # rectangle
            vec[dof] = 1.0
    vec.apply("insert")

    def rect_trace(q):
        return chi_rect('+')*q('+') + chi_rect('-')*q('-')

    # rectangle-side traces
    n_rect = rect_trace(n)
    c_r    = rect_trace(c)
    u_r    = rect_trace(u)

    diff_open = -Constant(D_val) * dot(grad(c_r), n_rect)
    adv_open  =  dot(u_r, n_rect) * c_r

    J_open_diff = assemble(diff_open * dS_y0(Y0_ID))
    J_open_adv  = assemble(adv_open  * dS_y0(Y0_ID))

    # --- exchange strength on the mouth (rectangle-side trace)
    q_open = diff_open + adv_open                     # local (signed) flux density J·n on mouth
    E_L1   = assemble(abs(q_open) * dS_y0(Y0_ID))     # ∫ |J·n| ds  (exchange strength)
    Q_in   = assemble(conditional(ge(q_open, 0.0),  q_open, 0.0) * dS_y0(Y0_ID))  # ∫ (J·n)^+ ds
    Q_out  = assemble(conditional(le(q_open, 0.0), -q_open, 0.0) * dS_y0(Y0_ID))  # ∫ (J·n)^- ds
    L_sig  = assemble(Constant(1.0) * dS_y0(Y0_ID))  # mouth length

    fluxes['sulcus_opening'] = {
        'diffusive': float(J_open_diff),
        'advective': float(J_open_adv),
        'total':     float(J_open_diff + J_open_adv)
    }

    fluxes['sulcus_opening_extra'] = {
        'E_L1':      float(E_L1),
        'E_avg':     float(E_L1 / L_sig),
        'Q_in':      float(Q_in),
        'Q_out':     float(Q_out),
        'net_check': float(Q_in - Q_out),
        'length':    float(L_sig)
    }

    # --- 3) one-shot full line flux (external measure)
    # one-shot full y=0 flux = exterior part + interior opening (rectangle side)
    J_y0_diff_ext = assemble(diff_ext * ds_y0(Y0_ID))
    J_y0_adv_ext  = assemble(adv_ext  * ds_y0(Y0_ID))

    # interior opening from the rectangle side
    J_y0_diff_int = assemble(diff_open * dS_y0(Y0_ID))
    J_y0_adv_int  = assemble(adv_open  * dS_y0(Y0_ID))

    J_y0_diff = J_y0_diff_ext + J_y0_diff_int
    J_y0_adv  = J_y0_adv_ext  + J_y0_adv_int

    fluxes['y0_flux'] = {
        'diffusive': float(J_y0_diff),
        'advective': float(J_y0_adv),
        'total':     float(J_y0_diff + J_y0_adv),
    }

    # --- 4) combined identities
    def sum_fields(keys):
        out = {}
        names = set().union(*(fluxes[k].keys() for k in keys))
        for nm in names:
            out[nm] = float(sum(fluxes[k][nm] for k in keys if nm in fluxes[k]))
        return out

    fluxes['bottom_combined'] = sum_fields(['bottom_left', 'sulcus', 'bottom_right'])
    fluxes['y0_combined']     = sum_fields(['bottom_left', 'bottom_right', 'sulcus_opening'])

    # --- 5) internal consistency check
    diff_val = abs(fluxes['y0_flux']['total'] - fluxes['y0_combined']['total'])
    if diff_val > 1e-10:
        print(f"⚠️ y0_flux vs y0_combined differ by {diff_val:.3e}")

    return fluxes

#endregion

# ========================================================
# region Uptake Flux Calculations (Robin B.C)
# ========================================================
# Calculated on all individual external lower boundaries

def compute_uptake_flux_bottom(c, measures, mu_val):
    """Compute uptake flux on bottom boundary: J = mu * c"""
    uptake_expr = uptake_flux_expression(c, mu_val)
    ds_bc = measures['ds_bc']
    return float(assemble(uptake_expr * ds_bc(MARKERS['bottom'])))

def compute_uptake_flux_segments(c, measures, mu_val):
    """
    Compute uptake flux across external bottom boundary segments in the sulcus domain.
    Specifically: bottom_left, sulcus, and bottom_right.
    Returns a dictionary with individual segment fluxes and the total
    """
    uptake_expr = uptake_flux_expression(c, mu_val)
    ds_bottom = measures['ds_bottom']

    # Compute individual fluxes
    flux_left = float(assemble(uptake_expr * ds_bottom(MARKERS['bottom_left'])))
    flux_sulcus = float(assemble(uptake_expr * ds_bottom(MARKERS['sulcus'])))
    flux_right = float(assemble(uptake_expr * ds_bottom(MARKERS['bottom_right'])))

    # Return results with total
    return {
        'bottom_left': flux_left,
        'sulcus': flux_sulcus,
        'bottom_right': flux_right,
        'total': flux_left + flux_sulcus + flux_right
    }

#endregion

# ========================================================
# region Profile Extraction
# ========================================================

def extract_concentration_vertical_line_profile(c, mesh, x_location, y_range=None, n_points=100):
    """Extract concentration profile along a vertical line at fixed x-coordinate."""
    from dolfin import Point

    # Get domain bounds if not specified
    if y_range is None:
        coords = mesh.coordinates()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    else:
        y_min, y_max = y_range

    # Create sampling points along the line
    y_coords = np.linspace(y_min, y_max, n_points)

    # Initialise arrays
    c_vals = np.zeros(n_points)
    valid_points = np.zeros(n_points, dtype=bool)

    # Allow mild extrapolation if needed
    try:
        c.set_allow_extrapolation(True)
    except Exception:
        pass

    # Sample concentration at each point
    for i, y in enumerate(y_coords):
        try:
            point = Point(x_location, y)
            # Check if point is inside mesh
            if mesh.bounding_box_tree().compute_first_entity_collision(point) != 4294967295:
                c_vals[i] = float(c(point))
                valid_points[i] = True
        except Exception:
            valid_points[i] = False

    return {
        'y_coords': y_coords[valid_points],
        'c': c_vals[valid_points],
    }

def extract_concentration_horizontal_line_profile(c, mesh, y_location, x_range=None, n_points=100):
    """Extract concentration profile along a horizontal line at fixed y-coordinate."""
    from dolfin import Point

    # Get domain bounds if not specified
    if x_range is None:
        coords = mesh.coordinates()
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    else:
        x_min, x_max = x_range

    # Create sampling points along the line
    x_coords = np.linspace(x_min, x_max, n_points)

    # Initialise arrays
    c_vals = np.zeros(n_points)
    valid_points = np.zeros(n_points, dtype=bool)

    # Allow mild extrapolation if needed
    try:
        c.set_allow_extrapolation(True)
    except Exception:
        pass

    # Sample concentration at each point
    for i, x in enumerate(x_coords):
        try:
            point = Point(x, y_location)
            # Check if point is inside mesh
            if mesh.bounding_box_tree().compute_first_entity_collision(point) != 4294967295:
                c_vals[i] = float(c(point))
                valid_points[i] = True
        except Exception:
            valid_points[i] = False

    return {
        'x_coords': x_coords[valid_points],
        'c': c_vals[valid_points],
    }

def compute_conc_profiles(results, *, n_points=400):
    """
    Compute horizontal and vertical concentration line-profiles and store BOTH:"""

    # --- extract core objects
    c = results.get('c')
    mesh = (results.get('mesh_results') or {}).get('mesh')
    params = results.get('params', None)
    if c is None or mesh is None or params is None:
        return results  # nothing to do

    L = float(getattr(params, 'L_dim', getattr(params, 'L', 1.0)))
    H = float(getattr(params, 'H_dim', getattr(params, 'H', 1.0)))

    # Try to infer domain_type if not present
    domain_type = results.get('domain_type', None)
    if domain_type is None:
        # Heuristic: if a dimensional sulcus depth exists and is > 0, call it sulcus
        h_dim = getattr(params, 'sulci_h_dim', 0.0)
        domain_type = 'sulcus' if (h_dim and h_dim > 0) else 'rectangular'
        results['domain_type'] = domain_type

    mass_metrics = results.setdefault('mass_metrics', {})

    # ---------------------------------------------
    # helpers
    # ---------------------------------------------
    def _stats(vals):
        vals = np.asarray(vals)
        if vals.size == 0:
            return {'min_c': None, 'max_c': None, 'avg_c': None, 'n_samples': 0}
        return {
            'min_c': float(np.min(vals)),
            'max_c': float(np.max(vals)),
            'avg_c': float(np.mean(vals)),
            'n_samples': int(vals.size),
        }

    # Use existing extractors
    # Each extractor should return a dict with arrays for 'x', 'y', 'c'
    #   extract_concentration_horizontal_line_profile(c, mesh, y_location, x_range=None, n_points=100)
    #   extract_concentration_vertical_line_profile(c, mesh, x_location, y_range=None, n_points=100)

    # ---------------------------------------------
    # choose sampling lines + bounds
    # ---------------------------------------------
    if domain_type == 'rectangular':
        horiz_lines = [
            (1e-6 * H, "mouth_level"),
            (0.25 * H, "lower_channel"),
            (0.50 * H, "mid_channel"),
            (0.75 * H, "upper_channel"),
        ]
        vert_lines = [
            (0.25 * L, "x_quarter"),
            (0.50 * L, "x_mid"),
            (0.75 * L, "x_three_quarters"),
        ]
        x_range = (0.0, float(L))
        y_range = (0.0, float(H))  # never below 0
    else:
        # use full mesh bounds (include sulcus dip)
        coords = mesh.coordinates()
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_sulcus_mid = 0.5 * (y_min + 0.0)

        horiz_lines = [
            (y_sulcus_mid, "sulcus_mid"),
            (1e-6 * H,     "mouth_level"),
            (0.25 * H,     "lower_channel"),
            (0.50 * H,     "mid_channel"),
            (0.75 * H,     "upper_channel"),
        ]
        vert_lines = [
            (0.25 * L, "x_quarter"),
            (0.50 * L, "x_mid"),
            (0.75 * L, "x_three_quarters"),
        ]
        x_range = (x_min, x_max)
        y_range = None  # let extractor use full bounds

    # ---------------------------------------------
    # compute + store stats and samples
    # ---------------------------------------------
    profiles_stats = {'horizontal': {}, 'vertical': {}}
    profiles_full  = {'horizontal': {}, 'vertical': {}}

    # Allow safe extrapolation for sampling
    try:
        c.set_allow_extrapolation(True)
    except Exception:
        pass

    for y_loc, name in horiz_lines:
        prof = extract_concentration_horizontal_line_profile(
            c, mesh, y_location=float(y_loc), x_range=x_range, n_points=n_points
        )
        s = _stats(prof['c'])
        if s['n_samples'] > 0:
            profiles_stats['horizontal'][name] = {'y': float(y_loc), **s}
            profiles_full['horizontal'][name]  = {
                'y': float(y_loc),
                'x': np.asarray(prof['x_coords']).tolist(),
                'c': np.asarray(prof['c']).tolist(),
            }

    for x_loc, name in vert_lines:
        prof = extract_concentration_vertical_line_profile(
            c, mesh, x_location=float(x_loc), y_range=y_range, n_points=n_points
        )
        s = _stats(prof['c'])
        if s['n_samples'] > 0:
            profiles_stats['vertical'][name] = {'x': float(x_loc), **s}
            profiles_full['vertical'][name]  = {
                'x': float(x_loc),
                'y': np.asarray(prof['y_coords']).tolist(),
                'c': np.asarray(prof['c']).tolist(),
            }

    mass_metrics['profiles'] = profiles_stats
    mass_metrics['profiles_full'] = profiles_full
    mass_metrics['profiles_meta'] = {
        'n_points': int(n_points),
        'domain_type': domain_type,
        'x_range': tuple(map(float, x_range)) if x_range is not None else None,
        'y_range': tuple(map(float, y_range)) if y_range is not None else None,
    }

    return results

def extract_velocity_vertical_line_profile(u, mesh, x_location, y_range=None, n_points=100):
    """Extract velocity profile along a vertical line at fixed x-coordinate."""
    from dolfin import Point

    # Get domain bounds if not specified
    if y_range is None:
        coords = mesh.coordinates()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    else:
        y_min, y_max = y_range

    # Create sampling points along the line
    y_coords = np.linspace(y_min, y_max, n_points)

    # Initialize arrays for velocity components
    u_x_vals = np.zeros(n_points)
    u_y_vals = np.zeros(n_points)
    u_mag_vals = np.zeros(n_points)
    valid_points = np.zeros(n_points, dtype=bool)

    # Sample velocity at each point
    for i, y in enumerate(y_coords):
        try:
            point = Point(x_location, y)
            # Check if point is inside mesh
            if mesh.bounding_box_tree().compute_first_entity_collision(point) != 4294967295:
                u_val = u(point)
                u_x_vals[i] = u_val[0]
                u_y_vals[i] = u_val[1]
                u_mag_vals[i] = np.sqrt(u_val[0]**2 + u_val[1]**2)
                valid_points[i] = True
        except:
            valid_points[i] = False

    return {
        'y_coords': y_coords[valid_points],
        'u_x': u_x_vals[valid_points],
        'u_y': u_y_vals[valid_points],
        'u_mag': u_mag_vals[valid_points]
    }

def extract_velocity_horizontal_line_profile(u, mesh, y_location, x_range=None, n_points=100):
    """Extract velocity profile along a horizontal line at fixed y-coordinate."""
    from dolfin import Point

    # Get domain bounds if not specified
    if x_range is None:
        coords = mesh.coordinates()
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    else:
        x_min, x_max = x_range

    # Create sampling points along the line
    x_coords = np.linspace(x_min, x_max, n_points)

    # Initialize arrays for velocity components
    u_x_vals = np.zeros(n_points)
    u_y_vals = np.zeros(n_points)
    u_mag_vals = np.zeros(n_points)
    valid_points = np.zeros(n_points, dtype=bool)

    # Sample velocity at each point
    for i, x in enumerate(x_coords):
        try:
            point = Point(x, y_location)
            # Check if point is inside mesh
            if mesh.bounding_box_tree().compute_first_entity_collision(point) != 4294967295:
                u_val = u(point)
                u_x_vals[i] = u_val[0]
                u_y_vals[i] = u_val[1]
                u_mag_vals[i] = np.sqrt(u_val[0]**2 + u_val[1]**2)
                valid_points[i] = True
        except:
            valid_points[i] = False

    return {
        'x_coords': x_coords[valid_points],
        'u_x': u_x_vals[valid_points],
        'u_y': u_y_vals[valid_points],
        'u_mag': u_mag_vals[valid_points]
    }

#endregion

# ========================================================
# region Overall Calculations
# ========================================================

def compute_flux_metrics(c, u, mesh_results, domain_type, measures, D_val, mu_val):
    """
    compute flux metrics using mesh_results and measures.

    inputs:
        c: Concentration field
        u: Velocity field
        mesh_results: Dictionary containing mesh and markers
        domain_type: 'sulcus' or 'rectangular'
        measures: Dictionary containing integration measures
        D_val: Diffusion coefficient
        mu_val: Uptake parameter

    outputs:
        dict: Dictionary containing flux metrics
    """

    # Basic flux metrics for all domain types
    flux_metrics = {
        'physical_flux': {
            'left': compute_physical_flux_boundary(c, u, mesh_results, measures, MARKERS['left'], D_val),
            'right': compute_physical_flux_boundary(c, u, mesh_results, measures, MARKERS['right'], D_val),
            'top': compute_physical_flux_boundary(c, u, mesh_results, measures, MARKERS['top'], D_val),
            'bottom': compute_physical_flux_boundary(c, u, mesh_results, measures, MARKERS['bottom'], D_val)
        },
        'uptake_flux': compute_uptake_flux_bottom(c, measures, mu_val)
    }

    # Add sulcus-specific flux calculations if applicable
    if domain_type == 'sulcus':
        flux_metrics['sulcus_specific'] = {
            'physical_flux': compute_sulcus_segment_fluxes(c, u, mesh_results, measures, D_val),
            'uptake_flux': compute_uptake_flux_segments(c, measures, mu_val)
        }

    return flux_metrics

def compute_mass_metrics(c, measures, domain_type):
    """
    Compute total/region mass & averages, plus concentration line-profile stats.
    Rectangular: sample only y in [0, H]; Sulcus: sample full mesh bounds (incl. y<0).
    """

    # -----------------------------
    # 1) Mass metrics
    # -----------------------------
    if domain_type == 'sulcus':
        dx_domain_sulc = measures['dx_domain_sulc']
        total_mass = float(assemble(c * (dx_domain_sulc(1) + dx_domain_sulc(2))))
        total_area = float(assemble(Constant(1.0) * (dx_domain_sulc(1) + dx_domain_sulc(2))))

        sulcus_mass    = float(assemble(c * dx_domain_sulc(1)))
        rectangle_mass = float(assemble(c * dx_domain_sulc(2)))
        sulcus_area    = float(assemble(Constant(1.0) * dx_domain_sulc(1)))
        rect_area      = float(assemble(Constant(1.0) * dx_domain_sulc(2)))

        mass_metrics = {
            'total_mass': total_mass,
            'sulcus_mass': sulcus_mass,
            'rectangle_mass': rectangle_mass,
            'total_area': total_area,
            'sulcus_area': sulcus_area,
            'rectangle_area': rect_area,
            'average_concentration': {
                'total': total_mass / total_area if total_area > 0 else None,
                'sulcus_region': sulcus_mass / sulcus_area if sulcus_area > 0 else None,
                'rectangle_region': rectangle_mass / rect_area if rect_area > 0 else None,
            }
        }
    else:
        dx_domain_rect = measures['dx_domain_rect']
        total_mass = float(assemble(c * dx_domain_rect))
        total_area = float(assemble(Constant(1.0) * dx_domain_rect))
        mass_metrics = {
            'total_mass': total_mass,
            'total_area': total_area,
            'average_concentration': total_area and total_mass / total_area or 0.0
        }

    return mass_metrics

def compute_velocity_metrics(u, mesh_results, params):
    """
    Extract key velocity metrics for DataFrame storage and comparison.
    """

    if u is None:
        return {}

    mesh = mesh_results['mesh']

    # Skip if no flow
    mode = getattr(params, 'mode', 'unknown')
    if mode not in ['adv-diff', 'no-uptake']:
        return {}

    try:
        # Domain geometry
        L = params.L
        H = params.H
        sulcus_w = params.sulci_w
        sulcus_center_x = L / 2
        sulcus_left_x = sulcus_center_x - sulcus_w / 2
        sulcus_right_x = sulcus_center_x + sulcus_w / 2

        velocity_metrics = {}

        # 1. Key horizontal line profiles
        horizontal_lines = [
            (1e-6 * H, "mouth_level"),
            (0.25 * H, "lower_channel"),
            (0.50 * H, "mid_channel"),
            (0.75 * H, "upper_channel")
        ]

        for y_loc, line_name in horizontal_lines:
            if 0 <= y_loc <= H:
                profile = extract_velocity_horizontal_line_profile(u, mesh, y_loc, x_range=(0, L))
                if len(profile['u_x']) > 0:
                    velocity_metrics[f'max_ux_{line_name}'] = np.max(np.abs(profile['u_x']))
                    velocity_metrics[f'max_umag_{line_name}'] = np.max(profile['u_mag'])
                    velocity_metrics[f'avg_ux_{line_name}'] = np.mean(np.abs(profile['u_x']))
                    velocity_metrics[f'avg_umag_{line_name}'] = np.mean(profile['u_mag'])
                else:
                    velocity_metrics[f'max_ux_{line_name}'] = 0
                    velocity_metrics[f'max_umag_{line_name}'] = 0
                    velocity_metrics[f'avg_ux_{line_name}'] = 0
                    velocity_metrics[f'avg_umag_{line_name}'] = 0

        # 2. Key vertical line profiles
        vertical_lines = [
            (sulcus_left_x, "sulcus_leading"),
            (sulcus_center_x, "sulcus_center"),
            (sulcus_right_x, "sulcus_trailing")
        ]

        for x_loc, line_name in vertical_lines:
            if 0 <= x_loc <= L:
                profile = extract_velocity_vertical_line_profile(u, mesh, x_loc, y_range=(0, H))
                if len(profile['u_mag']) > 0:
                    velocity_metrics[f'max_umag_{line_name}'] = np.max(profile['u_mag'])
                    velocity_metrics[f'max_uy_{line_name}'] = np.max(np.abs(profile['u_y']))
                    velocity_metrics[f'avg_umag_{line_name}'] = np.mean(profile['u_mag'])
                    velocity_metrics[f'avg_uy_{line_name}'] = np.mean(np.abs(profile['u_y']))
                else:
                    velocity_metrics[f'max_umag_{line_name}'] = 0
                    velocity_metrics[f'max_uy_{line_name}'] = 0
                    velocity_metrics[f'avg_umag_{line_name}'] = 0
                    velocity_metrics[f'avg_uy_{line_name}'] = 0

        # 3. Global velocity metrics
        # Sample velocity across the entire domain for global statistics
        coords = mesh.coordinates()
        n_sample = min(1000, len(coords))  # Sample at most 1000 points
        sample_indices = np.random.choice(len(coords), n_sample, replace=False)

        global_u_x = []
        global_u_y = []
        global_u_mag = []

        from dolfin import Point
        for idx in sample_indices:
            try:
                point = Point(coords[idx])
                u_val = u(point)
                global_u_x.append(u_val[0])
                global_u_y.append(u_val[1])
                global_u_mag.append(np.sqrt(u_val[0]**2 + u_val[1]**2))
            except:
                continue

        if global_u_mag:
            velocity_metrics['global_max_umag'] = np.max(global_u_mag)
            velocity_metrics['global_avg_umag'] = np.mean(global_u_mag)
            velocity_metrics['global_max_ux'] = np.max(np.abs(global_u_x))
            velocity_metrics['global_avg_ux'] = np.mean(np.abs(global_u_x))
            velocity_metrics['global_max_uy'] = np.max(np.abs(global_u_y))
            velocity_metrics['global_avg_uy'] = np.mean(np.abs(global_u_y))
        else:
            velocity_metrics['global_max_umag'] = 0
            velocity_metrics['global_avg_umag'] = 0
            velocity_metrics['global_max_ux'] = 0
            velocity_metrics['global_avg_ux'] = 0
            velocity_metrics['global_max_uy'] = 0
            velocity_metrics['global_avg_uy'] = 0

        return velocity_metrics

    except Exception as e:
        print(f"⚠️ Warning: Could not extract velocity metrics: {e}")
        return {}

#endregion

# ========================================================
# region Mu Effective Analysis
# ========================================================

def sample_mu_along_bottom(results, n_points=500, y_at_bottom=0.0, save_csv_path=None):
    """
    Sample the Robin uptake coefficient mu(x) along the bottom wall (y ~= 0).
    """
    params = results.get('params', None)
    mesh   = results.get('mesh_results', {}).get('mesh', None)
    if params is None or mesh is None:
        raise ValueError("results must contain 'params' and 'mesh_results[mesh]'")

    mu_obj = getattr(params, 'mu', None)  # may be float, Constant, or UserExpression
    coords = mesh.coordinates()
    x_min, x_max = float(coords[:,0].min()), float(coords[:,0].max())

    # Build x-grid along the bottom
    xs = np.linspace(x_min, x_max, int(n_points))

    # Normalise access to mu(x): support float, Constant, Expression
    def eval_mu_at(x):
        if np.isscalar(mu_obj):
            return float(mu_obj)
        # dolfin.Constant supports call operator: Constant(…)()
        if 'Constant' in type(mu_obj).__name__:
            return float(mu_obj())
        # UserExpression / UFL expression: sample at Point
        try:
            return float(mu_obj(Point(float(x), float(y_at_bottom))))
        except Exception:
            # last resort: cast to Constant inside
            return float(Constant(mu_obj)(Point(float(x), float(y_at_bottom))))

    mus = np.array([eval_mu_at(x) for x in xs], dtype=float)

    out = {
        'x': xs,
        'mu': mus,
        'mu_mean': float(np.trapz(mus, xs) / (xs[-1] - xs[0]) if len(xs) > 1 else mus.mean()),
        'mu_min':  float(np.min(mus)),
        'mu_max':  float(np.max(mus)),
    }

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        pd.DataFrame({'x': xs, 'mu': mus}).to_csv(save_csv_path, index=False)

    return out

def compute_concentration_profiles(results):
    """Concentration line-integrals along y=0 using the rectangle-side trace on the mouth."""
    c = results['c']
    mesh_results = results['mesh_results']
    measures = results['measures']

    mesh = mesh_results['mesh']
    ds_y0       = measures['ds_y0']       # external facets on y=0 (flat wall)
    dS_y0       = measures['dS_y0']       # interior interface at y=0 (mouth line)

    # --- markers
    Y0_ID = MARKERS['y0_line']

    # --- build DG0 selector for rectangle cells and a rectangle-side trace
    V0 = FunctionSpace(mesh, "DG", 0)
    chi_rect = Function(V0)
    dm  = V0.dofmap()
    vec = chi_rect.vector(); vec.zero()
    dmark = mesh_results['domain_markers']   # 1 = sulcus, 2 = rectangle
    for cell in cells(mesh):
        dof = dm.cell_dofs(cell.index())[0]
        vec[dof] = 1.0 if dmark[cell] == 2 else 0.0
    vec.apply("insert")

    def rect_trace(q):
        # pick the rectangle-side value on interior facets;
        # equals q on exterior facets (only one side there)
        return chi_rect('+')*q('+') + chi_rect('-')*q('-')

    # --- exterior flat y=0 contribution: ∫ c ds over the flat parts
    C_y0_ext = float(assemble(c * ds_y0(Y0_ID)))

    # --- interior mouth contribution using rectangle-side trace
    c_rect = rect_trace(c)
    C_mouth = float(assemble(c_rect * dS_y0(Y0_ID)))

    # --- totals
    C_y0_total = C_y0_ext + C_mouth

    results = {
        'C_y0_ext':   C_y0_ext,
        'C_mouth':    C_mouth,
        'C_y0_total': C_y0_total,
    }

    # lengths of each piece (handy for line-averages)
    L_y0_ext   = float(assemble(1.0 * ds_y0(Y0_ID)))
    L_mouth    = float(assemble(1.0 * dS_y0(Y0_ID)))

    results['lengths'] = {
        'L_y0_ext':   L_y0_ext,
        'L_mouth':    L_mouth,
        'L_y0_total': L_y0_ext + L_mouth,
    }

    # optional line-averages
    results['means'] = {
        'mean_y0_ext':   C_y0_ext / L_y0_ext   if L_y0_ext   > 0 else np.nan,
        'mean_mouth':    C_mouth  / L_mouth    if L_mouth    > 0 else np.nan,
        'mean_y0_total': C_y0_total / (L_y0_ext + L_mouth) if (L_y0_ext + L_mouth) > 0 else np.nan,
    }

    return results

def compute_mu_eff_arc(results):
    """Analytical mu_eff^arc = mu * (1 + (L_sulcus - w)/L)."""
    params = results['params']
    L  = float(params.L)
    h  = float(params.sulci_h)
    w  = float(params.sulci_w)
    mu = float(params.mu)

    if w <= 0 or h <= 0 or L <= 0:
        return None

    def _integrand(u):
        return np.sqrt(1.0 + (np.pi * h / w * np.cos(np.pi * u))**2)

    try:
        from scipy.integrate import quad
        integral, _ = quad(_integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=200)
    except Exception:
        print("Could not find arc length of sulcus")
        return None

    L_sulcus = w * float(integral)
    return float(mu * (1.0 + (L_sulcus - w) / L))

def compute_mu_eff_enh(results, kappa=10.0):
    params = results['params']
    L  = float(params.L)
    h  = float(params.sulci_h)
    w  = float(params.sulci_w)
    mu = float(params.mu)

    if L <= 0 or mu < 0:
        return None
    if w <= 0:
        return None

    f = 1.0 / np.sqrt(1.0 + kappa * mu * (h**2) / w)
    return float(mu * ((L - w) / L + (w / L) * f))

def compute_mu_eff_sim(results, conc=None):
    """
    mu_eff^sim = J_{y=0} / ∫_{y=0} c ds,using the sulcus solution values. Sign follows normal on y=0 (channel -> sulcus).
    """
    import numpy as np

    if conc is None:
        conc = compute_concentration_profiles(results)

    C_y0 = conc['C_y0_total']
    if not np.isfinite(C_y0) or C_y0 <= 0.0:
        return None

    pf = results.get('flux_metrics', {}).get('sulcus_specific', {}).get('physical_flux', {})
    J_y0 = None
    for key in ('y0_flux', 'y0_combined'):
        if key in pf and 'total' in pf[key]:
            J_y0 = float(pf[key]['total'])
            break
    if J_y0 is None:
        return None

    return float(J_y0 / C_y0)

def compute_mu_eff_sim_mouth(results, conc=None):
    """mu_mouth^sim = F_Σ / C_Σ, using the channel-side trace on Σ from conc['C_mouth'].Sign follows normal on Σ."""
    import numpy as np

    if conc is None:
        conc = compute_concentration_profiles(results)

    C_sigma = conc['C_mouth']
    if not np.isfinite(C_sigma) or C_sigma <= 0.0:
        return None

    pf = results.get('flux_metrics', {}).get('sulcus_specific', {}).get('physical_flux', {})
    J_sigma = None
    for key in ('opening', 'mouth', 'y0_opening', 'y0_mouth', 'sulcus_opening'):
        if key in pf and 'total' in pf[key]:
            J_sigma = float(pf[key]['total'])
            break
    if J_sigma is None:
        return None

    return float(J_sigma / C_sigma)

def compute_mu_eff_metrics(results, kappa=10.0):
    params = results['params']
    mu = float(params.mu)

    conc = compute_concentration_profiles(results)

    mu_eff_arc  = compute_mu_eff_arc(results)
    mu_eff_enh  = compute_mu_eff_enh(results, kappa=kappa)
    mu_eff_sim  = compute_mu_eff_sim(results, conc=conc)
    mu_eff_open = compute_mu_eff_sim_mouth(results, conc=conc)

    def _safe_ratio(x, y):
        return float(x / y) if (x is not None and y not in (None, 0.0)) else None

    def _safe_pct_err(approx, truth):
        if truth in (None, 0.0) or approx is None:
            return None
        return float(abs(approx - truth) / abs(truth) * 100.0)

    ratios = {
        'arc' : _safe_ratio(mu_eff_arc,  mu),
        'enh' : _safe_ratio(mu_eff_enh,  mu),
        'sim' : _safe_ratio(mu_eff_sim,  mu),
        'open': _safe_ratio(mu_eff_open, mu),
    }

    errors_vs_sim = {
        'arc'  : _safe_pct_err(mu_eff_arc,  mu_eff_sim),
        'enh'  : _safe_pct_err(mu_eff_enh,  mu_eff_sim),
        'open' : _safe_pct_err(mu_eff_open, mu_eff_sim),
    }

    # Optional quick sign sanity checks:
    # assert ratios['sim'] is None or ratios['sim'] >= 0, "mu_eff^sim turned negative — check normal orientation and flux signs."
    # assert ratios['open'] is None or ratios['open'] >= 0, "mu_mouth^sim turned negative — check Σ normal and flux sign."

    # Pull fluxes for audit
    pf = results.get('flux_metrics', {}).get('sulcus_specific', {}).get('physical_flux', {})
    J_y0 = next((float(pf[k]['total']) for k in ('y0_flux','y0_combined') if k in pf and 'total' in pf[k]), None)
    J_sigma = next((float(pf[k]['total']) for k in ('opening','mouth','y0_opening','y0_mouth','sulcus_opening')
                    if k in pf and 'total' in pf[k]), None)

    audit = {
        'concentrations': {
            'C_y0_ext'  : conc['C_y0_ext'],
            'C_mouth'   : conc['C_mouth'],
            'C_y0_total': conc['C_y0_total'],
        },
        'lengths': conc.get('lengths', {}),
        'means': conc.get('means', {}),
        'fluxes': {
            'J_y0_total'   : J_y0,
            'J_sigma_mouth': J_sigma,
        },
    }

    return {
        'mu_eff_arc'    : mu_eff_arc,
        'mu_eff_enh'    : mu_eff_enh,
        'mu_eff_sim'    : mu_eff_sim,
        'mu_eff_open'   : mu_eff_open,
        'ratios'        : ratios,
        'errors_vs_sim' : errors_vs_sim,
        'audit'         : audit,
    }