##########################################################
# Advection-Diffusion Solver Module
##########################################################

# ========================================================
# Imports
# ========================================================

from dolfin import *
import numpy as np

# ========================================================
# Advection-Diffusion Solver
# ========================================================

def advdiff_solver(mesh_results, u, C, D, mu, mesh_type="sulcus"):
    """
    Solve the advection-diffusion equation for concentration field.
    """

    # Extract mesh data
    mesh = mesh_results['mesh']
    bc_markers = mesh_results['bc_markers']

    # ----------------------------------------------------
    # Boundary Conditions
    # ----------------------------------------------------

    # Dirichlet BC on left and right (consistent markers: 1=left, 2=right)
    bc_left = DirichletBC(C, Constant(1.0), bc_markers, 1)   # C = 1.0
    bc_right = DirichletBC(C, Constant(0.0), bc_markers, 2)  # C = 0.0
    bcs_c = [bc_left, bc_right]

    # ----------------------------------------------------
    # Setup Integration Measures
    # ----------------------------------------------------

    # Build the variational form
    c_sol = TrialFunction(C)
    phi = TestFunction(C)

    # Standard advection-diffusion terms
    a_c = (D * inner(grad(c_sol), grad(phi))
           + inner(dot(u, grad(c_sol)), phi)) * dx

    # Add Robin boundary condition on bottom boundary (marker 4 for both mesh types)
    ds_bc = Measure("ds", domain=mesh, subdomain_data=bc_markers)
    a_c += mu * c_sol * phi * ds_bc(4)  # bottom boundary

    # Right-hand side
    L_c = Constant(0.0) * phi * dx

    # Solve
    c_sol = Function(C)
    solve(a_c == L_c, c_sol, bcs_c)

    return c_sol

def advdiff_solver_variable_mu(mesh_results, u, C, D, mu_function, mesh_type="sulcus"):
    """
    Advection-diffusion with spatially-varying Robin coefficient μ(x) on bottom boundary.
    """
    import numpy as np
    mesh = mesh_results['mesh']
    bc_markers = mesh_results['bc_markers']

    # Dirichlet BCs: 1=left (c=1), 2=right (c=0)
    bc_left  = DirichletBC(C, Constant(1.0), bc_markers, 1)
    bc_right = DirichletBC(C, Constant(0.0), bc_markers, 2)
    bcs_c = [bc_left, bc_right]

    c_sol = TrialFunction(C)
    phi   = TestFunction(C)

    ds_bc = Measure("ds", domain=mesh, subdomain_data=bc_markers)

    # Standard advection-diffusion + Robin(μ(x))
    a_c = (D * inner(grad(c_sol), grad(phi)) + inner(dot(u, grad(c_sol)), phi)) * dx
    a_c += mu_function * c_sol * phi * ds_bc(4)

    L_c = Constant(0.0) * phi * dx

    c_sol = Function(C)
    solve(a_c == L_c, c_sol, bcs_c)

    # --- Robustness: clamp non-finite and tiny negatives ---
    c_array = c_sol.vector().get_local()
    bad = ~np.isfinite(c_array)
    if np.any(bad):
        print(f"WARNING: {bad.sum()} non-finite concentration entries; clamping to 0.")
        c_array[bad] = 0.0
        c_sol.vector().set_local(c_array); c_sol.vector().apply("insert")

    c_fin = c_sol.vector().get_local()
    neg = c_fin < 0
    if np.any(neg):
        mn = c_fin[neg].min()
        if abs(mn) < 1e-12:
            c_fin[neg] = 0.0
            c_sol.vector().set_local(c_fin); c_sol.vector().apply("insert")
            print("✓ Clamped tiny negative values to 0 (numerical noise).")
        else:
            print(f"WARNING: {neg.sum()} negative values; most negative {mn:.3e}")

    arr = c_sol.vector().get_local()
    print(f"Solution stats: min={np.min(arr):.6e}, max={np.max(arr):.6e}, mean={np.mean(arr):.6e}")
    return c_sol

# ========================================================
# Diffusion-Only Solver
# ========================================================

def pure_diffusion_solver(mesh_results, C, D, mu, mesh_type="sulcus"):
    """
    Solve the steady-state diffusion equation for concentration field.
    """

    # Extract mesh data based on type
    mesh = mesh_results['mesh']
    bc_markers = mesh_results['bc_markers']

    # ----------------------------------------------------
    # Boundary Conditions
    # ----------------------------------------------------

    # Dirichlet BC on left and right (consistent markers: 1=left, 2=right)
    bc_left = DirichletBC(C, Constant(1.0), bc_markers, 1)   # C = 1.0
    bc_right = DirichletBC(C, Constant(0.0), bc_markers, 2)  # C = 0.0
    bcs_c = [bc_left, bc_right]

    # ----------------------------------------------------
    # Build Variational Form
    # ----------------------------------------------------

    # Define trial and test functions
    c_sol = TrialFunction(C)
    phi = TestFunction(C)

    # Variational form for steady-state diffusion: ∇²c = 0
    a_c = D * inner(grad(c_sol), grad(phi)) * dx

    # Add Robin boundary condition on bottom boundary (marker 4 for both mesh types)
    ds_bc = Measure("ds", domain=mesh, subdomain_data=bc_markers)
    a_c += mu * c_sol * phi * ds_bc(4)  # bottom boundary

    # Right-hand side (zero for steady-state diffusion)
    L_c = Constant(0.0) * phi * dx

    # Solve the linear system
    c_sol = Function(C)
    solve(a_c == L_c, c_sol, bcs_c)

    # Validate solution
    c_array = c_sol.vector().get_local()
    negative_count = np.sum(c_array < 0)

    if negative_count > 0:
        most_negative = np.min(c_array)

        # Only clamp if values are tiny (likely numerical noise)
        if abs(most_negative) < 1e-12:
            c_array = np.maximum(c_array, 0.0)
            c_sol.vector().set_local(c_array)
            c_sol.vector().apply("insert")
        else:
            print(f"WARNING: {negative_count} negative concentration values found!")
            print(f"  Most negative: {most_negative:.6e}")
            print(f"  Min: {np.min(c_array):.6e}, Max: {np.max(c_array):.6e}")
            print("  Check: mesh quality, boundary conditions, solver settings")
    else:
        print("✓ All concentration values are non-negative")

    print(f"Solution stats: min={np.min(c_array):.6e}, max={np.max(c_array):.6e}, mean={np.mean(c_array):.6e}")
    return c_sol

def pure_diffusion_solver_variable_mu(mesh_results, C, D, mu_function,
                                      mesh_type="rectangular", bottom_id=4, u=None):
    """
    Steady diffusion with spatially-varying Robin coefficient μ(x) on bottom boundary.
    Enforces C=1 at left (id=1), C=0 at right (id=2).
    """

    # Extract mesh and markers
    mesh = mesh_results['mesh']
    bc_markers = mesh_results['bc_markers']

    # Dirichlet BCs
    bc_left  = DirichletBC(C, Constant(1.0), bc_markers, 1)
    bc_right = DirichletBC(C, Constant(0.0), bc_markers, 2)
    bcs_c = [bc_left, bc_right]

    # Marked boundary measure
    ds_bc = Measure("ds", domain=mesh, subdomain_data=bc_markers)

    # Trial/Test
    c  = TrialFunction(C)
    v  = TestFunction(C)

    # Velocity (zero by default)
    if u is None:
        u = Constant((0.0, 0.0))

    # Clamp μ to nonnegative (avoid artificial sources from smoothing noise)
    mu_clamped = conditional(ge(mu_function, 0.0), mu_function, 0.0)

    # Variational forms
    a = (D * inner(grad(c), grad(v)) + inner(dot(u, grad(c)), v)) * dx \
        + mu_clamped * c * v * ds_bc(bottom_id)
    L = Constant(0.0) * v * dx

    # Solve
    c_sol_func = Function(C)
    solve(a == L, c_sol_func, bcs_c)

    # Diagnostics (optional)
    c_array = c_sol_func.vector().get_local()
    negative_count = np.sum(c_array < 0)
    if negative_count > 0:
        most_negative = np.min(c_array)
        if abs(most_negative) < 1e-12:
            # tiny negative due to round-off: clip safely
            c_array = np.maximum(c_array, 0.0)
            c_sol_func.vector().set_local(c_array)
            c_sol_func.vector().apply('insert')
        else:
            print(f"WARNING: {negative_count} negative concentration values found!")
            print(f"  Most negative: {most_negative:.6e}")
            print("  Check: mesh quality, BCs, solver settings")

    print(f"Solution stats: min={np.min(c_array):.6e}, max={np.max(c_array):.6e}, mean={np.mean(c_array):.6e}")
    return c_sol_func

# ========================================================
# Stokes Solver
# ========================================================

def stokes_solver(mesh_results, W, L_domain, H, mesh_type="sulcus"):
    """
    Solve the Stokes equations for incompressible flow with robust point constraint.
    """

    unique_vals = np.unique(mesh_results['bc_markers'].array())
    print(f"Boundary markers present: {unique_vals}")

    # Extract mesh data based on type
    bc_markers = mesh_results['bc_markers']
    mesh = mesh_results['mesh']

    # --------------------------------------------------------
    # Define Boundary Conditions
    # --------------------------------------------------------

    # Poiseuille flow inlet profile
    inflow = Expression(
        ("4.0*x[1]*(H - x[1])", "0.0"),
        H=H,
        degree=2
    )
    noslip = Constant((0.0, 0.0))

    # Boundary conditions using mesh markers
    bc_inlet = DirichletBC(W.sub(0), inflow, bc_markers, 1)           # left boundary
    bc_noslip_bottom = DirichletBC(W.sub(0), noslip, bc_markers, 4)   # bottom boundary
    bc_noslip_top = DirichletBC(W.sub(0), noslip, bc_markers, 3)      # top boundary
    bcs = [bc_inlet, bc_noslip_bottom, bc_noslip_top]

    # --------------------------------------------------------
    # Method 1: Try boundary-based point constraint first
    # --------------------------------------------------------

    try:
        # Use a point on the right boundary (outlet) - more reliable
        class OutletPoint(SubDomain):
            def inside(self, x, on_boundary):
                return (on_boundary and
                        near(x[0], L_domain, DOLFIN_EPS*1000) and  # Right boundary
                        near(x[1], H/2, H/10))  # Middle of right boundary, with tolerance

        outlet_point = OutletPoint()
        bc_p_point = DirichletBC(W.sub(1), Constant(0.0), outlet_point, method="pointwise")
        bcs_with_p = bcs + [bc_p_point]

        print(f"Trying pressure constraint at outlet center: ({L_domain}, {H/2})")

        # Try to solve with this constraint
        f = Constant((0.0, 0.0))
        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)

        # Use standard Stokes formulation (more stable)
        a = (inner(grad(u), grad(v)) * dx
             - div(v) * p * dx
             - q * div(u) * dx)
        L = inner(f, v) * dx

        # Solve directly (simpler and more robust)
        U = Function(W)
        solve(a == L, U, bcs_with_p)

        u, p = U.split(deepcopy=True)
        print(f"✓ Stokes solver completed using outlet point constraint for {mesh_type} mesh")
        return u, p

    except Exception as e:
        print(f"Outlet point constraint failed: {e}")
        raise RuntimeError("Unable to solve Stokes system.")

def stokes_solver_no_adv(V, Q):
    """Create zero velocity and pressure fields for compatibility with no advection."""
    u_zero = Function(V)
    u_zero.vector()[:] = 0.0

    p_zero = Function(Q)
    p_zero.vector()[:] = 0.0

    return u_zero, p_zero