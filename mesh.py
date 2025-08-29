##########################################################
# Mesh Module
##########################################################
"""
Generates sulcus and rectangular domains.
"""
# ========================================================
# region Imports
# ========================================================
import numpy as np
import os
from textwrap import dedent
import subprocess
import meshio
import logging
import numpy as np
from scipy.integrate import quad
from dolfin import *
import matplotlib.pyplot as plt

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
#endregion

# ========================================================
# region Mesh Generation Class
# ========================================================

class MeshGenerator:

    # --------------------------------------------------------
    # region Class Constants
    # --------------------------------------------------------

    # Default file names and extensions
    SULCUS_BASE, RECT_BASE = "sulcus_mesh", "rect_mesh"
    GEO, MSH, XML, PVD = ".geo", ".msh", ".xml", ".pvd"

    # No. of segments to generate along the sulcus dip
    N_SULCUS_SEGMENTS = 20

    # Boundary marker labels
    MARKERS = {
        'left': 1, 'right': 2, 'top': 3, 'bottom': 4,
        'bottom_left': 5, 'sulcus': 6, 'bottom_right': 7, 'sulcus_opening': 8,
        'y0_line': 10
    }

    # Tolerance settings
    TOLERANCE = 2 * DOLFIN_EPS

    # Sulcus point indexing
    SULCUS_LEFT_IDX = 5
    SULCUS_RIGHT_IDX = 6
    SULCUS_BASE_IDX = 10


    #endregion
    # --------------------------------------------------------
    # region Set-Up
    # --------------------------------------------------------

    def __init__(self,
                 width, height,
                 sulcus_depth, sulcus_width,
                 mesh_size, refinement_factor,
                 domain_type,
                 output_dir=None):

        # Setup output directory
        self.output_dir = os.path.abspath(output_dir) if output_dir else os.getcwd()
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"✗ Failed to create output directory {self.output_dir}: {e}")

        # Store and validate parameters
        self._store_parameters(width, height,
                               sulcus_depth, sulcus_width,
                               mesh_size, refinement_factor,
                               domain_type
        )
        self._validate_parameters()

        # Generate file paths
        self._setup_file_paths()

        # Initialise mesh storage
        self.sulcus_mesh = self.rect_mesh = None

    def _store_parameters(self,
                          width, height,
                          sulcus_depth, sulcus_width,
                          mesh_size, refinement_factor,
                          domain_type):
        """Store all parameters and calculate derived values"""
        self.width, self.height = width, height
        self.sulcus_depth, self.sulcus_width = sulcus_depth, sulcus_width
        self.mesh_size, self.refinement_factor = mesh_size, refinement_factor
        self.sulcus_left_x = width/2 - sulcus_width/2
        self.sulcus_right_x = width/2 + sulcus_width/2
        self.domain_type = domain_type

    def _validate_parameters(self):
        """Validate all input parameters"""
        valid_domain_types = ['sulcus', 'rectangular']
        validations = [
            (self.height > 0, "Channel height must be positive"),
            (self.width > 0, "Channel width must be positive"),
            (self.mesh_size > 0, "Mesh size must be positive"),
            (self.sulcus_width > 0, "Sulcus width must be positive"),
            (self.sulcus_depth > 0, "Sulcus depth must be positive"),
            (self.refinement_factor > 0, "Refinement factor must be positive"),
            (self.sulcus_width < self.width, "Sulcus width must be less than channel width"),
            (self.domain_type in valid_domain_types, f"domain_type must be one of {valid_domain_types}")
        ]
        for condition, message in validations:
            if not condition:
                raise ValueError(message)

    def _setup_file_paths(self):
        """Generate all file paths using class constants"""
        def join_path(base, ext):
            return os.path.join(self.output_dir, base + ext)
        self.sulcus_geo_fn = join_path(self.SULCUS_BASE, self.GEO)
        self.sulcus_msh_fn = join_path(self.SULCUS_BASE, self.MSH)
        self.sulcus_xml_fn = join_path(self.SULCUS_BASE, self.XML)
        self.sulcus_pvd_fn = join_path(self.SULCUS_BASE, self.PVD)
        self.rect_geo_fn = join_path(self.RECT_BASE, self.GEO)
        self.rect_msh_fn = join_path(self.RECT_BASE, self.MSH)
        self.rect_xml_fn = join_path(self.RECT_BASE, self.XML)
        self.rect_pvd_fn = join_path(self.RECT_BASE, self.PVD)

    #endregion
    # --------------------------------------------------------
    # region Generate Sulcus Points
    # --------------------------------------------------------

    def _generate_sulcus_points(self):
        """Generate sulcus point data"""
        sulcus_points = []

        # Create N_SULCUS_SEGMENTS+1 points to define sulcus curve
        for i in range(self.N_SULCUS_SEGMENTS + 1):

            # Convert segment index to relative position (0 to 1)
            x_rel = i / self.N_SULCUS_SEGMENTS

            # Convert relative position to absolute x-coordinate within sulcus bounds
            x_abs = self.sulcus_left_x + x_rel * self.sulcus_width

            # Create sinusoidal dip: sin(π*x) gives 0 at endpoints, max at center
            # Only apply depth to interior points (not at i=0 or i=N_SULCUS_SEGMENTS)
            y_abs = -self.sulcus_depth * np.sin(np.pi * x_rel) if 0 < i < self.N_SULCUS_SEGMENTS else 0.0
            sulcus_points.append((x_abs, y_abs))

        # Generate point definitions with indexing
        sulcus_point_indices = []
        sulcus_points_section = []

        # Starting index for intermediate sulcus points
        base_idx = 10

        # Convert points to Gmsh format with specific indexing scheme
        for i, (x, y) in enumerate(sulcus_points):

            # Assign special indices to sulcus opening points for boundary marking
            if i == 0:
                point_idx = 5  # Left sulcus opening
            elif i == len(sulcus_points) - 1:
                point_idx = 6  # Right sulcus opening
            else:
                point_idx = base_idx + i - 1  # Intermediate points

            # Create Gmsh point definition with fine mesh size
            sulcus_points_section.append(f'Point({point_idx}) = {{{x:.6f}, {y:.6f}, lc_fine}};')
            sulcus_point_indices.append(point_idx)

        return {
            'points_section': '\n'.join(sulcus_points_section),  # Gmsh point definitions
            'first_point_idx': 5,                                # Left sulcus opening index
            'last_point_idx': 6,                                 # Right sulcus opening index

            # Reversed order for spline creation (right to left for proper orientation)
            'spline_points': ','.join([str(idx) for idx in reversed(sulcus_point_indices)]),

            # Normal order for mesh refinement field
            'nodes_list': ','.join([str(idx) for idx in sulcus_point_indices])
        }

    #endregion
    # --------------------------------------------------------
    # region Boundary Marking
    # --------------------------------------------------------

    def _create_boundary_functions(self):
        """Create boundary marker functions for FEniCS"""

        # Define all boundary functions as a dictionary
        boundary_definitions = {
            "left": lambda x, on_boundary: on_boundary and near(x[0], 0.0, DOLFIN_EPS),
            "right": lambda x, on_boundary: on_boundary and near(x[0], self.width, DOLFIN_EPS),
            "top": lambda x, on_boundary: on_boundary and near(x[1], self.height, DOLFIN_EPS),
            "bottom": lambda x, on_boundary: on_boundary and x[1] <= 0.0,
            "bottom_left": lambda x, on_boundary: (on_boundary and near(x[1], 0.0, self.TOLERANCE)
                                                  and x[0] <= self.sulcus_left_x - DOLFIN_EPS),
            "bottom_right": lambda x, on_boundary: (on_boundary and near(x[1], 0.0, self.TOLERANCE)
                                                   and x[0] >= self.sulcus_right_x + DOLFIN_EPS),
            "sulcus": lambda x, on_boundary: (on_boundary and self.sulcus_left_x <= x[0] <= self.sulcus_right_x
                                             and x[1] < -DOLFIN_EPS),
            "sulcus_opening": lambda x, on_boundary: (near(x[1], 0.0, self.TOLERANCE)
                                                     and self.sulcus_left_x + DOLFIN_EPS < x[0] < self.sulcus_right_x - DOLFIN_EPS),
            "y0_line": lambda x, on_boundary: near(x[1], 0.0, self.TOLERANCE)
        }
        self.boundary_functions = boundary_definitions

    def _create_and_mark_boundaries(self, boundary_list, name_suffix, mesh):
        """Create MeshFunction and mark specified boundaries"""

        # Create object that will store integer markers for 1D surfaces
        markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

        # Initialise all boundary markers to 0
        markers.set_all(0)

        class BoundarySubdomain(SubDomain):
            """Wrapper to convert a boundary function into a FEniCS SubDomain."""

            def __init__(self, func):
                super().__init__()
                # Store the boundary identification function
                self.func = func

            def inside(self, x, on_boundary):
                # Delegate boundary check to the wrapped function
                return self.func(x, on_boundary)

        # Loop through each boundary name to be marked
        for boundary_name in boundary_list:

            # Check if we have both the function and marker ID for this boundary
            if boundary_name in self.boundary_functions and boundary_name in self.MARKERS:

                # Get the function that defines this boundary's geometry
                boundary_func = self.boundary_functions[boundary_name]

                # Get the integer ID to mark this boundary with
                marker_id = self.MARKERS[boundary_name]

                # Wrap the function in a FEniCS-compatible SubDomain
                subdomain = BoundarySubdomain(boundary_func)

                # Apply the marker ID to all mesh facets that match this boundary
                subdomain.mark(markers, marker_id)

        return markers

    #endregion
    # --------------------------------------------------------
    # region Generate Mesh
    # --------------------------------------------------------

    def _generate_geo_content(self, sulcus_data, is_sulcus=True):
        """Generate .geo file content for sulcus or rectangular mesh"""
        lc = self.mesh_size
        lc_fine = self.mesh_size / self.refinement_factor

        mesh_type = "sulcus" if is_sulcus else "rectangular"

        # Common content
        common_content = dedent(f"""\
            // Auto-generated {mesh_type} mesh

            // Mesh parameters
            lc = {lc};
            lc_fine = {lc_fine};

            // Geometry parameters
            width = {self.width};
            height = {self.height};
            sulcus_depth = {self.sulcus_depth};
            sulcus_width = {self.sulcus_width};

            // Rectangle corners
            Point(1) = {{0, 0, 0, lc}};
            Point(2) = {{width, 0, 0, lc}};
            Point(3) = {{width, height, 0, lc}};
            Point(4) = {{0, height, 0, lc}};

            // Sinusoidal sulcus points (from left to right)
            {sulcus_data['points_section']}
            """)

        if is_sulcus:
            # Sulcus-specific content
            sulcus_specific = dedent(f"""\
                // External boundary lines forming single closed domain
                Line(1) = {{4, 3}};           // Top: left to right
                Line(2) = {{3, 2}};           // Right: top to bottom
                Line(3) = {{2, {sulcus_data['last_point_idx']}}};   // Right bottom flat
                Spline(4) = {{{sulcus_data['spline_points']}}};     // Sulcus curve
                Line(5) = {{{sulcus_data['first_point_idx']}, 1}};  // Left bottom flat
                Line(6) = {{1, 4}};           // Left: bottom to top

                // Create the main surface (single domain)
                Line Loop(1) = {{1, 2, 3, 4, 5, 6}};
                Plane Surface(1) = {{1}};

                // Internal line across sulcus opening
                Line(7) = {{{sulcus_data['first_point_idx']}, {sulcus_data['last_point_idx']}}};
                Line{{7}} In Surface{{1}};
                """)
        else:
            # Rectangular-specific content
            sulcus_specific = dedent(f"""\
                // External boundary lines forming rectangular domain
                Line(1) = {{4, 3}};           // Top: left to right
                Line(2) = {{3, 2}};           // Right: top to bottom
                Line(3) = {{2, 1}};           // Bottom: right to left
                Line(4) = {{1, 4}};           // Left: bottom to top

                // Create the main surface (single domain)
                Line Loop(1) = {{1, 2, 3, 4}};
                Plane Surface(1) = {{1}};
                """)

        # Common refinement and meshing settings
        refinement_content = dedent(f"""\
            // Mesh refinement near {'sulcus' if is_sulcus else 'imaginary sulcus'}
            Field[1] = Distance;
            Field[1].NodesList = {{{sulcus_data['nodes_list']}}};
            Field[2] = Threshold;
            Field[2].IField = 1;
            Field[2].LcMin = lc_fine;
            Field[2].LcMax = lc;
            Field[2].DistMin = {self.sulcus_width/10};
            Field[2].DistMax = {self.sulcus_width/2};

            Background Field = 2;

            // Force triangular meshing
            Mesh.Algorithm = 6;
            Mesh.RecombineAll = 0;
            Mesh.CharacteristicLengthExtendFromBoundary = 1;
            Mesh.CharacteristicLengthFromPoints = 1;
            """)

        return common_content + sulcus_specific + refinement_content

    def _run_gmsh(self, geo_file, msh_file, mesh_type="sulcus"):
        """Run Gmsh with error handling"""
        try:
            cmd = ["gmsh", geo_file, "-2", "-format", "msh2", "-algo", "del2d", "-smooth", "1"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if result.returncode == 0:
                return True
            else:
                logging.error(f"Gmsh error: {result.stderr}")
                return False
        except FileNotFoundError:
            logging.error("Gmsh not found. Please install Gmsh.")
            return False

    def _convert_msh_to_xml(self, msh_file, xml_file, mesh_type="sulcus"):
        """Convert .msh to .xml with 2D forcing"""

        # Note:
        # Using legacy XML format for FEniCS compatibility

        # Temporarily suppress legacy .xml warnings
        original_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

        try:
            if not os.path.exists(msh_file):
                logging.error(f"{mesh_type.capitalize()} mesh file not found.")
                return False
            mesh = meshio.read(msh_file)

            # Convert to 2D by stripping Z dimension if present
            if mesh.points.shape[1] >= 3:
                mesh.points = mesh.points[:, :2]

            meshio.write(xml_file, mesh, file_format="dolfin-xml")
            return True

        except Exception as e:
            logging.error(f"{mesh_type.capitalize()} meshio conversion error: {e}")
            return False
        finally:
            logging.getLogger().setLevel(original_level)

    def _generate_sulcus_mesh(self):
        """Generate sulcus mesh"""

        # Step 1: Generate sulcus points and .geo file
        try:
            sulcus_data = self._generate_sulcus_points()
            content = self._generate_geo_content(sulcus_data, is_sulcus=True)

            with open(self.sulcus_geo_fn, 'w') as f:
                f.write(content)
        except Exception as e:
            logging.error(f"Error generating sulcus geometry: {e}")
            return None

        # Step 2: Generate .msh file
        if not self._run_gmsh(self.sulcus_geo_fn, self.sulcus_msh_fn, "sulcus"):
            return None

        # Step 3: Convert .msh to .xml
        if not self._convert_msh_to_xml(self.sulcus_msh_fn, self.sulcus_xml_fn, "sulcus"):
            return None

        # Step 4: Load mesh and create boundaries
        try:
            if not os.path.exists(self.sulcus_xml_fn):
                logging.error(f"Sulcus XML file not found: {os.path.basename(self.sulcus_xml_fn)}")
                return None

            self.sulcus_mesh = Mesh(self.sulcus_xml_fn)
            self._create_boundary_functions()

            # Create different marker sets
            boundary_sets = {
                "bc": (["left", "right", "top", "bottom"], "boundary condition"),
                "bottom_segment": (["bottom_left", "bottom_right", "sulcus", "sulcus_opening"], "bottom segment"),
                "y0": (["y0_line"], "y0 line")
            }

            markers = {}
            for marker_type, (boundary_list, name_suffix) in boundary_sets.items():
                markers[f"sulcus_{marker_type}_markers"] = self._create_and_mark_boundaries(boundary_list, name_suffix, self.sulcus_mesh)

            # Store markers as instance attributes
            self.sulcus_bc_markers = markers["sulcus_bc_markers"]
            self.sulcus_bottom_segment_markers = markers["sulcus_bottom_segment_markers"]
            self.sulcus_y0_markers = markers["sulcus_y0_markers"]

        except Exception as e:
            logging.error(f"Error loading sulcus mesh and creating boundaries: {e}")
            return None

        # Step 5: Create domain markers
        try:
            domain_markers = MeshFunction("size_t", self.sulcus_mesh, self.sulcus_mesh.topology().dim())
            domain_markers.set_all(0)

            for cell in cells(self.sulcus_mesh):
                center_y = cell.midpoint().y()
                domain_markers[cell] = 1 if center_y <= 0.0 else 2  # Sulcus : Rectangle

            self.domain_markers = domain_markers
        except Exception as e:
            logging.error(f"Error creating sulcus domain markers: {e}")
            return None

        return sulcus_data

    def _generate_rectangular_mesh(self, sulcus_data):
        """Generate rectangular mesh"""

        # Step 1: Generate rectangular .geo file
        try:
            rect_geo_content = self._generate_geo_content(sulcus_data, is_sulcus=False)

            with open(self.rect_geo_fn, 'w') as f:
                f.write(rect_geo_content)
        except Exception as e:
            logging.error(f"Error generating rectangular geometry: {e}")
            return False

        # Step 2: Generate rectangular .msh file
        if not self._run_gmsh(self.rect_geo_fn, self.rect_msh_fn, "rectangular"):
            return False

        # Step 3: Convert rectangular .msh to .xml
        if not self._convert_msh_to_xml(self.rect_msh_fn, self.rect_xml_fn, "rectangular"):
            return False

        # Step 4: Load rectangular mesh and create boundaries
        try:
            if not os.path.exists(self.rect_xml_fn):
                logging.error(f"Rectangular XML file not found: {os.path.basename(self.rect_xml_fn)}")
                return False

            self.rect_mesh = Mesh(self.rect_xml_fn)

            # Ensure boundary functions exist
            if not hasattr(self, 'boundary_functions') or not self.boundary_functions:
                self._create_boundary_functions()

            # Create rectangular boundary markers using the reusable method
            boundary_list = ["left", "right", "top", "bottom"]
            self.rect_bc_markers = self._create_and_mark_boundaries(
                boundary_list, "boundary condition", self.rect_mesh)

        except Exception as e:
            logging.error(f"Error in creating rectangular boundary markers: {e}")
            return False

        return True

    def generate_mesh(self):
        """
        Generate and return a FEniCS-compatible mesh for the selected domain type.
        Note: uses legacy DOLFIN XML mesh files due to compatibility with older FEniCS versions.
        To future-proof, consider transitioning to XDMF format.

        This function runs the full mesh generation pipeline, including:
        - Gmsh geometry and mesh file creation
        - Conversion to DOLFIN XML format (2D enforced)
        - Mesh loading and boundary marking
        - Domain region marking (for sulcus meshes)

        Returns:
            dict: A dictionary containing mesh objects and metadata.

            If domain_type == 'sulcus':
                {
                    "mesh": dolfin.Mesh,                               # Full sulcus mesh
                    "bc_markers": MeshFunction("size_t"),              # Marks left, right, top, and bottom boundaries
                    "bottom_segment_markers": MeshFunction("size_t"),  # Marks sulcus floor, flanks, and openings
                    "y0_markers": MeshFunction("size_t"),              # Marks the full y = 0 line
                    "domain_markers": MeshFunction("size_t"),          # Region label: 1 = sulcus, 2 = rectangular
                    "mesh_info": {
                        "num_vertices": int,
                        "num_cells": int,
                        "hmin": float,
                        "hmax": float
                    }
                }

            If domain_type == 'rectangular':
                {
                    "mesh": dolfin.Mesh,                              # Rectangular domain mesh
                    "bc_markers": MeshFunction("size_t"),             # Marks left, right, top, and bottom boundaries
                    "mesh_info": {
                        "num_vertices": int,
                        "num_cells": int,
                        "hmin": float,
                        "hmax": float
                    }
                }

        Returns None if mesh generation fails at any stage.
        """

        logging.info(f"Generating {self.domain_type} mesh...")

        if self.domain_type == 'sulcus':

            sulcus_data = self._generate_sulcus_mesh()
            if sulcus_data is None:
                return None

            logging.info("✓ Sulcus mesh generation complete")
            return {
                "mesh": self.sulcus_mesh,
                "bc_markers": self.sulcus_bc_markers,
                "bottom_segment_markers": self.sulcus_bottom_segment_markers,
                "y0_markers": self.sulcus_y0_markers,
                "domain_markers": self.domain_markers,
                "mesh_info": {
                    "num_vertices": self.sulcus_mesh.num_vertices(),
                    "num_cells": self.sulcus_mesh.num_cells(),
                    "hmin": self.sulcus_mesh.hmin(),
                    "hmax": self.sulcus_mesh.hmax()
                }
            }

        elif self.domain_type == 'rectangular':

            # Generate sulcus_data for rectangular mesh refinement
            try:
                sulcus_data = self._generate_sulcus_points()
            except Exception as e:
                logging.error(f"Error generating sulcus points for rectangular mesh: {e}")
                return None

            if not self._generate_rectangular_mesh(sulcus_data):
                return None

            logging.info("✓ Rectangular mesh generation complete")
            return {
                "mesh": self.rect_mesh,
                "bc_markers": self.rect_bc_markers,
                "mesh_info": {
                    "num_vertices": self.rect_mesh.num_vertices(),
                    "num_cells": self.rect_mesh.num_cells(),
                    "hmin": self.rect_mesh.hmin(),
                    "hmax": self.rect_mesh.hmax()
                }
            }

        else:
            # This should never happen due to validation, but good to be explicit
            raise ValueError(f"Invalid domain_type: {self.domain_type}")

    def save_mesh_pvd_files(self, pvd_output_dir):
        """Save mesh and marker .pvd files to a separate directory for visualisation"""

        def _save_facet_normals(mesh, output_path, facet_markers=None, marker_id=None):
            """
            Project and save outward unit facet normals for specific boundaries.
            """
            try:
                # Create symbolic outward normal
                facet_normal_expr = FacetNormal(mesh)

                # Define vector DG(0) function space
                V_normals = VectorFunctionSpace(mesh, "DG", 0)
                v_trial_normals = TrialFunction(V_normals)
                v_test_normals = TestFunction(V_normals)
                normal_func = Function(V_normals)

                # Use either full ds or restricted ds
                if facet_markers is not None and marker_id is not None:
                    ds_custom = Measure("ds", domain=mesh, subdomain_data=facet_markers)
                    a_normals = inner(v_trial_normals, v_test_normals) * ds_custom(marker_id)
                    L_normals = inner(facet_normal_expr, v_test_normals) * ds_custom(marker_id)
                else:
                    # Default: entire external boundary
                    a_normals = inner(v_trial_normals, v_test_normals) * ds
                    L_normals = inner(facet_normal_expr, v_test_normals) * ds

                # Assemble and solve
                A_normals = assemble(a_normals)
                b_normals = assemble(L_normals)
                solve(A_normals, normal_func.vector(), b_normals)

                # Save to file
                File(output_path) << normal_func
                logging.info(f"✓ Facet normals saved to {os.path.basename(output_path)}")

            except Exception as e:
                logging.warning(f"Could not save normals to {os.path.basename(output_path)}: {e}")

        logging.info("Saving mesh visualisation files...")
        try:
            os.makedirs(pvd_output_dir, exist_ok=True)
            # Create normals subdirectory
            normals_dir = os.path.join(pvd_output_dir, "normals")
            os.makedirs(normals_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create PVD output directory: {e}")
            return

        if self.sulcus_mesh is not None:
            # Save sulcus mesh and markers
            sulcus_pvd = os.path.join(pvd_output_dir, "sulcus_mesh.pvd")
            File(sulcus_pvd) << self.sulcus_mesh

            # Save all marker sets
            if hasattr(self, 'sulcus_bc_markers'):
                File(os.path.join(pvd_output_dir, "sulcus_bc_markers.pvd")) << self.sulcus_bc_markers
            if hasattr(self, 'sulcus_bottom_segment_markers'):
                File(os.path.join(pvd_output_dir, "sulcus_bottom_markers.pvd")) << self.sulcus_bottom_segment_markers
            if hasattr(self, 'domain_markers'):
                File(os.path.join(pvd_output_dir, "sulcus_domains.pvd")) << self.domain_markers

            # Save normals for each boundary marker
            marker_groups = {
                "bc": {
                    "markers": ["left", "right", "top", "bottom"],
                    "meshfunction": self.sulcus_bc_markers
                },
                "bottom_segments": {
                    "markers": ["bottom_left", "sulcus", "bottom_right", "sulcus_opening"],
                    "meshfunction": self.sulcus_bottom_segment_markers
                },
                "y0": {
                    "markers": ["y0_line"],
                    "meshfunction": self.sulcus_y0_markers
                }
            }

            # Generate normals for each boundary
            for group_name, group_data in marker_groups.items():
                if hasattr(self, f'sulcus_{group_name.replace("_segments", "")}_markers'):
                    for marker_key in group_data["markers"]:
                        if marker_key in self.MARKERS:
                            marker_id = self.MARKERS[marker_key]
                            output_file = os.path.join(normals_dir, f"normals_{marker_key}_id{marker_id}.pvd")
                            _save_facet_normals(
                                mesh=self.sulcus_mesh,
                                facet_markers=group_data["meshfunction"],
                                marker_id=marker_id,
                                output_path=output_file
                            )

        if self.rect_mesh is not None:
            # Save rectangular mesh and markers
            rect_pvd = os.path.join(pvd_output_dir, "rect_mesh.pvd")
            File(rect_pvd) << self.rect_mesh

            if hasattr(self, 'rect_bc_markers'):
                File(os.path.join(pvd_output_dir, "rect_bc_markers.pvd")) << self.rect_bc_markers

                # Save normals for rectangular boundaries (1,2,3,4)
                basic_boundaries = ['left', 'right', 'top', 'bottom']
                for marker_key in basic_boundaries:
                    if marker_key in self.MARKERS:
                        marker_id = self.MARKERS[marker_key]
                        output_file = os.path.join(normals_dir, f"normals_{marker_key}_id{marker_id}.pvd")
                        _save_facet_normals(
                            mesh=self.rect_mesh,
                            facet_markers=self.rect_bc_markers,
                            marker_id=marker_id,
                            output_path=output_file
                        )

        logging.info(f"✓ Saved mesh visualisation files to {pvd_output_dir}")

    #endregion

# ========================================================
# Set-up Measures
# ========================================================

def setup_sulcus_measures(mesh, bc_markers, bottom_segment_markers, y0_markers, domain_markers):
    """Setup integration measures for sulcus mesh."""
    ds_bc_sulc = Measure('ds', domain=mesh, subdomain_data=bc_markers)
    ds_bottom = Measure('ds', domain=mesh, subdomain_data=bottom_segment_markers)
    dS_bottom = Measure('dS', domain=mesh, subdomain_data=bottom_segment_markers)
    ds_y0 = Measure('ds', domain=mesh, subdomain_data=y0_markers)
    dS_y0 = Measure('dS', domain=mesh, subdomain_data=y0_markers)
    dx_domain_sulc = Measure('dx', domain=mesh, subdomain_data=domain_markers)

    return ds_bc_sulc, ds_bottom, dS_bottom, ds_y0, dS_y0, dx_domain_sulc

def setup_rectangular_measures(mesh, bc_markers):
    """Setup integration measures for rectangular mesh."""
    ds_bc_rect = Measure('ds', domain=mesh, subdomain_data=bc_markers)
    dx_domain_rect = Measure('dx', domain=mesh)

    return ds_bc_rect, dx_domain_rect

# ========================================================
# Test Implementation
# ========================================================

if __name__ == "__main__":
    print("="*50)
    print("MESH GENERATOR TEST")
    print("="*50)

    test_params = {
        'width': 5.0, 'height': 1.0, 'sulcus_depth': 1,
        'sulcus_width': 0.5, 'mesh_size': 0.1, 'refinement_factor': 1.0,
        'domain_type': 'sulcus',
        'output_dir': "Results/Mesh Test/Mesh Files"
    }

    for domain_type in ['sulcus', 'rectangular']:
        test_params['domain_type'] = domain_type
        print(f"\nTesting {domain_type.upper()} domain...")

        try:
            mesh_gen = MeshGenerator(**test_params)
            mesh_results = mesh_gen.generate_mesh()

            if mesh_results is not None:
                mesh_gen.save_mesh_pvd_files('Results/Mesh Test/Paraview Files')
                print(f"✓ {domain_type.capitalize()} mesh test completed successfully!")
                print(f"  - Vertices: {mesh_results['mesh_info']['num_vertices']}")
                print(f"  - Cells: {mesh_results['mesh_info']['num_cells']}")
                print(f"  - hmin: {mesh_results['mesh_info']['hmin']:.6f}")
                print(f"  - hmax: {mesh_results['mesh_info']['hmax']:.6f}")
            else:
                print(f"✗ {domain_type.capitalize()} mesh test failed!")

        except Exception as e:
            print(f"✗ {domain_type.capitalize()} test error: {e}")
            import traceback
            traceback.print_exc()