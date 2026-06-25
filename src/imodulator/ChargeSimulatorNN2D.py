from __future__ import annotations

import os
import copy
from collections import OrderedDict

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from shapely.geometry import Polygon, MultiPolygon

from scipy.interpolate import RegularGridInterpolator

from imodulator import PhotonicDevice
from imodulator.PhotonicPolygon import (
    SemiconductorPolygon,
    MetalPolygon,
    InsulatorPolygon,
)

from imodulator.Config import config_instance
nn = config_instance.get_nextnanopy()


class ChargeSimulatorNN2D:
    """
    2D nextnano++ charge transport simulator for photonic device cross-sections.

    Mirrors the structure of ChargeSimulatorNN but operates on a full 2D cross-section
    defined by a user-supplied simulation_window polygon.  Key differences:

    - simulate2D{} in the nextnano global section
    - xgrid + ygrid built from polygon boundary coordinates
    - Device polygons mapped to nextnano polygon{} regions (arbitrary shapes)
    - MetalPolygon instances are auto-detected as ohmic contacts
      (first = contact1 / biased, second = contact2 / grounded)
    - Outputs loaded from 2D .fld files via nextnanopy DataFile
    - True 2D interpolation transferred to device.charge (no x-broadcast)
    - Native Ex, Ey field components; no sim_vector_norm approximation

    Workflow::

        window = shapely.geometry.box(xmin, ymin, xmax, ymax)   # in µm
        charge2d = ChargeSimulatorNN2D(
            device=modulator.device,
            simulation_window=window,
            bias_start_stop_step=[0, 5, 11],
        )
        charge2d.plot_with_window()          # inspect geometry
        charge2d.NNinputf.execute(show_log=False)
        charge2d.load_output_data()
        charge2d.transfer_results_to_device()
    """

    def __init__(
        self,
        device: PhotonicDevice,
        simulation_window: Polygon,
        inputfile_name: str = "quicksave2d",
        output_directory: str = nn.config.config['nextnano++']['outputdirectory'],
        temperature: float = 300.0,
        bias_start_stop_step: list = [0, 1, 1],
    ):
        """
        Args:
            device: PhotonicDevice with photo_polygons already configured.
            simulation_window: Shapely Polygon (in µm) defining the 2D simulation domain.
                All device polygons are clipped to this window.
            inputfile_name: Base name for the nextnano++ .in file.
            output_directory: Directory for nextnano output. Defaults to config value.
            temperature: Simulation temperature in Kelvin.
            bias_start_stop_step: [start, stop, n_steps] for contact1 bias sweep.
                contact2 is always grounded.
        """
        self.temperature = temperature
        self.inputfile_name = inputfile_name
        self.output_directory = output_directory
        self.photonicdevice = device
        self.bias_start_stop_step = bias_start_stop_step
        self.simulation_window = simulation_window

        self.optical_photopolygons = copy.deepcopy(self.photonicdevice.photo_polygons)
        self.polygon_entities = OrderedDict()
        for polygon in self.optical_photopolygons:
            self.polygon_entities[polygon.name] = polygon.polygon

        self._select_region(simulation_window)
        self._create_in_file()

    # ------------------------------------------------------------------
    # Geometry selection
    # ------------------------------------------------------------------

    def _select_region(self, simulation_window: Polygon):
        """
        Clip all device polygons to simulation_window.

        Semiconductors → self.region_polygons (OrderedDict name→Polygon)
        MetalPolygons  → self.contact_polygons (OrderedDict name→Polygon)
            order determines contact1 (biased) and contact2 (ground)
        """
        region_polygons = OrderedDict()
        contact_polygons = OrderedDict()
        names_to_print = []

        for polygon in self.optical_photopolygons:
            name = polygon.name
            geom = polygon.polygon

            if name in ("substrate", "background"):
                continue

            clipped = simulation_window.intersection(geom)
            if clipped.is_empty:
                continue

            # Keep only the largest piece when intersection yields a MultiPolygon
            if isinstance(clipped, MultiPolygon):
                clipped = max(clipped.geoms, key=lambda p: p.area)

            if isinstance(polygon, MetalPolygon):
                contact_polygons[name] = clipped
            elif isinstance(polygon, SemiconductorPolygon):
                region_polygons[name] = clipped
                for poly in self.photonicdevice.photo_polygons:
                    if poly.name == name:
                        poly.has_charge_transport_data = True
                names_to_print.append(name)

        if len(contact_polygons) < 2:
            raise ValueError(
                f"At least 2 MetalPolygons must intersect the simulation window. "
                f"Found: {list(contact_polygons.keys())}"
            )

        print("Charge transport regions:")
        print(*names_to_print, sep="\n")
        print(f"Contacts: {list(contact_polygons.keys())}")
        print(f"  contact1 (biased) = {list(contact_polygons.keys())[0]}")
        print(f"  contact2 (ground) = {list(contact_polygons.keys())[1]}")

        self.region_polygons = region_polygons
        self.contact_polygons = contact_polygons

    # ------------------------------------------------------------------
    # Input file construction
    # ------------------------------------------------------------------

    def _create_in_file(self):
        """Assemble and write the nextnano++ input file."""
        output_path = os.path.join(self.output_directory, f"{self.inputfile_name}.in")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        content_sections = [
            self._create_global_section(),
            self._create_grid_section(),
            self._create_structure_section(),
            self._create_impurities_section(),
            self._create_classical_section(),
            self._create_poisson_section(),
            self._create_currents_section(),
            self._create_contacts_section(),
            self._create_run_section(),
        ]
        self.complete_content = "\n".join(content_sections)

        with open(output_path, "w") as f:
            f.write(self.complete_content)

        print(f"Input file created: {output_path}")
        self.NNinputf = nn.InputFile(output_path)
        self.NNinputf.config = nn.config

    def _create_global_section(self):
        return f"""
        global{{
        simulate2D{{}}

        temperature = {self.temperature}
        substrate{{ name = "InP" }}
        crystal_zb{{
            x_hkl = [1, 0, 0]
            y_hkl = [0, 1, 0]
        }}
        }}
        """

    def _create_grid_section(self):
        """
        Build xgrid and ygrid from simulation_window bounds and polygon edge coordinates.

        For each axis, a grid line is placed at every polygon boundary position using
        the finest charge resolution of polygons that share that boundary.  The window
        edges always get a coarse background spacing of 2 nm.
        """
        bounds = self.simulation_window.bounds  # (xmin, ymin, xmax, ymax) µm
        xmin_nm, ymin_nm = bounds[0] * 1e3, bounds[1] * 1e3
        xmax_nm, ymax_nm = bounds[2] * 1e3, bounds[3] * 1e3

        # seed with window boundaries at coarse spacing
        x_positions = {round(xmin_nm, 4): 2.0, round(xmax_nm, 4): 2.0}
        y_positions = {round(ymin_nm, 4): 2.0, round(ymax_nm, 4): 2.0}

        all_regions = {**self.region_polygons, **self.contact_polygons}

        for name, poly in all_regions.items():
            res_nm = (
                self.photonicdevice.resolutions_charge.get(name, {}).get("resolution", 0.05)
                * 1e3  # µm → nm
            )
            for x, y in list(poly.exterior.coords):
                x_nm = round(x * 1e3, 4)
                y_nm = round(y * 1e3, 4)
                if xmin_nm <= x_nm <= xmax_nm:
                    x_positions[x_nm] = min(x_positions.get(x_nm, res_nm), res_nm)
                if ymin_nm <= y_nm <= ymax_nm:
                    y_positions[y_nm] = min(y_positions.get(y_nm, res_nm), res_nm)

        def build_lines(positions_dict):
            return "\n".join(
                f"\t\t\tline{{pos = {pos:.4f} spacing = {spacing:.4f}}}"
                for pos, spacing in sorted(positions_dict.items())
            )

        return f"""
        grid{{
            xgrid{{
        {build_lines(x_positions)}
            }}
            ygrid{{
        {build_lines(y_positions)}
            }}
        }}
        """

    def _polygon_to_nn_coords(self, poly: Polygon) -> str:
        """Return nextnano corner-coordinates string (µm coords converted to nm)."""
        # drop the repeated closing vertex
        coords = list(poly.exterior.coords)[:-1]
        return " ".join(f"{x * 1e3:.4f} {y * 1e3:.4f}" for x, y in coords)

    def _create_structure_section(self):
        """
        Map device polygons to nextnano region{} blocks.

        Semiconductor polygons are defined first; contact (MetalPolygon) regions are
        defined last so they take priority over any overlapping semiconductor material.
        """
        region_defs = []

        for name, poly in self.region_polygons.items():
            for p in self.optical_photopolygons:
                if p.name != name:
                    continue
                kw = p.charge_transport_simulator_kwargs
                material_def = kw["material_definition"] or "Ga(x)In(1-x)As(y)P(1-y)"
                corner_coords = self._polygon_to_nn_coords(poly)
                region_defs.append(f"""
            region{{
                polygon{{ corner-coordinates = {corner_coords} }}
                quaternary_constant{{
                    name = "{material_def}"
                    alloy_x = {kw["alloy_x"]:.4f}
                    alloy_y = {kw["alloy_y"]:.4f}
                }}
                doping{{
                    constant{{
                        name = "{kw["doping_type"]}-type"
                        conc = {kw["doping_conc"]:.4e}
                    }}
                }}
            }}
            """)
                break

        for i, (name, poly) in enumerate(self.contact_polygons.items()):
            contact_name = f"contact{i + 1}"
            corner_coords = self._polygon_to_nn_coords(poly)
            region_defs.append(f"""
            region{{
                polygon{{ corner-coordinates = {corner_coords} }}
                contact{{ name = "{contact_name}" }}
            }}
            """)

        return f"""
        structure{{
            output_region_index{{}}
            output_material_index{{}}
            output_user_index{{}}
            output_contact_index{{}}
            output_alloy_composition{{}}
            output_impurities{{}}

            region{{
                everywhere{{}}
                binary{{ name = 'Air' }}
            }}

        {"".join(region_defs)}
        }}
        """

    def _create_impurities_section(self):
        return """
        impurities{
            donor { name = "n-type" energy = -1000 degeneracy = 2 }
            acceptor { name = "p-type" energy = -1000 degeneracy = 4 }
        }"""

    def _create_classical_section(self):
        return """
        classical{
            Gamma{}
            HH{}

            output_bandedges{ averaged = yes }
            output_carrier_densities{}
        }"""

    def _create_poisson_section(self):
        return """
        poisson{
            charge_neutral{}
            output_electric_field{}
        }"""

    def _create_currents_section(self):
        return """
        currents{
            output_mobilities{}
            recombination_model{}
        }"""

    def _create_contacts_section(self):
        return f"""
        contacts{{
            ohmic{{ name = "contact1" bias = [{self.bias_start_stop_step[0]}, {self.bias_start_stop_step[1]}] steps = {self.bias_start_stop_step[2]}}}
            ohmic{{ name = "contact2" bias = 0.0 }}
        }}"""

    def _create_run_section(self):
        return """
        run{
            current_poisson{
                iterations = 10000
                output_log = yes
            }
        }"""

    # ------------------------------------------------------------------
    # Output loading
    # ------------------------------------------------------------------

    def load_output_data(self, folderpath=None):
        """
        Load 2D nextnano++ output from a completed bias sweep.

        Args:
            folderpath: Path to results folder. Uses default output dir if None.

        Reads bias-ordered subfolders containing .fld files.  Subfolders without
        any .fld files (e.g. the 'structure' folder) are silently skipped.

        Creates instance variables (all shape (n_bias, n_y, n_x)):
            grid_x  [nm]          grid_y  [nm]
            Ec      [eV]          Ev      [eV]
            Efn     [eV]          Efp     [eV]
            N       [cm⁻³]        P       [cm⁻³]
            Ex      [kV/cm]       Ey      [kV/cm]
            mun     [cm²/Vs]      mup     [cm²/Vs]
            V       [V]   (n_bias,)

        Note on variable names inside .fld files:
            nextnano++ names may differ slightly across versions.  If a KeyError
            occurs, inspect df.variables.keys() on the loaded DataFile object and
            update the patterns below accordingly.
        """
        if folderpath is None:
            nndata = nn.DataFolder(self.NNinputf.folder_output)
        else:
            nndata = nn.DataFolder(folderpath)

        f_iv = [f for f in nndata.files if "IV_characteristics.dat" in f][0]
        self.V = pd.read_csv(f_iv, delim_whitespace=True).iloc[:, 0]

        f_grid_x = [f for f in nndata.files if "grid_x.dat" in f][0]
        f_grid_y = [f for f in nndata.files if "grid_y.dat" in f][0]
        self.grid_x = pd.read_csv(f_grid_x, delim_whitespace=True).iloc[:, 0].values
        self.grid_y = pd.read_csv(f_grid_y, delim_whitespace=True).iloc[:, 0].values

        nx = len(self.grid_x)
        ny = len(self.grid_y)
        nV = len(self.V)

        self.Ec  = np.zeros((nV, ny, nx))
        self.Ev  = np.zeros((nV, ny, nx))
        self.Efn = np.zeros((nV, ny, nx))
        self.Efp = np.zeros((nV, ny, nx))
        self.N   = np.zeros((nV, ny, nx))
        self.P   = np.zeros((nV, ny, nx))
        self.Ex  = np.zeros((nV, ny, nx))
        self.Ey  = np.zeros((nV, ny, nx))
        self.mun = np.zeros((nV, ny, nx))
        self.mup = np.zeros((nV, ny, nx))

        # Keep only subfolders that contain .fld output (skips 'structure' etc.)
        bias_folders = sorted(
            [
                nndata.folders[i]
                for i in range(len(nndata.folders))
                if any(".fld" in f for f in nndata.folders[i].files)
            ],
            key=lambda folder: folder.folder,
        )

        if len(bias_folders) != nV:
            print(
                f"Warning: found {len(bias_folders)} bias folders but {nV} bias points "
                f"in IV file. Loading min({len(bias_folders)}, {nV}) points."
            )

        for i, folder in enumerate(bias_folders[: nV]):
            files = folder.files

            def _load(pattern, var_name, files=files):
                matches = [f for f in files if pattern in f and ".fld" in f]
                if not matches:
                    return None
                df = nn.DataFile(matches[0], product="nextnano++")
                return df.variables[var_name].value.reshape(ny, nx)

            self.Ec[i]  = _load("bandedges", "Gamma")
            self.Ev[i]  = _load("bandedges", "HH")
            self.Efn[i] = _load("bandedges", "electron_Fermi_level")
            self.Efp[i] = _load("bandedges", "hole_Fermi_level")
            self.N[i]   = _load("density_electron", "density")
            self.P[i]   = _load("density_hole", "density")
            self.Ex[i]  = _load("electric_field", "x")
            self.Ey[i]  = _load("electric_field", "y")
            self.mun[i] = _load("mobility_electron", "mobility")
            self.mup[i] = _load("mobility_hole", "mobility")

    # ------------------------------------------------------------------
    # Transfer to device
    # ------------------------------------------------------------------

    def transfer_results_to_device(self):
        """
        Build RegularGridInterpolators from 2D simulation data and store on the device.

        No xmin/xmax/dx arguments required — the data is already on the correct 2D mesh.
        The E-field is a proper (Ex, Ey, 0) vector field, not a 1D projection.

        Writes self.photonicdevice.charge with keys matching ChargeSimulatorNN:
            Ec, Ev, Efn, Efp, N, P, Efield, mun, mup  (lists of callables, one per bias)
            V  (array of bias voltages)
        """
        reg = self.photonicdevice.reg
        x_um = self.grid_x * 1e-3  # nm → µm
        y_um = self.grid_y * 1e-3

        def _interp(arr_2d):
            return RegularGridInterpolator(
                (y_um, x_um),
                arr_2d,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

        Ec_int = []; Ev_int = []; Efn_int = []; Efp_int = []
        N_int = []; P_int = []; Efield_int = []
        mun_int = []; mup_int = []

        for i in range(len(self.V)):
            Ec_int.append(
                lambda x, y, a=self.Ec[i]: _interp(a)((y, x)) * reg.eV
            )
            Ev_int.append(
                lambda x, y, a=self.Ev[i]: _interp(a)((y, x)) * reg.eV
            )
            Efn_int.append(
                lambda x, y, a=self.Efn[i]: _interp(a)((y, x)) * reg.eV
            )
            Efp_int.append(
                lambda x, y, a=self.Efp[i]: _interp(a)((y, x)) * reg.eV
            )
            N_int.append(
                lambda x, y, a=self.N[i]: _interp(a)((y, x)) * reg.cm ** -3
            )
            P_int.append(
                lambda x, y, a=self.P[i]: _interp(a)((y, x)) * reg.cm ** -3
            )
            Efield_int.append(
                lambda x, y, ex=self.Ex[i], ey=self.Ey[i]: np.stack(
                    [_interp(ex)((y, x)), _interp(ey)((y, x)), np.zeros_like(x)],
                    axis=-1,
                ) * reg.kV / reg.cm
            )
            mun_int.append(
                lambda x, y, a=self.mun[i]: _interp(a)((y, x)) * reg.cm ** 2 / reg.V / reg.s
            )
            mup_int.append(
                lambda x, y, a=self.mup[i]: _interp(a)((y, x)) * reg.cm ** 2 / reg.V / reg.s
            )

        self.photonicdevice.charge = {
            "Ec": Ec_int,
            "Ev": Ev_int,
            "Efn": Efn_int,
            "Efp": Efp_int,
            "N": N_int,
            "P": P_int,
            "Efield": Efield_int,
            "mun": mun_int,
            "mup": mup_int,
            "V": self.V,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_results(self, V_idx=None, quantity="Ec"):
        """
        Colour-map plot of a 2D quantity at selected bias points.

        Args:
            V_idx: List of bias indices to plot. Defaults to first and last.
            quantity: One of 'Ec', 'Ev', 'Efn', 'Efp', 'N', 'P', 'Ex', 'Ey',
                      'mun', 'mup'.

        Returns:
            fig, axes
        """
        data_map = {
            "Ec": (self.Ec, "Conduction band edge (eV)"),
            "Ev": (self.Ev, "Valence band edge (eV)"),
            "Efn": (self.Efn, "Electron quasi-Fermi level (eV)"),
            "Efp": (self.Efp, "Hole quasi-Fermi level (eV)"),
            "N": (self.N, "Electron density (cm⁻³)"),
            "P": (self.P, "Hole density (cm⁻³)"),
            "Ex": (self.Ex, "Electric field Ex (kV/cm)"),
            "Ey": (self.Ey, "Electric field Ey (kV/cm)"),
            "mun": (self.mun, "Electron mobility (cm²/Vs)"),
            "mup": (self.mup, "Hole mobility (cm²/Vs)"),
        }
        if quantity not in data_map:
            raise ValueError(f"quantity must be one of {list(data_map.keys())}")

        arr, label = data_map[quantity]

        if V_idx is None:
            V_idx = [0, len(self.V) - 1]

        fig, axes = plt.subplots(1, len(V_idx), figsize=(5 * len(V_idx), 4), squeeze=False)
        x_um = self.grid_x * 1e-3
        y_um = self.grid_y * 1e-3

        for col, vi in enumerate(V_idx):
            ax = axes[0][col]
            im = ax.pcolormesh(x_um, y_um, arr[vi], shading="auto", cmap="viridis")
            fig.colorbar(im, ax=ax, label=label)
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            ax.set_title(f"V = {self.V.iloc[vi]:.2f} V")
            ax.set_aspect("equal")

        fig.suptitle(quantity)
        fig.tight_layout()
        return fig, axes

    def plot_with_window(
        self,
        color_polygon="black",
        color_contact="red",
        color_window="blue",
        fill_polygons=False,
        fig=None,
        ax=None,
    ):
        """
        Plot device cross-section with the simulation window and detected regions.

        Args:
            color_polygon: Colour for semiconductor polygon outlines.
            color_contact: Colour for metal contact polygon outlines.
            color_window: Colour for the simulation window boundary.
            fill_polygons: Fill polygons with random colours if True.
            fig, ax: Optional existing matplotlib figure/axis.

        Returns:
            fig, ax
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        for name, poly in self.polygon_entities.items():
            if not hasattr(poly, "exterior"):
                continue
            ax.plot(*poly.exterior.xy, color=color_polygon, linewidth=0.8)
            if fill_polygons:
                ax.fill(*poly.exterior.xy, alpha=0.3, color=np.random.rand(3))

        for name, poly in self.region_polygons.items():
            ax.fill(*poly.exterior.xy, alpha=0.25, color="steelblue")
            cx, cy = poly.centroid.x, poly.centroid.y
            ax.text(cx, cy, name, fontsize=6, ha="center", va="center")

        for i, (name, poly) in enumerate(self.contact_polygons.items()):
            ax.fill(*poly.exterior.xy, alpha=0.5, color=color_contact)
            cx, cy = poly.centroid.x, poly.centroid.y
            ax.text(cx, cy, f"contact{i+1}\n({name})", fontsize=6, ha="center", va="center")

        wx, wy = self.simulation_window.exterior.xy
        ax.plot(wx, wy, color=color_window, linewidth=1.5, linestyle="--", label="simulation window")

        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title("PhotonicDevice — 2D simulation window")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig, ax
