# Author: Niko Bleidistel
# last change: 2026-08-17

##############################################################################
# import packages
##############################################################################
import numpy as np
import math

##############################################################################
##############################################################################
# functions
##############################################################################
##############################################################################

def calculate_grid_voltages(
        z_target: float,
        B_goal: tuple[float, float, float],
        conductor_grid_length: float = 500e-6,
        conductor_grid_width: float = 2e-6,
        conductor_grid_height: float = 1e-6,
        conductivity: float = 6.30e7):
    """
    Calculates the 20 boundary terminal voltages required to generate a specific 
    target magnetic field vector (B_goal) at a central spatial point below or above 
    a 5x5 conductor crossbar grid.

    The model dynamically computes trace segment resistances based on material 
    electrical conductivity and geometric cross-sections. Terminal arm segments 
    extending to boundaries have exactly half the electrical resistance of inner segments.

    Terminal Numbering (1-indexed mapping to 0-19 in array):
    Starts at Pin 1 on the bottom-left edge and increments COUNTER-CLOCKWISE:
    - Pins 1 to 5: Bottom boundary (y = -0.5 * L, running left-to-right from index i=0 to 4)
    - Pins 6 to 10: Right boundary (x = 0.5 * L, running bottom-to-top from index j=0 to 4)
    - Pins 11 to 15: Top boundary (y = 0.5 * L, running right-to-left from index i=4 to 0)
    - Pins 16 to 20: Left boundary (x = -0.5 * L, running top-to-bottom from index j=4 to 0)

    Parameters:
    -----------
    z_target : float
        The vertical z-coordinate of the target center point in meters (must be non-zero).
    B_goal : tuple
        The desired 3D magnetic field target vector (Bx, By, Bz) in Tesla.
    conductor_grid_length : float, optional
        The overall side length of the square substrate footprint in meters. Default is 2 mm.
    conductor_grid_width : float, optional
        The cross-sectional width of internal conductor traces in meters. Default is 10 um.
    conductor_grid_height : float, optional
        The physical height/thickness of deposited metal traces in meters. Default is 3 um.
    conductivity : float, optional
        Material bulk electrical conductivity in S/m. Default is 6.30e7 S/m (bulk silver).

    Returns:
    --------
    V_terminals : numpy.ndarray
        An array of 20 calculated optimal voltage values corresponding to Pin 1 through Pin 20.
    """
    if np.isclose(z_target, 0.0):
        raise ValueError("z_target cannot be 0. Within the plane z=0, B_x and B_y are physically zero.")

    # 1. Spatial Geometry Coordinates Setup
    inner_coords = np.linspace(-(5 - 1) / (2 * 5) * conductor_grid_length, (5 - 1) / (2 * 5) * conductor_grid_length, 5)
    
    # 45 Nodes total: 0..24 are inner crossings, 25..44 are open terminal edges
    coords = np.zeros((45, 3))
    for i in range(5):
        for j in range(5):
            coords[i * 5 + j] = [inner_coords[i], inner_coords[j], 0.0]

    terminal_node_indices = np.zeros(20, dtype=int)
    
    # Pins 1 to 5: Bottom (y = -0.5 * L, x moves left-to-right)
    for idx, i in enumerate(range(5)):
        t_idx = 0 + idx
        terminal_node_indices[t_idx] = 25 + t_idx
        coords[terminal_node_indices[t_idx]] = [inner_coords[i], -0.5 * conductor_grid_length, 0.0]
        
    # Pins 6 to 10: Right (x = 0.5 * L, y moves bottom-to-top)
    for idx, j in enumerate(range(5)):
        t_idx = 5 + idx
        terminal_node_indices[t_idx] = 25 + t_idx
        coords[terminal_node_indices[t_idx]] = [0.5 * conductor_grid_length, inner_coords[j], 0.0]
        
    # Pins 11 to 15: Top (y = 0.5 * L, x moves right-to-left)
    for idx, i in enumerate(reversed(range(5))):
        t_idx = 10 + idx
        terminal_node_indices[t_idx] = 25 + t_idx
        coords[terminal_node_indices[t_idx]] = [inner_coords[i], 0.5 * conductor_grid_length, 0.0]
        
    # Pins 16 to 20: Left (x = -0.5 * L, y moves top-to-bottom)
    for idx, j in enumerate(reversed(range(5))):
        t_idx = 15 + idx
        terminal_node_indices[t_idx] = 25 + t_idx
        coords[terminal_node_indices[t_idx]] = [-0.5 * conductor_grid_length, inner_coords[j], 0.0]

    # 2. Material-Based Resistivity & Admittance Computation
    cross_section = conductor_grid_width * conductor_grid_height
    d_inner = inner_coords[1] - inner_coords[0]  # Pitch distance between intersection nodes
    
    # Resistance calculations via Pouillet's Law
    R_internal = d_inner / (conductivity * cross_section)
    R_arm = 0.5 * R_internal  # Terminal arm resistance constraint enforced here

    # 3. Define Trace Segments and Associate Specific Resistance Values
    segments = []
    
    # Horizontal paths (Left Pin -> Intersections -> Right Pin)
    for j in range(5):
        left_pin_node = terminal_node_indices[19 - j]   
        right_pin_node = terminal_node_indices[5 + j]   
        
        # Left boundary terminal arm
        segments.append((left_pin_node, 0 * 5 + j, R_arm))
        # Inner lattice paths
        for i in range(4):
            segments.append((i * 5 + j, (i + 1) * 5 + j, R_internal))
        # Right boundary terminal arm
        segments.append((4 * 5 + j, right_pin_node, R_arm))

    # Vertical paths (Bottom Pin -> Intersections -> Top Pin)
    for i in range(5):
        bottom_pin_node = terminal_node_indices[0 + i]  
        top_pin_node = terminal_node_indices[14 - i]   
        
        # Bottom boundary terminal arm
        segments.append((bottom_pin_node, i * 5 + 0, R_arm))
        # Inner lattice paths
        for j in range(4):
            segments.append((i * 5 + j, i * 5 + (j + 1), R_internal))
        # Top boundary terminal arm
        segments.append((i * 5 + 4, top_pin_node, R_arm))

    # 4. Assemble Internal Kirchhoff Admittance Matrix (25 x 25)
    A_kirchhoff = np.zeros((25, 25))
    for i in range(5):
        for j in range(5):
            n = i * 5 + j
            
            g_left  = 1.0 / R_internal if i > 0 else 1.0 / R_arm
            g_right = 1.0 / R_internal if i < 4 else 1.0 / R_arm
            g_down  = 1.0 / R_internal if j > 0 else 1.0 / R_arm
            g_up    = 1.0 / R_internal if j < 4 else 1.0 / R_arm
            
            A_kirchhoff[n, n] = g_left + g_right + g_down + g_up
            
            if i > 0: A_kirchhoff[n, (i - 1) * 5 + j] = -1.0 / R_internal
            if i < 4: A_kirchhoff[n, (i + 1) * 5 + j] = -1.0 / R_internal
            if j > 0: A_kirchhoff[n, i * 5 + (j - 1)] = -1.0 / R_internal
            if j < 4: A_kirchhoff[n, i * 5 + (j + 1)] = -1.0 / R_internal

    # 5. Superposition Loop to Construct Transmission Coupling Matrix M (3 x 20)
    M = np.zeros((3, 20))
    r_target = np.array([0.0, 0.0, z_target])
    mu_0_over_4pi = 1e-7

    for b_idx in range(20):
        V_boundary_excitation = np.zeros(20)
        V_boundary_excitation[b_idx] = 1.0
        
        b_kirchhoff = np.zeros(25)
        
        # Enforcing a single strict loop configuration using pure index equations
        for i in range(5):
            # Bottom Edge: Pins 1..5 map to inner nodes (i, 0)
            b_kirchhoff[i * 5 + 0] += V_boundary_excitation[0 + i] / R_arm
            
            # Right Edge: Pins 6..10 map to inner nodes (4, i)
            b_kirchhoff[4 * 5 + i] += V_boundary_excitation[5 + i] / R_arm
            
            # Top Edge: Pins 11..15 map to inner nodes (4-i, 4) -> running right-to-left
            b_kirchhoff[(4 - i) * 5 + 4] += V_boundary_excitation[10 + i] / R_arm
            
            # Left Edge: Pins 16..20 map to inner nodes (0, 4-i) -> running top-to-bottom
            b_kirchhoff[0 * 5 + (4 - i)] += V_boundary_excitation[15 + i] / R_arm
            
        V_inner = np.linalg.solve(A_kirchhoff, b_kirchhoff)
        
        V_all = np.zeros(45)
        V_all[0:25] = V_inner
        for t_idx in range(20):
            V_all[terminal_node_indices[t_idx]] = V_boundary_excitation[t_idx]
            
        B_column = np.zeros(3)
        for node_start, node_end, r_seg in segments:
            current = (V_all[node_start] - V_all[node_end]) / r_seg
            r_mid = (coords[node_start] + coords[node_end]) / 2.0
            dl = coords[node_end] - coords[node_start]
            R_vec = r_target - r_mid
            R_mag = np.linalg.norm(R_vec)
            
            dB = mu_0_over_4pi * current * np.cross(dl, R_vec) / (R_mag**3)
            B_column += dB
            
        M[:, b_idx] = B_column

    # 6. Inversion Matrix Solution Using Pseudoinverse
    M_pinv = np.linalg.pinv(M)
    V_terminals = -np.dot(M_pinv, B_goal)
    
    return V_terminals

def calculate_xy_vector(alpha: float, magnitude: float, in_degrees: bool = True):
    """
    Calculates a normalized 3D vector in the xy-plane.
    
    :param alpha: The angle relative to the vector (1,0,0)
    :param magnitude: The magnitude of the vector
    :param in_degrees: If True, angle is expected in degrees. 
                       If False, in radians.
    :return: A tuple (x, y, z) representing the normalized vector.
    """
    if in_degrees:
        alpha = math.radians(alpha)
        
    x = math.cos(alpha) * magnitude
    y = math.sin(alpha) * magnitude
    z = 0.0
    
    return (x, y, z)

def get_voltage_sweep_dict(
        angles: list,
        magnitude: float = 10e-6,
        conductor_grid_length: float = 500e-6
        ):
    """
    Generates a dictionary containing the calculated terminal voltages for a sweep of angles in the xy-plane.

    Args:
        angles (list): A list of angles (in degrees) for which to calculate the terminal voltages.
        magnitude (float): The magnitude of the target magnetic field vector. Default is 10e-6 Tesla.
        conductor_grid_length (float): The length of the conductor grid. Default is 500e-6 meters.

    Returns:
        dict: A dictionary where keys are terminal names (e.g., "V01", "V02", ..., "V20") and values are lists of calculated voltages corresponding to each angle in the sweep.
    """
    
    parameter_values = []

    for alpha in angles:
        # 1. Calculate target vector
        b_goal = calculate_xy_vector(alpha, magnitude, in_degrees=True)

        # 2. Simulate grid voltages
        v_terminals = calculate_grid_voltages(
            z_target=3e-6,
            B_goal=b_goal,
            conductor_grid_length=conductor_grid_length,
            conductor_grid_width=2e-6,
            conductor_grid_height=1e-6,
            conductivity=61.6e6,
        )
        
        # store parameter names
        parameter_names = [f"V{terminal+1:02d}" for terminal in range(len(v_terminals))]

        # store parameter values for this iteration in the array with all iterations
        iter_parameter_values = [f"{v:.4g}[V]" for v in v_terminals]
        parameter_values.append(iter_parameter_values)

    # return sweep_dict
    return parameter_names, parameter_values
