# Author: Niko Bleidistel
# last change: 2026-08-20

##############################################################################
# import packages
##############################################################################

import numpy as np
import pandas as pd

##############################################################################
##############################################################################
# theory helpers
##############################################################################
##############################################################################

MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

def biot_savart_vec_theo(
        pos: tuple[float, float, float],    
        R_con : list[tuple[float, float, float]],  
        I: float = 1e-6,
        isclose_tol: float = 1e-12,            
    ) -> tuple[float, float, float]:
    """
    Calculate the magnetic field at a point in space due to a finite straight current-carrying conductor using the Biot-Savart law.

    Args:
        pos (tuple[float, float, float]):               The observation point where the magnetic field is calculated (3D vector).
        R_con (list[tuple[float, float, float]]):       List of two conductor points defining the current element (3D vectors).
        I (float):                                      Current in Amperes (default: 1 μA).
        isclose_tol (float):                            Tolerance for floating-point comparisons.

    Returns:
        tuple: The magnetic field components (Bx, By, Bz) at the observation point.
    """
    if len(R_con) != 2:
        raise ValueError("R_con must contain exactly two points defining the start and end of the current element.")
    
    # Convert inputs to numpy arrays for vector math
    r_obs = np.array(pos)
    r1 = np.array(R_con[0])
    r2 = np.array(R_con[1])
    
    # Displacement vectors
    dl = r2 - r1           # Direction and length of the current element
    r1_obs = r_obs - r1    # Vector from start point to observation point
    r2_obs = r_obs - r2    # Vector from end point to observation point
    
    # Calculate magnitudes (distances)
    mag_r1 = np.linalg.norm(r1_obs)
    mag_r2 = np.linalg.norm(r2_obs)
    
    # Handle edge case: observation point lies exactly on one of the endpoints
    if np.isclose(mag_r1, 0.0, atol=isclose_tol) or np.isclose(mag_r2, 0.0, atol=isclose_tol):
        return (0.0, 0.0, 0.0)
        
    # Cross product (dl x r1_obs) to get the direction of B
    cross_prod = np.cross(dl, r1_obs)
    mag_cross = np.linalg.norm(cross_prod)
    
    # Handle edge case: observation point is collinear with the current element (cross product is zero)
    if np.isclose(mag_cross, 0.0, atol=isclose_tol):
        return (0.0, 0.0, 0.0)
    
    # Analytical integration for a finite straight filament
    # B = (mu_0 * I / (4 * pi)) * (dl x r1) / |dl x r1|^2 * [ dl . r2 / |r2| - dl . r1 / |r1| ]
    dot_r2 = np.dot(dl, r2_obs)
    dot_r1 = np.dot(dl, r1_obs)
    
    term_bracket = (dot_r2 / mag_r2) - (dot_r1 / mag_r1)        # scales the magnitude of B based on the resulting angles between the current element and the observation point 
    scalar_factor = (MU_0 * I) / (4 * np.pi * (mag_cross ** 2))
    
    B_vec = scalar_factor * term_bracket * cross_prod

    return B_vec

import numpy as np

def biot_savart_rectangular_conductor(
    pos: tuple[float, float, float],    
    R_con : list[tuple[float, float, float]],  
    I: float = 1e-6,
    isclose_tol: float = 1e-12, 

    width_vec: tuple[float, float, float] = (0.0, 0.0, 0.0),
    height_vec: tuple[float, float, float] = (0.0, 0.0, 0.0),
    num_w: int = 1,
    num_h: int = 1,
) -> tuple[float, float, float]:
    """
    Calculates the magnetic field of a straight, rectangular cross-section 
    conductor by approximating it as a grid of thin parallel filaments.
    
    Parameters:
    -----------
    pos :           Observation point (x, y, z)
    R_con :         List of two points defining the conductor's axis
    I :             Total current flowing through the conductor
    isclose_tol :   Tolerance for the underlying filament function

    width_vec :     Vector defining the width direction and full width dimension
    height_vec :    Vector defining the height direction and full height dimension
    num_w :         Number of filaments along the width axis
    num_h :         Number of filaments along the height axis
    """
    # Convert inputs to numpy arrays
    w_dir = np.array(width_vec)
    h_dir = np.array(height_vec)
    
    # Current per filament assuming uniform distribution
    total_filaments = num_w * num_h
    I_filament = I / total_filaments
    
    # Initialize total B-field vector
    B_total = np.zeros(3)
    
    # Generate linear spacing factors from -0.5 to 0.5 to shift from the center
    w_factors = np.linspace(-0.5, 0.5, num_w) if num_w > 1 else np.zeros(1)
    h_factors = np.linspace(-0.5, 0.5, num_h) if num_h > 1 else np.zeros(1)

    # Superposition: Loop through the cross-section grid
    for dw in w_factors:
        for dt in h_factors:
            # Shift vector from the center of the conductor
            shift = (dw * w_dir) + (dt * h_dir)
            
            # Displaced start and end points for the current filament
            r_con_filament = [tuple(r + shift) for r in R_con]
            
            # Call your base function for a single filament
            B_filament = biot_savart_vec_theo(
                pos=pos,
                R_con=r_con_filament,
                I=I_filament,
                isclose_tol=isclose_tol
            )
            
            B_total += np.array(B_filament)
            Bx, By, Bz = B_total  # Unpack the total magnetic field components

    return (Bx, By, Bz)


##############################################################################
##############################################################################

def power_per_length(
        I: float = 10e-3,                # current in A
        sigma: float = 61.6e6,          # conductivity in S/m
        h: float = 1e-6,                # height of the conductor in m
        w: float = 100e-6,              # width of the conductor in m
        ):
    """
    Calculate the theoretical power dissipation in a rectangular conductor.

    Args:
        I (float): Current in Amperes (default: 10 mA).
        sigma (float): Electrical conductivity in Siemens per meter (default: 61.6e6 S/m).
        h (float): Height of the conductor in meters (default: 1 μm).
        w (float): Width of the conductor in meters (default: 100 μm).

    Returns:
        float: Theoretical power dissipation (joule heating) in Watts per meter (W/m).
    """
    return I**2 / (sigma * h * w)

##############################################################################

# example df structure:
# df = pd.DataFrame({
#     "name": ["conductor", "insulator", "epilayer", "substrate"],
#     "width": [100e-6, 100e-6, 100e-6, 3e-3],  # width in meters
#     "thickness": [1e-6, 500e-9, 10e-6, 50e-6],  # thickness in meters
#     "thermal_conductivity": [429, 1.38, 450, 450],  # thermal conductivity in W/(m·K)
# })

##############################################################################
# old version of temp_est() is commented out below, kept for reference

# def temp_est(
#         df: pd.DataFrame,               # DataFrame with the layer data
#         power: float = power_per_length(), # Heating power in W/m (P/l in the PDF)
#         z: float = 0.0,                 # Distance from the conductor (z=0) in m, must be <= 0
#         y: float = 0.0,                 # Distance from the conductor (y=0) in m
#         T_iso: float = 293.15,          # Isothermal temperature in K (T_0 in the PDF)
#     ):
#     """
#     Estimate the temperature at a given point in space due to heat dissipation 
#     from a rectangular conductor using a combined 1D/2D analytical approach.

#     Args:
#         df (pd.DataFrame):  DataFrame containing the layer properties.
#         power (float):      Heating power in W/m (default: calculated using power_per_length()).
#         z (float):          Distance from the conductor along the z-axis (must be <= 0).
#         y (float):          Distance from the conductor along the y-axis.
#         T_iso (float):      Isothermal temperature in K (default: 293.15 K).
#     """
    
#     # 1. Identify conductor properties
#     cond_mask = df["name"] == "conductor"
#     w = float(df.loc[cond_mask, "width"].iloc[0])
    
#     # 2. Filter out the conductor from the thermal path calculation
#     layers_df = df[~cond_mask].copy().reset_index(drop=True)
    
#     # 3. Classify layers as narrow (1D heat flow) or wide (2D/logarithmic heat flow)
#     rtol = 1e-5  
#     atol = 1e-9  

#     # isclose logic
#     absolute_diff = (layers_df["width"] - w).abs()
#     tolerance_limit = atol + rtol * abs(w)

#     # apply logic to layer dataframe
#     layers_df["is_narrow"] = absolute_diff <= tolerance_limit
#     layers_df["is_wide"] = ~layers_df["is_narrow"] & (layers_df["width"] > w)

#     # 4. Calculate negative z-boundaries for each layer below the conductor (z=0)
#     cumsum_thickness = layers_df["thickness"].cumsum()
#     layers_df["z_end"] = -cumsum_thickness
#     layers_df["z_start"] = layers_df["z_end"] + layers_df["thickness"]
    
#     # Total thickness of all layers combined (isothermal boundary at negative z)
#     z_const = cumsum_thickness.iloc[-1]
    
#     # Clamp z to the valid physical domain: z must be between -z_const and 0.0
#     z = np.clip(z, -z_const, 0.0)

#     # 5. Find the transition boundary Z_0 between narrow stack and wide stack
#     # Z_0 is the bottom interface of the last narrow layer. 
#     # If the first layer is already wide, Z_0 remains 0.0.
#     narrow_layers = layers_df[layers_df["is_narrow"]]
#     Z_0 = narrow_layers["z_end"].iloc[-1] if not narrow_layers.empty else 0.0

#     # Define the core conformal mapping distance functions relative to the source interface (Z_0)
#     def r1(val_z):
#         return np.sqrt(y**2 + (val_z - Z_0)**2)

#     def r2(val_z):
#         # Image source mirrored at the isothermal boundary (z = -z_const)
#         return np.sqrt(y**2 + (val_z + 2 * z_const + Z_0)**2)

#     # 6. Piecewise Temperature Calculation
#     # Step A: Target z lies within the 1D narrow layer stack (closer to conductor, z >= Z_0)
#     if z >= Z_0:
#         T_Z0 = T_iso
#         wide_layers = layers_df[layers_df["is_wide"]]
        
#         # Integrate 2D thermal resistance upwards from the isothermal bottom to Z_0
#         for _, row in wide_layers.iterrows():
#             k_sub = row["thermal_conductivity"]
#             z_s = row["z_start"]  
#             z_e = row["z_end"]    
            
#             # Avoid the core singularity at the line source using the w/2 boundary condition
#             r1_s = max(r1(z_s), w / 2.0)
#             r1_e = max(r1(z_e), w / 2.0)
            
#             T_Z0 += (power / (np.pi * k_sub)) * np.log((r2(z_s) * r1_e) / (r1_s * r2(z_e)))
            
#         T = T_Z0


#         # Add the linear 1D resistance from the transition boundary Z_0 up to the target z
#         for _, row in narrow_layers.iterrows():
#             # Bestimme die Grenzen der Schicht im Intervall [Z_0, z]
#             overlap_bottom = max(Z_0, row["z_end"])
#             overlap_top = min(z, row["z_start"])

#             # Wenn die Schicht im aktiven Bereich liegt, ist top > bottom
#             if overlap_top > overlap_bottom:
#                 thickness_active = overlap_top - overlap_bottom
#                 T += power * (
#                     thickness_active / (row["thermal_conductivity"] * w)
#                 )

#     # Step B: Target z lies within the 2D wide layer stack (deeper in substrate, z < Z_0)
#     else:
#         T = T_iso
#         wide_layers = layers_df[layers_df["is_wide"]]
        
#         for _, row in wide_layers.iterrows():
#             k_sub = row["thermal_conductivity"]
#             z_s = row["z_start"]   
#             z_e = row["z_end"]     
            
#             # If the layer is entirely below (more negative than) z, it contributes its full profile
#             if z <= z_s:
#                 r1_s = max(r1(z_s), w / 2.0)
#                 r1_e = max(r1(z_e), w / 2.0)
#                 T += (power / (np.pi * k_sub)) * np.log((r2(z_s) * r1_e) / (r1_s * r2(z_e)))
            
#             # If target z falls inside this layer, integrate from the bottom boundary (z_e) up to z
#             elif z_e <= z < z_s:
#                 r1_z = max(r1(z), w / 2.0)
#                 r1_e = max(r1(z_e), w / 2.0)
#                 T += (power / (np.pi * k_sub)) * np.log((r2(z) * r1_e) / (r1_z * r2(z_e)))
#                 break  # Layers above the target coordinate do not influence its temperature

#     return T

##############################################################################
# old version of temp_est(), fits approximation, but not simulations data

def temp_est(
        df: pd.DataFrame,               # DataFrame with the layer data
        power: float = power_per_length(), # Heating power in W/m (P/l in the PDF)
        z: float = 0.0,                 # Distance from the conductor (z=0) in m, must be <= 0
        y: float = 0.0,                 # Distance from the conductor (y=0) in m
        T_iso: float = 293.15,          # Isothermal temperature in K (T_0 in the PDF)
    ):
    """
    Estimate the temperature at a given point in space due to heat dissipation 
    from a rectangular conductor using a combined 1D/2D analytical approach.

    Args:
        df (pd.DataFrame):  DataFrame containing the layer properties.
        power (float):      Heating power in W/m (default: calculated using power_per_length()).
        z (float):          Distance from the conductor along the z-axis (must be <= 0).
        y (float):          Distance from the conductor along the y-axis.
        T_iso (float):      Isothermal temperature in K (default: 293.15 K).
    """
    
    # 1. Identify conductor properties
    cond_mask = df["name"] == "conductor"
    w = float(df.loc[cond_mask, "width"].iloc[0])
    
    # 2. Filter out the conductor from the thermal path calculation
    layers_df = df[~cond_mask].copy().reset_index(drop=True)
    
    # 3. Classify layers as narrow (1D heat flow) or wide (2D/logarithmic heat flow)
    rtol = 1e-5  
    atol = 1e-9  

    # isclose logic
    absolute_diff = (layers_df["width"] - w).abs()
    tolerance_limit = atol + rtol * abs(w)

    # apply logic to layer dataframe
    layers_df["is_narrow"] = absolute_diff <= tolerance_limit
    layers_df["is_wide"] = ~layers_df["is_narrow"] & (layers_df["width"] > w)

    # 4. Calculate negative z-boundaries for each layer below the conductor (z=0)
    cumsum_thickness = layers_df["thickness"].cumsum()
    layers_df["z_end"] = -cumsum_thickness
    layers_df["z_start"] = layers_df["z_end"] + layers_df["thickness"]
    
    # Total thickness of all layers combined (isothermal boundary at negative z)
    z_const = cumsum_thickness.iloc[-1]
    
    # Clamp z to the valid physical domain: z must be between -z_const and 0.0
    z = np.clip(z, -z_const, 0.0)

    # 5. Find the transition boundary Z_0 between narrow stack and wide stack
    # Z_0 is the bottom interface of the last narrow layer. 
    # If the first layer is already wide, Z_0 remains 0.0.
    narrow_layers = layers_df[layers_df["is_narrow"]]
    Z_0 = narrow_layers["z_end"].iloc[-1] if not narrow_layers.empty else 0.0

    # 5. Definiere die Abstandsfunktionen relativ zur Quelle Z_0
    def r1(val_z):
        return np.sqrt(y**2 + (val_z - Z_0)**2)

    def r2(val_z):
        # Spiegelquelle an der isothermen Unterseite (z = -z_const)
        # Da Z_0 negativ ist, verschiebt '- Z_0' die Position korrekt nach unten
        return np.sqrt(y**2 + (val_z + 2 * z_const - Z_0)**2)


        # 6. Piecewise Temperature Calculation
    # Step A: Target z lies within the 1D narrow layer stack (z >= Z_0)
    if z >= Z_0:
        T_Z0 = T_iso
        wide_layers = layers_df[layers_df["is_wide"]]
        
        # Integrate 2D thermal resistance upwards from bottom to Z_0
        for _, row in wide_layers.iterrows():
            k_sub = row["thermal_conductivity"]
            z_s = row["z_start"]  
            z_e = row["z_end"]    
            
            r1_s = max(r1(z_s), w / 2.0)
            r1_e = max(r1(z_e), w / 2.0)
            
            T_Z0 += (power / (np.pi * k_sub)) * np.log((r2(z_s) * r1_e) / (r1_s * r2(z_e)))
            
        T = T_Z0

        # Add the linear 1D resistance from Z_0 up to the target z
        for _, row in narrow_layers.iterrows():
            overlap_bottom = max(Z_0, row["z_end"])
            overlap_top = min(z, row["z_start"])

            if overlap_top > overlap_bottom:
                thickness_active = overlap_top - overlap_bottom
                T += power * (thickness_active / (row["thermal_conductivity"] * w))

    # Step B: Target z lies within the 2D wide layer stack (z < Z_0)
    else:
        T = T_iso
        # WICHTIG: Von unten (Kühlkörper) nach oben durchlaufen
        wide_layers_reversed = layers_df[layers_df["is_wide"]].iloc[::-1]
        
        for _, row in wide_layers_reversed.iterrows():
            k_sub = row["thermal_conductivity"]
            z_s = row["z_start"]   # Top der Schicht (näher an 0)
            z_e = row["z_end"]     # Bottom der Schicht (näher an -z_const)
            
            # Schicht liegt komplett unterhalb des Ziel-z (bereits durchquert)
            if z_s <= z:
                r1_s = max(r1(z_s), w / 2.0)
                r1_e = max(r1(z_e), w / 2.0)
                # Korrektes Verhältnis: r1_e/r1_s für die Quelle, r2_s/r2_e für die Senke
                T += (power / (np.pi * k_sub)) * np.log((r1_e * r2(z_s)) / (r1_s * r2(z_e)))
            
            # Ziel-z liegt genau in dieser Schicht: Integriere von Schicht-Untergrenze (z_e) bis z
            elif z_e <= z < z_s:
                r1_z = max(r1(z), w / 2.0)
                r1_e = max(r1(z_e), w / 2.0)
                # Integration von der isothermen Grenze (z_e) hoch zum Punkt z
                T += (power / (np.pi * k_sub)) * np.log((r1_e * r2(z)) / (r1_z * r2(z_e)))
                break

    return T

##############################################################################


# def temp_est(
#         df: pd.DataFrame,               # DataFrame with the layer data
#         power: float = 1.0,             # Heating power in W/m (P/l)
#         z: float = 0.0,                 # Distance from the conductor (z=0) in m, must be <= 0
#         T_iso: float = 293.15,          # Isothermal temperature in K (T_0)
#         y: float = 0.0,                 # Distance from the conductor (y=0) in m
#     ):
#     """
#     Estimate the temperature at a given point in space due to heat dissipation 
#     from a rectangular conductor using a continuous bottom-up integration.
#     """
#     # 1. Identify conductor properties
#     cond_mask = df["name"] == "conductor"
#     w = float(df.loc[cond_mask, "width"].iloc[0])
    
#     # 2. Filter out the conductor from the thermal path calculation
#     layers_df = df[~cond_mask].copy().reset_index(drop=True)
    
#     # 3. Classify layers as narrow (1D heat flow) or wide (2D/logarithmic heat flow)
#     rtol = 1e-5  
#     atol = 1e-9  
#     absolute_diff = (layers_df["width"] - w).abs()
#     tolerance_limit = atol + rtol * abs(w)
#     layers_df["is_narrow"] = absolute_diff <= tolerance_limit
#     layers_df["is_wide"] = ~layers_df["is_narrow"] & (layers_df["width"] > w)

#     # 4. Calculate negative z-boundaries for each layer below the conductor (z=0)
#     cumsum_thickness = layers_df["thickness"].cumsum()
#     layers_df["z_end"] = -cumsum_thickness
#     layers_df["z_start"] = layers_df["z_end"] + layers_df["thickness"]
    
#     # Total thickness of all layers combined (isothermal boundary at negative z)
#     z_const = cumsum_thickness.iloc[-1]
    
#     # Clamp z to the valid physical domain: z must be between -z_const and 0.0
#     z = np.clip(z, -z_const, 0.0)

#     # 5. Find the transition boundary Z_0 between narrow stack and wide stack
#     narrow_layers = layers_df[layers_df["is_narrow"]]
#     Z_0 = narrow_layers["z_end"].iloc[-1] if not narrow_layers.empty else 0.0

#     # Pure distance functions to preserve the exact geometric curvature in the substrate
#     def r1(val_z):
#         return np.sqrt(y**2 + (val_z - Z_0)**2)

#     def r2(val_z):
#         # Image source mirrored at the isothermal bottom boundary (z = -z_const)
#         return np.sqrt(y**2 + (val_z + 2 * z_const + Z_0)**2)

#     # 6. Continuous Bottom-Up Integration
#     T = T_iso
    
#     # Effective core radius to eliminate the mathematical log-singularity at the interface
#     # Setting this to a fraction of the conductor width keeps the substrate curve perfectly intact
#     r_core = 0.15 * w  

#     # Iterate through all layers from bottom to top (reverse order of dataframe)
#     for _, row in layers_df.iloc[::-1].iterrows():
#         z_s = row["z_start"]  # Top of current layer (larger, less negative z)
#         z_e = row["z_end"]    # Bottom of current layer (smaller, more negative z)
        
#         if z <= z_e:
#             continue  # This layer is entirely above our target z
            
#         z_top_active = min(z, z_s)
        
#         # Case A: 2D Wide Substrate Layer
#         if row["is_wide"]:
#             # Protect the active r1 distance from reaching 0 right at the Z_0 interface
#             r1_active_top = max(r1(z_top_active), r_core)
#             r1_active_bot = max(r1(z_e), r_core)
            
#             T += (power / (np.pi * row["thermal_conductivity"])) * np.log(
#                 (r2(z_top_active) * r1_active_bot) / (r1_active_top * r2(z_e))
#             )
            
#         # Case B: 1D Narrow Layer (Insulator / Epilayer)
#         elif row["is_narrow"]:
#             active_thickness = z_top_active - z_e
#             T += power * (active_thickness / (row["thermal_conductivity"] * w))

#     return T
