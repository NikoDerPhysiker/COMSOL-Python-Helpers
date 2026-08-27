# Author: Niko Bleidistel
# last change: 2026-08-20

##############################################################################
# import packages
##############################################################################

import numpy as np
import pandas as pd

##############################################################################
##############################################################################
# Magnetic field
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

##############################################################################

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
    
    Args:
        pos (tuple[float, float, float])            : Observation point (x, y, z)
        R_con (list[tuple[float, float, float]])    : List of two points defining the conductor's axis (centerline of the conductor)
        I (float)                                   : Total current flowing through the conductor
        isclose_tol (float)                         : Tolerance for the underlying filament function

        width_vec (tuple[float, float, float])      : Vector defining the width direction and full width dimension (width of the conductor is the length of this vector)
        height_vec (tuple[float, float, float])     : Vector defining the height direction and full height dimension (height of the conductor is the length of this vector)
        num_w (int)                                 : Number of filaments along the width axis
        num_h (int)                                 : Number of filaments along the height axis

    Returns:
        tuple[float, float, float]: The total magnetic field components (Bx, By, Bz) at the observation point.
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
# Temperature
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
def temp_est_line_source_approximation(
        df: pd.DataFrame,               # DataFrame with the layer data
        power_per_length: float = power_per_length(), # Heating power in W/m (P/l in the PDF)
        z: float = 0.0,                 # Distance from the conductor (z=0) in m, must be <= 0
        y: float = 0.0,                 # Distance from the conductor (y=0) in m
        T_iso: float = 293.15,          # Isothermal temperature in K (T_0 in the PDF)
    ):
    """
    Estimate the temperature at a given point in space due to heat dissipation 
    from a rectangular conductor using a combined 1D/2D analytical approach and approximating the conductor as a line source.

    Args:
        df (pd.DataFrame):          DataFrame containing the layer properties.
        power_per_length (float):   Heating power in W/m (default: calculated using power_per_length()).
        z (float):                  Distance from the conductor along the z-axis (must be <= 0).
        y (float):                  Distance from the conductor along the y-axis.
        T_iso (float):              Isothermal temperature in K (default: 293.15 K).
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
    cumsum_height = layers_df["height"].cumsum()
    layers_df["z_end"] = -cumsum_height
    layers_df["z_start"] = layers_df["z_end"] + layers_df["height"]
    
    # Total thickness of all layers combined (isothermal boundary at negative z)
    z_const = cumsum_height.iloc[-1]
    
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
            
            T_Z0 += (power_per_length / (np.pi * k_sub)) * np.log((r2(z_s) * r1_e) / (r1_s * r2(z_e)))
            
        T = T_Z0

        # Add the linear 1D resistance from Z_0 up to the target z
        for _, row in narrow_layers.iterrows():
            overlap_bottom = max(Z_0, row["z_end"])
            overlap_top = min(z, row["z_start"])

            if overlap_top > overlap_bottom:
                thickness_active = overlap_top - overlap_bottom
                T += power_per_length * (thickness_active / (row["thermal_conductivity"] * w))

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
                T += (power_per_length / (np.pi * k_sub)) * np.log((r1_e * r2(z_s)) / (r1_s * r2(z_e)))
            
            # Ziel-z liegt genau in dieser Schicht: Integriere von Schicht-Untergrenze (z_e) bis z
            elif z_e <= z < z_s:
                r1_z = max(r1(z), w / 2.0)
                r1_e = max(r1(z_e), w / 2.0)
                # Integration von der isothermen Grenze (z_e) hoch zum Punkt z
                T += (power_per_length / (np.pi * k_sub)) * np.log((r1_e * r2(z)) / (r1_z * r2(z_e)))
                break

    return T

##############################################################################

def strip_source_integral(y: float, z_rel: float, w: float) -> float:
    """
    Analytical integration of the logarithmic temperature contribution from a finite-width strip source.
    """
    # Protect against exact zero distance singularities
    z_eff = np.maximum(np.abs(z_rel), 1e-12)
    
    left = y - w / 2.0
    right = y + w / 2.0
    
    # Indefinite integral of ln(y'^2 + z^2) dy' evaluated at boundaries
    term_r = right * np.log(right**2 + z_eff**2) - 2.0 * right + 2.0 * z_eff * np.arctan(right / z_eff)
    term_l = left * np.log(left**2 + z_eff**2) - 2.0 * left + 2.0 * z_eff * np.arctan(left / z_eff)
    
    return 0.5 * (term_r - term_l)

##############################################################################

def temp_est_integration_spreading(
    df: pd.DataFrame,               # DataFrame with the layer data
    power_per_length: float,        # Heating power in W/m (P/l)
    z: float = 0.0,                 # Distance from the conductor (z=0) in m, must be <= 0
    y: float = 0.0,                 # Distance from the conductor (y=0) in m
    T_iso: float = 293.15,          # Isothermal boundary temperature in K (T_0)
) -> float:
    """
    Estimate the temperature at a given point in space due to heat dissipation 
    from a rectangular conductor using a combined 1D/2D analytical approach,
    while integrating the finite-width strip source contributions and
    calculating the spreading of the heat flux by integration.

    Args:
        df (pd.DataFrame):          DataFrame containing the layer properties.
        power_per_length (float):   Heating power in W/m (P/l).
        z (float):                  Distance from the conductor along the z-axis (must be <= 0).
        y (float):                  Distance from the conductor along the y-axis.
        T_iso (float):              Isothermal boundary temperature in K (default: 293.15 K).
    """
    
    # 1. Identify conductor properties
    cond_mask = df["name"] == "conductor"
    w = float(df.loc[cond_mask, "width"].iloc[0])
    
    # 2. Filter out the conductor from the thermal path calculation
    layers_df = df[~cond_mask].copy().reset_index(drop=True)

    # 3. Calculate negative z-boundaries (0.0 at conductor, moving down to negative values)
    cumsum_height = layers_df["height"].cumsum()
    layers_df["z_end"] = -cumsum_height
    layers_df["z_start"] = layers_df["z_end"] + layers_df["height"]
    
    z_const = cumsum_height.iloc[-1]
    z = np.clip(z, -z_const, 0.0)

    # 4. Clean Physical Geometry (Top-Down)
    # Fully physical: The source width remains strictly the physical conductor width (100 µm).
    # The 2D geometric spreading is natively handled by using absolute distances from z=0.
    layers_df["w_eff_top"] = w
    layers_df["w_eff_bottom"] = w

    # 5. Unified 2D Temperature Calculation (Bottom-Up Integration)
    T = T_iso
    layers_reversed = layers_df.iloc[::-1]
    
    for _, row in layers_reversed.iterrows():
        z_s = row["z_start"]   
        z_e = row["z_end"]     
        w_layer = row["w_eff_top"]  # Always the physical width (100 µm)
        k_sub = row["thermal_conductivity"]
        
        # Determine active depth boundaries for this layer relative to target z
        overlap_bottom = z_e
        overlap_top = min(z_s, z)
        
        if overlap_top > overlap_bottom:
            # CRITICAL FIX: Measure distances absolutely from the real source at z=0.
            # Do NOT subtract z_s. This preserves the true continuous 2D fluid shape.
            z_rel_top = overlap_top
            z_rel_bottom = overlap_bottom
            
            # Primary source contribution (r1 equivalent)
            int_r1_top = strip_source_integral(y, z_rel_top, w_layer)
            int_r1_bottom = strip_source_integral(y, z_rel_bottom, w_layer)
            
            # Mirrored source contribution across the bottom isothermal boundary (z_const)
            # The mirror plane is at -2.0 * z_const relative to the real origin z=0
            z_mirror = -2.0 * z_const
            int_r2_top = strip_source_integral(y, overlap_top - z_mirror, w_layer)
            int_r2_bottom = strip_source_integral(y, overlap_bottom - z_mirror, w_layer)
            
            delta_integral_r1 = int_r1_bottom - int_r1_top
            delta_integral_r2 = int_r2_top - int_r2_bottom
            
            # Add scaled 2D temperature contribution based purely on local k_sub
            T += (power_per_length / (np.pi * k_sub * w_layer)) * (delta_integral_r1 + delta_integral_r2)
            
        # Terminate loop once the target depth z has been processed
        if z_e <= z <= z_s:
            break
            
    return T

##############################################################################


def temp_est_split_framework(
    df: pd.DataFrame, 
    power_per_length: float, 
    z: float = 0.0, 
    y: float = 0.0, 
    T_iso: float = 293.15
) -> float:
    """
    Estimate the temperature at a given point in space due to heat dissipation from a rectangular conductor.
    The last layer is treated as a 2D layer, while all layers above it are treated as 1D layers.
    """
    # 1. Leiterbahn filtern und Breite extrahieren
    cond_mask = df["name"] == "conductor"
    # KORREKTUR: .iloc[0] statt .iloc nutzen und explizit casten
    w = float(df.loc[cond_mask, "width"].iloc[0]) 
    
    layers_df = df[~cond_mask].copy().reset_index(drop=True)
    
    # Geometrische Höhen akkumulieren
    cumsum_height = layers_df["height"].cumsum()
    layers_df["z_end"] = -cumsum_height
    layers_df["z_start"] = layers_df["z_end"] + layers_df["height"]
    
    # Übergangsgrenze (Oberkante der alleruntersten Schicht)
    bottom_layer_name = str(layers_df["name"].iloc[-1])
    bottom_mask = layers_df["name"] == bottom_layer_name
    z_transition = float(layers_df.loc[bottom_mask, "z_start"].iloc[0])
    
    # 2. FALL A: Zielpunkt z liegt direkt in der untersten 2D-Schicht
    if z <= z_transition:
        cond_row = df[cond_mask]
        bottom_layer_row = df[~cond_mask].iloc[[-1]]
        sub_df = pd.concat([cond_row, bottom_layer_row]).reset_index(drop=True)
        
        return temp_est_integration_spreading(sub_df, power_per_length, z, y, T_iso)
        
    # 3. FALL B: Zielpunkt z liegt in den oberen 1D-Schichten
    else:
        cond_row = df[cond_mask]
        bottom_layer_row = df[~cond_mask].iloc[[-1]]
        sub_df = pd.concat([cond_row, bottom_layer_row]).reset_index(drop=True)
        
        T_transition = temp_est_integration_spreading(sub_df, power_per_length, z_transition, y, T_iso)
        
        T = T_transition
        upper_layers = layers_df.iloc[:-1]
        
        for _, row in upper_layers.iterrows():
            # KORREKTUR: Werte aus der iterrows-Schleife explizit casten
            row_z_end = float(row["z_end"])
            row_z_start = float(row["z_start"])
            row_k = float(row["thermal_conductivity"])
            
            overlap_bottom = max(z_transition, row_z_end)
            overlap_top = min(z, row_z_start)
            
            if overlap_top > overlap_bottom:
                thickness_active = overlap_top - overlap_bottom
                T += power_per_length * (thickness_active / (row_k * w))
                
        return T

##############################################################################


def temp_est_split_framework_merged(
    df: pd.DataFrame, 
    power_per_length: float, 
    z: float = 0.0, 
    y: float = 0.0, 
    T_iso: float = 293.15
) -> float:
    """
    Verschmilzt direkt übereinanderliegende Schichten mit identischer Leitfähigkeit und Breite.
    Nutzt das originale temp_est_integration_spreading() für den verbleibenden, 
    zusammenhängenden 2D-Bodenblock und rechnet darüber rein linear in 1D.
    """
    # 1. Conductor isolieren
    cond_mask = df["name"] == "conductor"
    w = float(df.loc[cond_mask, "width"].iloc[0]) 
    
    # Reine Materialschichten extrahieren
    layers_df = df[~cond_mask].copy().reset_index(drop=True)
    
    # 2. VERSCHMELZUNGS-LOGIK (Adjacent Layer Merging)
    # Ein Block wechselt, wenn sich Leitfähigkeit ODER Breite zum vorherigen Element ändern
    cond_change = layers_df["thermal_conductivity"] != layers_df["thermal_conductivity"].shift()
    width_change = layers_df["width"] != layers_df["width"].shift()
    
    # Kumulative Summe erzeugt eine eindeutige ID für jeden zusammenhängenden Block
    layers_df["block_id"] = (cond_change | width_change).cumsum()
    
    # Schichten aggregieren: Höhen addieren, Eigenschaften beibehalten (da identisch)
    # Wir nehmen "name" des jeweils ersten Schichtelements im Block
    merged_layers = layers_df.groupby("block_id").agg({
        "name": "first",
        "height": "sum",
        "thermal_conductivity": "first",
        "width": "first"
    }).reset_index(drop=True)
    
    # 3. Geometrische Höhen auf dem verschmolzenen DataFrame neu berechnen
    cumsum_height = merged_layers["height"].cumsum()
    merged_layers["z_end"] = -cumsum_height
    merged_layers["z_start"] = merged_layers["z_end"] + merged_layers["height"]
    
    # Dynamisch den Namen der untersten (jetzt verschmolzenen) Bodenschicht holen
    bottom_layer_name = str(merged_layers["name"].iloc[-1])
    bottom_mask = merged_layers["name"] == bottom_layer_name
    
    # z_transition ist die Oberkante des gesamten zusammenhängenden 2D-Bodenblocks
    z_transition = float(merged_layers.loc[bottom_mask, "z_start"].iloc[0])
    
    # 4. Rekonstruktion des Sub-DFs für die originale 2D-Funktion
    # Wir filtern beide Zeilen strikt auf die für die Berechnung relevanten Kernspalten
    core_columns = ["name", "height", "width", "thermal_conductivity"]
    
    cond_row_clean = df.loc[cond_mask, core_columns]
    bottom_merged_row_clean = merged_layers.iloc[[-1]][core_columns]
    
    # Sicherer Concat ohne Gefahr von KeyErrors durch Zusatzspalten
    sub_df_2d = pd.concat([cond_row_clean, bottom_merged_row_clean]).reset_index(drop=True)
    
    # 5. FALL A: Zielpunkt z liegt direkt im verschmolzenen 2D-Bodenblock
    if z <= z_transition:
        return temp_est_integration_spreading(sub_df_2d, power_per_length, z, y, T_iso)

    # 6. FALL B: Zielpunkt z liegt in den oberen 1D-Kanälen
    else:
        # Temperatur an der neuen, nach oben verschobenen Blockgrenze berechnen
        T_transition = temp_est_integration_spreading(sub_df_2d, power_per_length, z_transition, y, T_iso)
        
        T = T_transition
        # Alle verschmolzenen Schichten oberhalb des Bodenblocks durchlaufen
        upper_layers = merged_layers.iloc[:-1]
        
        for _, row in upper_layers.iterrows():
            row_z_end = float(row["z_end"])
            row_z_start = float(row["z_start"])
            row_k = float(row["thermal_conductivity"])
            row_w = float(row["width"])  # Nutzt die reale Breite des oberen Kanals
            
            overlap_bottom = max(z_transition, row_z_end)
            overlap_top = min(z, row_z_start)
            
            if overlap_top > overlap_bottom:
                thickness_active = overlap_top - overlap_bottom
                # 1D-Kanalwärmeleitung basierend auf der jeweiligen Kanalbreite (row_w)
                T += power_per_length * (thickness_active / (row_k * row_w))
                
        return T
