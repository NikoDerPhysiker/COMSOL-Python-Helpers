# Author: Niko Bleidistel
# last change: 2026-08-19

##############################################################################
# import packages
##############################################################################


from os import makedirs

import pandas as pd
import numpy as np

import matplotlib as mpl

from pathlib import Path
import importlib
import re

from IPython.display import display

# custom packages
import advanced_plotting_functions as apf
_ = importlib.reload(apf)

import theory_fit_models as tfm
_ = importlib.reload(tfm)

import comsol_data_import as cdi
_ = importlib.reload(cdi)

# for type hints
from typing import Callable
import matplotlib.figure
import matplotlib.axes

##############################################################################
##############################################################################
# theory wrapper
##############################################################################
##############################################################################
THEORY_PLOT_POINTS = 150

# translate theory plot labels to theory formula indices
THEORY_FORMULA = {
    "mf.Bx (T)":        0, 
    "mf.By (T)":        1, 
    "mf.Bz (T)":        2,
}

T_0 = 293.15 # [K] # reference temperature

##############################################################################
##############################################################################
def get_limits(ax: matplotlib.axes.Axes, xparam: str, min_distance: float) -> list[float]:
    # Get the x-coordinates of all scatter points and lines in the current axis
    all_x_data = []
    for collection in ax.collections:
        offsets = collection.get_offsets() # get the coordinates of the scatter points
        all_x_data.extend(offsets[:, 0]) # extend the list with the x-coordinates
    for line in ax.get_lines():
        all_x_data.extend(line.get_xdata()) # extend the list with the x-coordinates of the lines

    # Determine the x-limits based on the data
    if all_x_data:
        xlimits = [min(all_x_data), max(all_x_data)]
    else: 
        xlimits = list(ax.get_xlim()) # fallback to the current axis limits if no data is found

    # apply a minimum distance to the x-limits
    if xparam == "z":
        for i in range(2):
            if abs(xlimits[i]) < 0.5*min_distance:
                xlimits[i] = np.sign(xlimits[i]) * min_distance

    # Check if the x-limits are equal, which would prevent generating a theory curve
    if xlimits[0] == xlimits[1]:
        raise ValueError("xlimits are equal, cannot generate theory curve.")

    return xlimits

##############################################################################
##############################################################################

def add_magnetic_theory_01_00(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        df_parameters: pd.DataFrame,
        xparam: str,
        Bidx: int,
        min_distance:float = 1.5e-6,
        z_pos: float | None = None,
        color: str = "tab:red",
        mark_z: bool = False,
        )-> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Add a theoretical magnetic field curve to the provided matplotlib axis based on the Biot-Savart law.
    
    Args:
        fig (matplotlib.figure.Figure):             The matplotlib figure object.
        ax (matplotlib.axes.Axes):                  The matplotlib axes object where the theory curve will be added.
        df_parameters (pd.DataFrame):               DataFrame containing the necessary parameters for the Biot-Savart law calculation.
        Bidx (int):                                 Index indicating which component of the magnetic field to plot (0 for Bx, 1 for By, 2 for Bz).
        min_distance (float, optional):             Minimum distance for the x-limits. Defaults to 3e-6.
        z_pos (float | None, optional):             Optional z-position for the theory curve. If None, the z-position will be determined based on the xparam. Defaults to None.
        
    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The updated figure and axes objects with the theory curve added.
    """
    if xparam not in ["x", "y", "z"]:
        print(f"Warning: xparam '{xparam}' not in ['x', 'y', 'z']. Skipping theory curve generation.")
        return fig, ax 

    xlimits = get_limits(ax, xparam, min_distance)

    # Generate x values for the theory curve
    x = np.linspace(xlimits[0], xlimits[1], THEORY_PLOT_POINTS)

    # extract the necessary parameters from the DataFrame for the Biot-Savart law calculation
    mask = df_parameters["name"] == "conductor_all_length"
    con_length = df_parameters.loc[mask, "evaluated_value"].iloc[0]

    mask = df_parameters["name"] == "conductor_all_width"
    con_width = df_parameters.loc[mask, "evaluated_value"].iloc[0]
    num_w = 2 * max(round(con_width / min_distance), 1)

    mask = df_parameters["name"] == "conductor_all_height"
    con_height = df_parameters.loc[mask, "evaluated_value"].iloc[0]
    num_h = 2 * max(round(con_height / min_distance), 1)

    mask = df_parameters["name"] == "I_conductor_terminal"
    I_con = df_parameters.loc[mask, "evaluated_value"].iloc[0]

    
    # get y values
    y = [np.nan] * len(x)
    for i, x_val in enumerate(x):
        # Determine the position
        z_pos = z_pos if z_pos is not None else 0.0
        pos = (0, 0, 0)
        if xparam == "x":
            pos = (x_val, 0, z_pos)
        if xparam == "y":
            pos = (0, x_val, z_pos)
        if xparam == "z":
            pos = (0, 0, x_val)

        # Calculate the magnetic field using the Biot-Savart law for a rectangular conductor
        B = tfm.biot_savart_rectangular_conductor(
            pos = pos,    
            R_con = [(0.5*con_length, 0, 0.5*con_height), (-0.5*con_length, 0, 0.5*con_height)],  
            I = I_con,
            width_vec = (0, con_width, 0),
            height_vec = (0, 0, con_height),
            num_w = num_w,
            num_h = num_h,           
        )     
        y[i]=B[Bidx]

    theory_name = fr"Biot-Savart (${num_w} \times {num_h}$)"
    if not xparam == "z" and mark_z:
        z_value_str = r"$z =" f" {z_pos}" + r" [\mathrm{m}]$"
        z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=False)
        label = z_value_str + ", " + theory_name
    else:
        label = theory_name

    ax.plot(x, y, label=label, color=color)#, s=apf.MARKERSIZE)

    return fig, ax

##############################################################################

def add_temperature_theory_01_00(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        df_parameters: pd.DataFrame,
        xparam: str,
        z_pos: float = 0.0,
        colorA: str = "tab:blue",
        colorB: str = "tab:red",
        mark_z: bool = False,
        ):
    """
    Add a theoretical temperature curve to the provided matplotlib axis.
    """

    df_layer = pd.DataFrame({
    "name": ["conductor", "insulator", "epilayer", "substrate"],    # layer names (order matters for the temperature calculation)
    "thermal_conductivity": [429, 1.38, 450, 450],               # thermal conductivity in W/(m·K)
    "electrical_conductivity": [61.6e6, np.nan, np.nan, np.nan],    # electrical conductivity in S/m
    })
    
    if xparam not in ["x", "y", "z"]:
        print(f"Warning: xparam '{xparam}' not in ['x', 'y', 'z']. Skipping theory curve generation.")
        return fig, ax 

    xlimits = get_limits(ax, xparam, 0.0)
    
    # Generate x values for the theory curve
    x = np.linspace(xlimits[0], xlimits[1], THEORY_PLOT_POINTS)

    # extract the necessary parameters from the DataFrame
    width_mask = df_parameters["name"].str.contains("_width", na=False)
    height_mask = df_parameters["name"].str.contains("_height", na=False)

    for layer in df_layer["name"]:
        layer_mask = df_parameters["name"].str.contains(layer, na=False) & ~df_parameters["name"].str.contains("distortion", na=False)

        width_series = df_parameters.loc[layer_mask & width_mask, "evaluated_value"]
        if not width_series.empty:
            df_layer.loc[df_layer["name"] == layer, "width"] = width_series.iloc[0]
        else:
            raise ValueError(f"Width for layer '{layer}' not found in df_parameters.")

        height_series = df_parameters.loc[layer_mask & height_mask, "evaluated_value"]
        if not height_series.empty:
            df_layer.loc[df_layer["name"] == layer, "height"] = height_series.iloc[0]
        else:
            raise ValueError(f"Height for layer '{layer}' not found in df_parameters.")

    # get the current through the conductor layer
    mask = df_parameters["name"] == "I_conductor_terminal"
    I_con = df_parameters.loc[mask, "evaluated_value"].iloc[0]

    # get power per length for the conductor layer
    power_per_length = tfm.power_per_length(
            I = I_con,
            sigma = df_layer[df_layer["name"] == "conductor"]["electrical_conductivity"].iloc[0],
            h = df_layer[df_layer["name"] == "conductor"]["height"].iloc[0],
            w = df_layer[df_layer["name"] == "conductor"]["width"].iloc[0],
        )

    # set start values for z and y positions and isothermal temperature
    T_iso = T_0  # Isothermal temperature in Kelvin
    z_plot: float = z_pos 
    y_plot: float = 0.0

    if True:
        # get y values
        y = [np.nan] * len(x)
        for i, x_val in enumerate(x):
            if xparam == "z":
                    z_plot = x_val
                    y_plot = 0.0
            if xparam == "y":
                    z_plot = z_pos 
                    y_plot = x_val
            
            T = tfm.temp_est_line_source_approximation(
                    df = df_layer,
                    power_per_length = power_per_length,
                    z = z_plot,
                    y = y_plot,
                    T_iso = T_iso,
                )
            
            y[i] = T - T_iso # store the relative temperature (T - T_iso) in the y array
            
        # plot
        theory_name = "line source approx."
        if not xparam == "z":
            z_value_str = r"$z =" + f" {z_pos}" + r" [\mathrm{m}]$"
            z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=False)
            label = z_value_str + ", " + theory_name
        else:
            label = theory_name

        ax.plot(x, y, label=label, color=colorB)#, s=apf.MARKERSIZE)

    if True:
        # get y values
        y = [np.nan] * len(x)
        for i, x_val in enumerate(x):
            if xparam == "z":
                    z_plot = x_val
                    y_plot = 0.0
            if xparam == "y":
                    z_plot = z_pos 
                    y_plot = x_val
            
            T = tfm.temp_est_split_framework_merged(
                    df = df_layer,
                    power_per_length = power_per_length,
                    z = z_plot,
                    y = y_plot,
                    T_iso = T_iso,
                )
            
            y[i] = T - T_iso # store the relative temperature (T - T_iso) in the y array
            
        # plot
        theory_name = "extended source approx."
        if not xparam == "z" and mark_z:
            z_value_str = r"$z =" + f" {z_pos}" + r" [\mathrm{m}]$"
            z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=False)
            label = z_value_str + ", " + theory_name
        else:
            label = theory_name

        ax.plot(x, y, label=label, color=colorA)#, s=apf.MARKERSIZE)


    return fig, ax

##############################################################################
##############################################################################

def add_magnetic_theory_02_00(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        df_parameters: pd.DataFrame,
        xparam: str,
        Bidx: int,
        min_distance:float = 1.5e-6,
        z_pos: float | None = None,
        withsupply: bool = False,
        color: str = "tab:red",
        mark_z: bool = True,
        )-> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Add a theoretical magnetic field curve to the provided matplotlib axis based on the Biot-Savart law.
    
    Args:
        fig (matplotlib.figure.Figure):             The matplotlib figure object.
        ax (matplotlib.axes.Axes):                  The matplotlib axes object where the theory curve will be added.
        df_parameters (pd.DataFrame):               DataFrame containing the necessary parameters for the Biot-Savart law calculation.
        Bidx (int):                                 Index indicating which component of the magnetic field to plot (0 for Bx, 1 for By, 2 for Bz).
        min_distance (float, optional):             Minimum distance for the x-limits. Defaults to 3e-6.
        z_pos (float | None, optional):             Optional z-position for the theory curve. If None, the z-position will be determined based on the xparam. Defaults to None.
        
    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The updated figure and axes objects with the theory curve added.
    """
    if xparam not in ["x", "y", "z"]:
        print(f"Warning: xparam '{xparam}' not in ['x', 'y', 'z']. Skipping theory curve generation.")
        return fig, ax 

    xlimits = get_limits(ax, xparam, min_distance)

    if xparam == "x": 
        xlimits = [0.9*xlimits[0], 0.9*xlimits[1]] 
    
    # Generate x values for the theory curve
    x = np.linspace(xlimits[0], xlimits[1], THEORY_PLOT_POINTS)

    # extract the necessary parameters from the DataFrame for the Biot-Savart law calculation
    # conductor A
    mask = df_parameters["name"] == "conductor_A_length"
    conA_length = df_parameters.loc[mask, "evaluated_value"].iloc[0]

    mask = df_parameters["name"] == "conductor_A_width"
    conA_width = df_parameters.loc[mask, "evaluated_value"].iloc[0]
    numA_w = 2 * max(round(conA_width / min_distance), 1)

    mask = df_parameters["name"] == "conductor_A_height"
    conA_height = df_parameters.loc[mask, "evaluated_value"].iloc[0]
    numA_h = 2 * max(round(conA_height / min_distance), 1)

    # conductor B
    mask = df_parameters["name"] == "conductor_B_length"
    conB_length = df_parameters.loc[mask, "evaluated_value"].iloc[0]

    mask = df_parameters["name"] == "conductor_B_width"
    conB_width = df_parameters.loc[mask, "evaluated_value"].iloc[0]
    numB_w = 2 * max(round(conB_width / min_distance), 1)

    mask = df_parameters["name"] == "conductor_B_height"
    conB_height = df_parameters.loc[mask, "evaluated_value"].iloc[0]
    numB_h = 2 * max(round(conB_height / min_distance), 1)

    mask = df_parameters["name"] == "epilayer_height"
    conB_height_offset = -1 * df_parameters.loc[mask, "evaluated_value"].iloc[0]

    conB_width_offset = 0.0
    mask = df_parameters["name"] == "epilayer_width"
    conB_width_offset += 0.5 * df_parameters.loc[mask, "evaluated_value"].iloc[0]
    mask = df_parameters["name"] == "conductor_epilayer_distance"
    conB_width_offset += df_parameters.loc[mask, "evaluated_value"].iloc[0]
    conB_width_offset += 0.5 * conB_width

    # currents
    mask = df_parameters["name"] == "I_conductor_A_terminal"
    I_conA = df_parameters.loc[mask, "evaluated_value"].iloc[0]

    mask = df_parameters["name"] == "I_conductor_B_minus_terminal"
    I_conBm = df_parameters.loc[mask, "evaluated_value"].iloc[0]

    mask = df_parameters["name"] == "I_conductor_B_plus_terminal"
    I_conBp = df_parameters.loc[mask, "evaluated_value"].iloc[0]

    if withsupply:
        # calculate conductor arm points
        mask = df_parameters["name"] == "arm_angle_B"
        conB_angle = df_parameters.loc[mask, "evaluated_value"].iloc[0]

        mask = df_parameters["name"] == "substrate_width"
        substrate_width = df_parameters.loc[mask, "evaluated_value"].iloc[0]

        mask = df_parameters["name"] == "substrate_length"
        substrate_length = df_parameters.loc[mask, "evaluated_value"].iloc[0]

        ## get the overshoot of the arm in x and y directions
        xovershoot_arm = np.abs(np.cos(conB_angle) +0.5*conB_length - 0.5*substrate_length)
        yovershoot_arm = np.abs(np.sin(conB_angle) +conB_width_offset - 0.5*substrate_width)

        ## calculate the arm length by limiting the arm to the substrate dimensions
        r_0 = 0
        r_1 = 1
        if xovershoot_arm > yovershoot_arm:
            xr = np.abs(0.5*substrate_length - 0.5*conB_length)
            r_1 = xr / np.cos(conB_angle)
        else:
            yr = np.abs(0.5*substrate_width - conB_width_offset)
            r_1 = yr / np.sin(conB_angle)
        
        # calculate the points of the conductor arms in 3D space
        p0_pp = ((r_0 * np.cos(conB_angle)+0.5*conB_length), (r_0*np.sin(conB_angle)+conB_width_offset), (conB_height_offset+0.5*conB_height))
        p1_pp = ((r_1 * np.cos(conB_angle)+0.5*conB_length), (r_1*np.sin(conB_angle)+conB_width_offset), (conB_height_offset+0.5*conB_height))

        p0_mp = (-(r_0 * np.cos(conB_angle)+0.5*conB_length), (r_0*np.sin(conB_angle)+conB_width_offset), (conB_height_offset+0.5*conB_height))
        p1_mp = (-(r_1 * np.cos(conB_angle)+0.5*conB_length), (r_1*np.sin(conB_angle)+conB_width_offset), (conB_height_offset+0.5*conB_height))

        p0_mm = (-(r_0 * np.cos(conB_angle)+0.5*conB_length), -(r_0*np.sin(conB_angle)+conB_width_offset), (conB_height_offset+0.5*conB_height))
        p1_mm = (-(r_1 * np.cos(conB_angle)+0.5*conB_length), -(r_1*np.sin(conB_angle)+conB_width_offset), (conB_height_offset+0.5*conB_height))

        p0_pm = ((r_0 * np.cos(conB_angle)+0.5*conB_length), -(r_0*np.sin(conB_angle)+conB_width_offset), (conB_height_offset+0.5*conB_height))
        p1_pm = ((r_1 * np.cos(conB_angle)+0.5*conB_length), -(r_1*np.sin(conB_angle)+conB_width_offset), (conB_height_offset+0.5*conB_height))

    # calculate the points of the main conductors in 3D space
    conA_p0 = (0.5*conA_length, 0, 0.5*conA_height)
    conA_p1 = (-0.5*conA_length, 0, 0.5*conA_height)

    conBp_p0 = (0.5*conB_length, conB_width_offset, conB_height_offset+0.5*conB_height)
    conBp_p1 = (-0.5*conB_length, conB_width_offset, conB_height_offset+0.5*conB_height)

    conBm_p0 = (0.5*conB_length, -1*conB_width_offset, conB_height_offset+0.5*conB_height)
    conBm_p1 = (-0.5*conB_length, -1*conB_width_offset, conB_height_offset+0.5*conB_height)


    # get y values
    y = [np.nan] * len(x)
    for i, x_val in enumerate(x):
        # Determine the position
        z_pos = z_pos if z_pos is not None else 0.0
        pos = (0, 0, 0)
        if xparam == "x":
            pos = (x_val, 0, z_pos)
        if xparam == "y":
            pos = (0, x_val, z_pos)
        if xparam == "z":
            pos = (0, 0, x_val)

        # Calculate the magnetic field using the Biot-Savart law
        B = np.zeros(3) # initialize the magnetic field vector using numpy array for easy addition

        # main conductors
        B_A = tfm.biot_savart_rectangular_conductor(
            pos = pos,    
            R_con = [conA_p0, conA_p1],
            I = I_conA,
            width_vec = (0, conA_width, 0),
            height_vec = (0, 0, conA_height),
            num_w = numA_w,
            num_h = numA_h,           
        )
        B += np.array(B_A)

        B_BP = tfm.biot_savart_rectangular_conductor(
                    pos = pos,    
                    R_con = [conBp_p0, conBp_p1],
                    I = I_conBp,
                    width_vec = (0, conB_width, 0),
                    height_vec = (0, 0, conB_height),
                    num_w = numB_w,
                    num_h = numB_h,           
                )
        B += np.array(B_BP)
        
        B_BM = tfm.biot_savart_rectangular_conductor(
                    pos = pos,    
                    R_con = [conBm_p0, conBm_p1],
                    I = I_conBm,
                    width_vec = (0, conB_width, 0),
                    height_vec = (0, 0, conB_height),
                    num_w = numB_w,
                    num_h = numB_h,           
                )
        B += np.array(B_BM)

        if withsupply:
            # arm conductors
            B_PP_arm = tfm.biot_savart_rectangular_conductor(
                        pos = pos,
                        R_con = [p0_pp, p1_pp],
                        I = I_conBp,
                        width_vec = (0, conB_width, 0),
                        height_vec = (0, 0, conB_height),
                        num_w = numB_w,
                        num_h = numB_h,
                    )
            B += np.array(B_PP_arm)

            B_MP_arm = tfm.biot_savart_rectangular_conductor(
                        pos = pos,
                        R_con = [p0_mp, p1_mp],
                        I = I_conBp,
                        width_vec = (0, conB_width, 0),
                        height_vec = (0, 0, conB_height),
                        num_w = numB_w,
                        num_h = numB_h,
                    )
            B += np.array(B_MP_arm)

            B_MM_arm = tfm.biot_savart_rectangular_conductor(
                        pos = pos,
                        R_con = [p0_mm, p1_mm],
                        I = I_conBm,
                        width_vec = (0, conB_width, 0),
                        height_vec = (0, 0, conB_height),
                        num_w = numB_w,
                        num_h = numB_h,
                    )
            B += np.array(B_MM_arm)

            B_PM_arm = tfm.biot_savart_rectangular_conductor(
                        pos = pos,
                        R_con = [p0_pm, p1_pm],
                        I = I_conBm,
                        width_vec = (0, conB_width, 0),
                        height_vec = (0, 0, conB_height),
                        num_w = numB_w,
                        num_h = numB_h,
                    )
            B += np.array(B_PM_arm)

        y[i]=B[Bidx]

    
    theory_name = fr"Biot-Savart (${numB_w} \times {numB_h}$)"
    if not xparam == "z" and mark_z:
        z_value_str = r"$z =" + f" {z_pos}" + r" [\mathrm{m}]$"
        z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=False)
        label = z_value_str + ", " + theory_name
    else:
        label = theory_name

    ax.plot(x, y, label=label, color=color)#, s=apf.MARKERSIZE)

    return fig, ax

##############################################################################
##############################################################################
# workflow helpers: plot types
##############################################################################
##############################################################################

def import_csv_to_df(input_folder: Path, modelfolder: str, modelname: str, ending: str, data_export: str = "Data Export"):
    path = input_folder / modelfolder / modelname / data_export / f"{modelname}{ending}"
    df = pd.read_csv(path)

    if ending == "-parameters.csv":
        df = df.drop(labels = "description", axis = 1)
    return df


def import_COMSOLTXT_to_df(input_folder: Path, modelfolder: str, modelname: str, ending: str, data_export: str = "Data Export", T_0: float = T_0):
    path = input_folder / modelfolder / modelname / data_export / f"{modelname}{ending}"
    header, df = cdi.read_comsol_export(str(path))

    # use relative temperatures
    if "T (K)" in df.columns:
        df["T (K)"] = df["T (K)"].apply(lambda T: T - T_0)

    return header, df

##############################################################################

def import_csv_to_df_sweep(iteration: int, input_folder: Path, modelfolder: str, modelname: str, sweep_dict: dict[str, str], ending: str, sweep_export: str = "Sweep Export", left_out_lines: float | None = None):
    if modelname not in sweep_dict:
        raise ValueError(f"Model name '{modelname}' not found in sweep_dict.")
    
    if left_out_lines is None or not "left_out_lines" in sweep_dict[modelname]:
        folder = input_folder / modelfolder / modelname / sweep_dict[modelname] / sweep_export 
    else:
        folder = input_folder / modelfolder / modelname / sweep_dict[modelname] / ("left_out_lines - " + str(left_out_lines)) / sweep_export

    path = folder/ f"{modelname}-iteration_{iteration}{ending}"
    df = pd.read_csv(path)
    
    if ending == "-parameters.csv":
        df = df.drop(labels = "description", axis = 1)
    return df

def import_COMSOLTXT_to_df_sweep(iteration: int, input_folder: Path, modelfolder: str, modelname: str, sweep_dict: dict[str, str], ending: str, sweep_export: str = "Sweep Export", T_0: float = T_0,  left_out_lines: float | None = None):
    if modelname not in sweep_dict:
            raise ValueError(f"Model name '{modelname}' not found in sweep_dict.")
        
    if left_out_lines is None or not "left_out_lines" in sweep_dict[modelname]:
        folder = input_folder / modelfolder / modelname / sweep_dict[modelname] / sweep_export 
    else:
        folder = input_folder / modelfolder / modelname / sweep_dict[modelname] / ("left_out_lines - " + str(left_out_lines)) / sweep_export

    path = folder/ f"{modelname}-iteration_{iteration}{ending}"
    header, df = cdi.read_comsol_export(str(path))

    # use relative temperatures
    if "T (K)" in df.columns:
        df["T (K)"] = df["T (K)"].apply(lambda T: T - T_0)

    return header, df

##############################################################################
##############################################################################

def display_params(params: list[str], df_parameters: pd.DataFrame, translation_dict: dict[str, str], df_terminals: pd.DataFrame | None = None) -> str:
    label = ""
    for param in params:
        if param != "V_Grid":
            mask = df_parameters["name"] == param
            if not mask.any():
                raise ValueError(f"Parameter '{param}' not found in df_parameters.")
            value = df_parameters.loc[mask, "evaluated_value"].iloc[0]
            name = df_parameters.loc[mask, "name"].iloc[0]
        else:
            if df_terminals is None:
                raise ValueError("df_terminals must be provided when displaying 'V_Grid'.")
            
            mask = df_terminals["Terminal"].str.contains("G", na=False)
            df_grid = df_terminals[mask]

            max_voltage = df_grid["Voltage (V)"].max()
            min_voltage = df_grid["Voltage (V)"].min()

            value = abs(min_voltage) + abs(max_voltage)
            name = "V_Grid"

        value_str = name + r" = " + f"{value}"
        value_str = apf.translate_and_prefix_label(value_str, translation_dict, bm=False)
        label += value_str + ", "

    return label.strip(", ")

##############################################################################
##############################################################################

def reduce_points_by_density(
    df: pd.DataFrame,
    sort_column: str,
    percentage: float,
    rtol: float = 1e-7
    ) -> pd.DataFrame:
    """
    Reduces the number of data points in a DataFrame by removing points from the densest regions based on a specified percentage.
    Aditionally, it merges near-identical floating-point values based on a relative tolerance (rtol) to handle duplicates.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data points.
        sort_column (str): The column name to sort and calculate density by.
        percentage (float): The percentage of points to remove from the densest regions (0 to 100).
        rtol (float): Relative tolerance for merging near-identical floating-point values. Defaults to 1e-7.

    Returns:
        pd.DataFrame: A new DataFrame with reduced data points, sorted by the specified column.
    """
    if not (0 <= percentage < 100):
        raise ValueError("Percentage must be between 0 and 100.")
    if sort_column not in df.columns:
        raise ValueError(f"Column '{sort_column}' not found in DataFrame.")

    # 1. Sort initially to find adjacent duplicate points across any scale
    df_sorted = df.sort_values(by=sort_column).copy()

    # 2. Group near-identical floating point values using relative tolerance
    # Extract native numpy array to satisfy Pylance type checking for np.diff
    values: np.ndarray = df_sorted[sort_column].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        # Rel. diff: |x1 - x0| / max(|x0|, |x1|)
        abs_diff = np.abs(np.diff(values))
        max_vals = np.maximum(np.abs(values[:-1]), np.abs(values[1:]))
        # Handle exact zeros to avoid 0/0 division
        max_vals[max_vals == 0] = 1.0
        rel_diff = abs_diff / max_vals

    # Create group IDs: increment group index whenever relative difference > rtol
    group_boundary = rel_diff > rtol
    # Prepend 0 to maintain array length alignment after diff
    group_ids = np.concatenate(
        (np.array([0], dtype=int), np.cumsum(group_boundary))
    )

    # Aggregate duplicates using the mean
    df_aggregated = df_sorted.groupby(group_ids).mean()

    if percentage == 0:
        return df_aggregated.reset_index(drop=True)

    # 3. Calculate local density on the aggregated data (using distance)
    # Replaced deprecated fillna(method="bfill") with pandas 2.0 compliant .bfill()
    distances = df_aggregated[sort_column].diff().bfill()

    # 4. Determine threshold to drop the densest points
    num_to_drop = int(len(df_aggregated) * (percentage / 100.0))

    if num_to_drop == 0:
        return df_aggregated.reset_index(drop=True)

    threshold_distance = np.partition(distances.to_numpy(), num_to_drop)[
        num_to_drop
    ]

    # 5. Filter out the densest points
    df_reduced = df_aggregated[distances >= threshold_distance]

    # Handle edge cases where identical distances exist
    target_size = len(df_aggregated) - num_to_drop
    if len(df_reduced) > target_size:
        df_reduced = df_reduced.head(target_size)

    return df_reduced.reset_index(drop=True)

##############################################################################

def reduce_points_randomly(df: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """
    Randomly reduces the number of data points in a DataFrame by removing a specified percentage of points.
    Args:
        df (pd.DataFrame): The input DataFrame containing the data points.
        percentage (float): The percentage of points to remove randomly (0 to 100).
    
    Returns:
        pd.DataFrame: A new DataFrame with randomly reduced data points.
    """
    if not (0 <= percentage < 100):
        raise ValueError("Percentage must be between 0 and 100.")

    keep_fraction = 1.0 - (percentage / 100.0)
    return df.sample(frac=keep_fraction, random_state=42).reset_index(drop=True)

##############################################################################

def reduce_points(
    df: pd.DataFrame,
    random_pct: float = 0.0,
    density_pct: float = 0.0,
    sort_column: str | None = None,
    rtol: float = 1e-7,
) -> pd.DataFrame:
    """
    Reduces the number of data points in a DataFrame by applying density-based reduction followed by random reduction.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data points.
        random_pct (float): The percentage of points to remove randomly (0 to 100).
        density_pct (float): The percentage of points to remove from densest regions (0 to 100).
        sort_column (str, optional): The column name to sort and calculate density by.
        rtol (float): Relative tolerance to merge floating-point duplicates.

    Returns:
        pd.DataFrame: A new DataFrame with reduced data points.
    """
    current_df = df.copy()

    # 1. Apply density reduction first (includes scale-independent duplicate merging)
    if density_pct > 0 or sort_column is not None:
        if sort_column is None:
            raise ValueError(
                "A 'sort_column' must be provided when density_pct > 0."
            )
        current_df = reduce_points_by_density(
            current_df, sort_column, density_pct, rtol
        )

    # 2. Apply random reduction on the remaining data
    if random_pct > 0:
        current_df = reduce_points_randomly(current_df, random_pct)

    return current_df

##############################################################################
##############################################################################

def iteration_setup(
    translation_dict: dict[str, str],

    # paths
    input_folder: Path, 
    modelfolder: str,
    modelnames: list[str], 
    ending: str = "-homogeneity_exported_data.txt", 
    data_export: str = "Data Export",
    
    sweep_dict: dict[str, str] | None = None,
    sweep_parameters_dict: dict[str, list[float]] | None = None,

    only_iterations: list[int] | None = None,
    ):
        
    group_length = len(modelnames)

    if sweep_dict is None:
        # model_labels: list[str] = [""] * len(modelnames)
        # for i, model in enumerate(modelnames):
        #     model_labels[i] = model.split("-")[-1]
        #     if translation_dict is not None and model_labels[i] in translation_dict:
        #          model_labels[i] = translation_dict[model_labels[i]]

        iterations = np.arange(group_length) if group_length > 0 else []
    else:
        if group_length != 1:
            raise ValueError("For sweep plotting, only one modelname should be provided.")
            
        modelname = modelnames[0] 

        if "left_out_lines" in sweep_dict[modelname]:
            folder = input_folder / modelfolder / modelname / sweep_dict[modelname] / ("left_out_lines - 0") / data_export
        else:
            folder = input_folder / modelfolder / modelname / sweep_dict[modelname] / data_export

        files = [f for f in folder.glob(f"*{ending}")]
        group_length = len(files)

        iterations = np.arange(group_length) if group_length > 0 else []
        iterations = [int(i) for i in iterations]

        if only_iterations is not None:
            iterations = [i for i in iterations if i in only_iterations]
            if len(iterations) == 0:
                raise ValueError("No valid iterations found after filtering with 'only_iterations'.")
            group_length = len(iterations)

        model_label = modelname.split("-")[1] if "-" in modelname else modelname
        if translation_dict is not None and model_label in translation_dict:
            model_label = translation_dict[model_label]

        modelnames = [modelname] * group_length
        # model_labels = [model_label] * group_length

    iterations = [int(i) for i in iterations]  # Ensure iterations are integers
    left_out_lines_list = sweep_parameters_dict["left_out_lines"] if sweep_parameters_dict is not None and "left_out_lines" in sweep_parameters_dict else [None] 

    return modelnames, iterations,  left_out_lines_list



##############################################################################
##############################################################################

def get_df_lists(
    iterations: list[int],

    input_folder: Path, 
    modelfolder: str,
    modelnames: list[str], 
    ending: str = "-depth_exported_data.txt", 
    data_export: str = "Data Export",
    sweep_dict: dict[str, str] | None = None,
    
    left_out_lines_list: list[float] | list[None] = [None],

    params_to_mean: list[str] | None = None,
    params_to_unique: list[str] | None = None,
    ):
    # import dataframes for each model in the group
    list_df: list[pd.DataFrame] = []

    meaned_values: dict[str, set] = {}
    if params_to_mean is not None:
        for param in params_to_mean:
            meaned_values[param] = set()

    unique_values: dict[str, set] = {}
    if params_to_unique is not None:
        for param in params_to_unique:
            unique_values[param] = set()
    
    for left_out_lines in left_out_lines_list:
        for midx, iteration in enumerate(iterations):
            # get all depth dataframes
            if sweep_dict is None:
                depth_header_data, df = import_COMSOLTXT_to_df(input_folder, modelfolder, modelnames[midx], ending, data_export)
            else:
                depth_header_data, df = import_COMSOLTXT_to_df_sweep(iteration, input_folder, modelfolder, modelnames[midx], sweep_dict, ending, data_export, left_out_lines=left_out_lines)

            if df.empty:
                print(f"Warning: DataFrame for model '{modelnames[midx]}' is empty. Skipping this model.")
                continue

            if params_to_mean is not None:
                for param in params_to_mean:
                    df[param] = cdi.round_to_6_sig_digits(df[param])
                    meaned_values[param].update(df[param].dropna())

            if params_to_unique is not None:
                for param in params_to_unique:
                    df[param] = cdi.round_to_6_sig_digits(df[param])
                    unique_values[param].update(df[param].dropna())

            list_df.append(df)

    if len(list_df) == 0:
        raise ValueError("No valid dataframes to plot. Exiting function.")

    meaned_values_float: dict[str, float] = {}
    if params_to_mean is not None:
        for param in params_to_mean:
            meaned_values_float[param] = float(np.mean(list(meaned_values[param])))
            if meaned_values_float[param] < 1e-20:
                meaned_values_float[param] = 0.0

    unique_values_list: dict[str, list[float]] = {}
    if params_to_unique is not None:
        for param in params_to_unique:
            unique_values_list[param] = sorted(list(unique_values[param]))

    return list_df, meaned_values_float, unique_values_list

##############################################################################

def apply_insulator_height(
    df_parameters: pd.DataFrame,
    plot_z: list[float],
    ):
    mask = df_parameters["name"] == "insulator_height"
    insulator_height = float(df_parameters.loc[mask, "evaluated_value"].iloc[0])
    plot_z = [z - insulator_height for z in plot_z]

    return plot_z, insulator_height

##############################################################################

def check_validity_of_z_values(
        plot_z: list[float],
        unique_z_values: list[float]
    ):
    # Round the z values to 6 significant figures to avoid floating point issues when comparing with unique_z_values
    positive_values = np.where(plot_z == 0, 1e-20, np.abs(plot_z))
    exponent = np.floor(np.log10(positive_values)) # get magnitude 
    factor = 10 ** (5 - exponent) # shift factor
    plot_z = (np.round(np.array(plot_z) * factor) / factor).tolist() # round shifted values and shift back
    plot_z = (np.where(plot_z == 0.0, 0.0, plot_z)).tolist() 

    for z_value in plot_z:
        if z_value not in unique_z_values:
            print(f"{z_value} is not a valid z value.")
    return plot_z

##############################################################################

def get_labels_and_title(
    translation_dict: dict[str, str],

    label_start: str,
    title_start: str, # = groupname

    title_parameters: list[str] | None,
    title_param_dict: dict[str, float] | None, # added to title, but with provided values, not from df

    label_parameters: list[str] | None,
    label_param_dict: dict[str, list[float]] | None, # added to label, but with provided values, not from df

    # paths
    input_folder: Path, 
    modelfolder: str,
    modelname: str, # modelnames[midx],
    data_export: str,
    sweep_dict: dict[str, str] | None,

    iteration: int,
    left_out_lines: float | None,
    midx: int | None, # = iteration

    show_modelname_in_title: bool = False,
    show_modelname_in_label: bool = True,
    ):
    if midx is None:
        midx = iteration

    label = label_start
    title = title_start + " ("

    modellabel = modelname.split("-")[1] if "-" in modelname else modelname
    modellabel = translation_dict[modellabel] if modellabel in translation_dict else modellabel

    if show_modelname_in_title:
        if title not in [""] and not title.endswith(", ") and not title.endswith("("):
            title += ", "
        title += modellabel + ", "

    if title_param_dict is not None:
        dict_title = ""
        for param in title_param_dict.keys():
            value = title_param_dict[param]
            name = param

            value_str = name + r" = " + f"{value}"
            value_str = apf.translate_and_prefix_label(value_str, translation_dict, bm=False)
            value_str = apf.clean_label_text(value_str)
            dict_title += value_str + ", "
        title += dict_title.strip(", ")


    if label_parameters is not None or title_parameters is not None:
        all_parameters = []
        all_parameters.extend(label_parameters if label_parameters is not None else [])
        all_parameters.extend(title_parameters if title_parameters is not None else [])

        if sweep_dict is None:
            df_parameters = import_csv_to_df(input_folder, modelfolder, modelname, "-parameters.csv", data_export)
            df_terminals = import_csv_to_df(input_folder, modelfolder, modelname, "-terminals.csv", data_export) if "V_Grid" in all_parameters else None
        else:
            df_parameters = import_csv_to_df_sweep(iteration, input_folder, modelfolder, modelname, sweep_dict, "-parameters.csv", data_export, left_out_lines)
            df_terminals = import_csv_to_df_sweep(iteration, input_folder, modelfolder, modelname, sweep_dict, "-terminals.csv", data_export, left_out_lines) if "V_Grid" in all_parameters else None

        if title_parameters is not None:
            title += ", " + display_params(title_parameters, df_parameters, translation_dict, df_terminals=df_terminals) 
        if label_parameters is not None:
            label += display_params(label_parameters, df_parameters, translation_dict, df_terminals=df_terminals)

    if label_param_dict is not None:
        dict_label = ""
        for param in label_param_dict.keys():
            value = label_param_dict[param][midx]
            name = param

            value_str = name + r" = " + f"{value}"
            value_str = apf.translate_and_prefix_label(value_str, translation_dict, bm=False)
            value_str = apf.clean_label_text(value_str)
            dict_label += value_str + ", "
        label += ", " + dict_label.strip(", ")

    if show_modelname_in_label:
        if label not in [""] and not label.endswith(", "):
            label += ", "
        label += modellabel

    label = label.strip(", ")
    title = title.strip(", ")
    title += ")"
    return label, title


##############################################################################

def get_color(lidx: int, num_lines: int, zidx: int, num_z: int, midx: int, num_iter: int, max_color_val: float = 0.9):
    cmap = mpl.colormaps['viridis'] # type: ignore
    total_steps = num_lines * num_z * num_iter

    # step number
    current_step = (lidx * num_z * num_iter) + (zidx * num_iter) + midx
                
    # normalize (0 to 1)
    norm_val = current_step / (total_steps - 1 if total_steps > 1 else 1)

    return cmap(norm_val * max_color_val)

##############################################################################
##############################################################################

def zaxis_plot(
    translation_dict: dict[str, str],

    # theory
    magnetic_theory: Callable | None,
    temperature_theory: Callable | None,

    # paths
    output_folder: Path,
    input_folder: Path, 
    modelfolder: str, 
    groupname: str,
    modelnames: list[str], 
    ending: str = "-depth_exported_data.txt", 
    data_export: str = "Data Export",

    # plotting
    fraction: float = 1.0,
    y_axis_params: list[str] = ["mf.Bx (T)", "mf.By (T)", "mf.Bz (T)", "T (K)"],

    title_parameters: list[str] | None = None,
    title_param_dict: dict[str, float] | None = None,

    label_parameters: list[str] | None = None,
    label_param_dict: dict[str, list[float]] | None = None,

    # sweep
    sweep_dict: dict[str, str] | None = None, # to get the path to the sweep folder
    display_left_out_lines : bool= True,

    only_iterations: list[int] | None = None,
    ):
    
    if len(y_axis_params) == 0:
        print("Skipping, since no valid parameters provided.")
        return

    modelnames, iterations,  left_out_lines_list = iteration_setup(
                                                        translation_dict = translation_dict,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending = ending, 
                                                        data_export = data_export,
                                                            
                                                        sweep_dict = sweep_dict,
                                                        sweep_parameters_dict = label_param_dict,
                                                        
                                                        only_iterations = only_iterations,
                                                    )

    # import dataframes for each model in the group
    list_df, meaned_values_float, unique_values_list = get_df_lists(
                                                        iterations = iterations,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending= ending, 
                                                        data_export= data_export,
                                                        sweep_dict = sweep_dict,
                                                            
                                                        left_out_lines_list = left_out_lines_list,
                                                        
                                                        params_to_mean = ["x", "y"],
                                                        params_to_unique = None,
                                                        )
    # plotting
    xparam = "z"
    for yparam in y_axis_params:
        if xparam not in list_df[0].columns:
            print(f"Skipping '{xparam}'.")
            continue
        if yparam not in list_df[0].columns:
            print(f"Skipping '{yparam}'.")
            continue

        # create figure and axis for plotting
        fig, ax = apf.get_fig_ax(fraction_textwidth=fraction)

        # Data
        for lidx, left_out_lines in enumerate(left_out_lines_list):
            for midx, iteration in enumerate(iterations):
                dfidx = lidx*len(iterations) + midx
                df_plot = list_df[dfidx][[xparam, yparam]].copy()

                color = get_color(lidx, len(left_out_lines_list), 0, 1, midx, len(iterations))

                if label_param_dict is not None and not display_left_out_lines:
                    label_param_dict.pop("left_out_lines", None)

                title_param_dict = {} if title_param_dict is None else title_param_dict
                label, title = get_labels_and_title(
                                translation_dict = translation_dict,

                                label_start  = "",
                                title_start = groupname,

                                title_parameters = title_parameters,
                                title_param_dict = meaned_values_float | title_param_dict,

                                label_parameters = label_parameters,
                                label_param_dict = label_param_dict,

                                # paths
                                input_folder = input_folder, 
                                modelfolder = modelfolder,
                                modelname = modelnames[midx],
                                data_export = data_export,
                                sweep_dict = sweep_dict,

                                iteration = iteration,
                                left_out_lines = left_out_lines,
                                midx = midx,
                                    
                                show_modelname_in_title = False,
                                show_modelname_in_label = len(set(modelnames)) > 1,
                            )

                fig, ax = apf.standard_scatter_plot_df(
                            df = df_plot,
                            x = xparam,
                            y = yparam,

                            label = label,
                            
                            color = color,
                            translation_dict = translation_dict,

                            fig = fig,
                            ax = ax,

                            title = title,
                            x_label = xparam,
                            y_label = yparam,
                        )

        # Theory
        if magnetic_theory is not None or temperature_theory is not None:
            df_parameters = import_csv_to_df(input_folder, modelfolder, modelnames[0], "-parameters.csv", data_export)
            
        if "mf.B" in yparam and magnetic_theory is not None:
            Bidx = THEORY_FORMULA[yparam]
            fig, ax = magnetic_theory(fig, ax, df_parameters, xparam=xparam, Bidx=Bidx, z_pos = None, color = "tab:red")

        if "T (K)" in yparam and temperature_theory is not None:
            fig, ax = temperature_theory(fig, ax, df_parameters, xparam=xparam)

        fig, ax = apf.plot_background(fig, ax)
        
        fig, ax = apf.dynamic_legend(fig, ax, fraction=fraction)
        
        # remove any text in parentheses and leading/trailing whitespace
        clean_yparam = re.sub(r"\s*\(.*?\)", "", str(yparam)) 

        # create output folder 
        plot_output_folder = output_folder / modelfolder / "z axis"
        makedirs(plot_output_folder, exist_ok=True)
        groupname_clean = groupname.replace(r"\enquote", "").replace("\"", "_").replace("{", "").replace("}", "")
        iterations_str = "--" + "__".join(map(str, only_iterations)) if only_iterations is not None else ""
        apf.save_figure(fig, path = plot_output_folder / f"{groupname_clean}-{clean_yparam}_over_{xparam}{iterations_str}")

##############################################################################

def yaxis_plot(
    translation_dict: dict[str, str],

    # theory
    magnetic_theory: Callable | None,
    temperature_theory: Callable | None,

    # paths
    output_folder: Path,
    input_folder: Path, 
    modelfolder: str, 
    groupname: str,
    modelnames: list[str], 
    ending: str = "-homogeneity_exported_data.txt", 
    data_export: str = "Data Export",

    # plotting
    fraction: float = 1.0,
    y_axis_params: list[str] = ["mf.Bx (T)", "mf.By (T)", "mf.Bz (T)", "T (K)"],

    title_parameters: list[str] | None = None,
    title_param_dict: dict[str, float] | None = None,

    label_parameters: list[str] | None = None,
    label_param_dict: dict[str, list[float]] | None = None,

    plot_z: list[float] = [-3e-06, -9e-06],
    insulator: bool = False,

    width_limit_param: str | None = "conductor_all_width", 
    width_limit: float | None = None,

    # sweep
    sweep_dict: dict[str, str] | None = None,  # to get the path to the sweep folder
    display_left_out_lines : bool= True,

    only_iterations: list[int] | None = None,
    ):

    if len(y_axis_params) == 0 or len(plot_z) == 0:
        print("Skipping, since no valid parameters provided.")
        return

    modelnames, iterations,  left_out_lines_list = iteration_setup(
                                                        translation_dict = translation_dict,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending = ending, 
                                                        data_export = data_export,
                                                            
                                                        sweep_dict = sweep_dict,
                                                        sweep_parameters_dict = label_param_dict,
                                                        
                                                        only_iterations = only_iterations,
                                                    )
    
    # import dataframes for each model in the group
    list_df, meaned_values_float, unique_values_list = get_df_lists(
                                                        iterations = iterations,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending= ending, 
                                                        data_export= data_export,
                                                        sweep_dict = sweep_dict,
                                                            
                                                        left_out_lines_list = left_out_lines_list,
                                                        
                                                        params_to_mean = ["x"],
                                                        params_to_unique = ["z"],
                                                        )
    unique_z_values = unique_values_list.get("z", [])

    if sweep_dict is None:
        df_parameters = import_csv_to_df(input_folder, modelfolder, modelnames[0], "-parameters.csv", data_export)
    else:
        df_parameters = import_csv_to_df_sweep(0, input_folder, modelfolder, modelnames[0], sweep_dict, "-parameters.csv", data_export, left_out_lines_list[0])

    if insulator:
        plot_z, insulator_height = apply_insulator_height(df_parameters = df_parameters, plot_z = plot_z)

    plot_z = check_validity_of_z_values(plot_z, unique_z_values)

    if width_limit is None:
        width_limit = np.inf
        if width_limit_param is not None and width_limit_param in df_parameters["name"].values:
            mask = df_parameters["name"] == width_limit_param
            width_limit = 1.5 / 2 * float(df_parameters.loc[mask, "evaluated_value"].iloc[0])

        if "02_00" in modelfolder:
            conB_width_offset = 0.0
            mask = df_parameters["name"] == "epilayer_width"
            conB_width_offset += 0.5 * df_parameters.loc[mask, "evaluated_value"].iloc[0]
            mask = df_parameters["name"] == "conductor_epilayer_distance"
            conB_width_offset += df_parameters.loc[mask, "evaluated_value"].iloc[0]
            mask = df_parameters["name"] == "conductor_B_width"
            conB_width_offset += 0.5 * df_parameters.loc[mask, "evaluated_value"].iloc[0]

            width_limit = 1.5 * conB_width_offset


    # plotting
    xparam = "y"
    for yparam in y_axis_params:
        if xparam not in list_df[0].columns:
            print(f"Skipping '{xparam}'.")
            continue
        if yparam not in list_df[0].columns:
            print(f"Skipping '{yparam}'.")
            continue

        # create figure and axis for plotting
        fig, ax = apf.get_fig_ax(fraction_textwidth=fraction)
            
        # Theory Colors
        Acolors = ["tab:red", "tab:orange", "tab:brown"]
        Bcolors = ["tab:blue", "tab:cyan", "tab:purple"]

        # Data
        for lidx, left_out_lines in enumerate(left_out_lines_list):
            for zidx, z_value in enumerate(plot_z):
                for midx, iteration in enumerate(iterations):
                    dfidx = lidx*len(iterations) + midx
                    df_plot = list_df[dfidx][[xparam, yparam]].copy()
                    mask = list_df[dfidx]["z"] == z_value
                    df_plot = df_plot[mask]
                    mask = df_plot[xparam].abs() <= width_limit
                    df_plot = df_plot[mask]

                    color = get_color(lidx, len(left_out_lines_list), zidx, len(plot_z), midx, len(iterations))

                    z_title_dict: dict[str, float] = {}
                    if len(plot_z) < 2:
                        z_title_dict["z"] = z_value
                        label_start = ""
                    else:
                        z_value_str = r"$z$ =" + f" {z_value}" + r" $[\mathrm{m}]$"
                        z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=False)
                        label_start = z_value_str

                    if label_param_dict is not None and not display_left_out_lines:
                        label_param_dict.pop("left_out_lines", None)

                    title_param_dict = {} if title_param_dict is None else title_param_dict
                    label, title = get_labels_and_title(
                                    translation_dict = translation_dict,

                                    label_start  = label_start,
                                    title_start = groupname,

                                    title_parameters = title_parameters,
                                    title_param_dict = meaned_values_float | z_title_dict | title_param_dict,

                                    label_parameters = label_parameters,
                                    label_param_dict = label_param_dict,

                                    # paths
                                    input_folder = input_folder, 
                                    modelfolder = modelfolder,
                                    modelname = modelnames[midx],
                                    data_export = data_export,
                                    sweep_dict = sweep_dict,

                                    iteration = iteration,
                                    left_out_lines = left_out_lines,
                                    midx = midx,
                                    
                                    show_modelname_in_title = False,
                                    show_modelname_in_label = len(set(modelnames)) > 1,
                                )

                    fig, ax = apf.standard_scatter_plot_df(
                                df = df_plot,
                                x = xparam,
                                y = yparam,

                                label = label,
                                color = color,
                                translation_dict = translation_dict,

                                fig = fig,
                                ax = ax,

                                title = title,
                                x_label = xparam,
                                y_label = yparam,

                            )

                # Theory            
                if "mf.B" in yparam and magnetic_theory is not None:
                    Bidx = THEORY_FORMULA[yparam]
                    color = Acolors[zidx % len(Acolors)]
                    fig, ax = magnetic_theory(fig, ax, df_parameters, xparam=xparam, Bidx=Bidx, z_pos = z_value, color = color)

                if "T (K)" in yparam and temperature_theory is not None:
                    colorA = Acolors[zidx % len(Acolors)]
                    colorB = Bcolors[zidx % len(Bcolors)]
                    fig, ax = temperature_theory(fig, ax, df_parameters, xparam=xparam, z_pos = z_value, colorA = colorA, colorB = colorB)

        fig, ax = apf.plot_background(fig, ax)
            
        fig, ax = apf.dynamic_legend(fig, ax, fraction=fraction)
        
        # remove any text in parentheses and leading/trailing whitespace
        clean_yparam = re.sub(r"\s*\(.*?\)", "", str(yparam)) 

        # create output folder 
        plot_output_folder = output_folder / modelfolder / "y axis"
        makedirs(plot_output_folder, exist_ok=True)
        groupname_clean = groupname.replace(r"\enquote", "").replace("\"", "_").replace("{", "").replace("}", "")
        iterations_str = "--" + "__".join(map(str, only_iterations)) if only_iterations is not None else ""
        apf.save_figure(fig, path = plot_output_folder / f"{groupname_clean}-{clean_yparam}_over_{xparam}--{"__".join(map(str, plot_z))}{iterations_str}")

##############################################################################

def xaxis_plot(
    translation_dict: dict[str, str],

    # theory
    magnetic_theory: Callable | None,
    temperature_theory: Callable | None,

    # paths
    output_folder: Path,
    input_folder: Path, 
    modelfolder: str, 
    groupname: str,
    modelnames: list[str], 
    ending: str = "-longitudinal_exported_data.txt",
    data_export: str = "Data Export",

    # plotting
    fraction: float = 1.0,
    y_axis_params: list[str] = ["mf.Bx (T)", "mf.By (T)", "mf.Bz (T)", "T (K)"],

    title_parameters: list[str] | None = None,
    title_param_dict: dict[str, float] | None = None,

    label_parameters: list[str] | None = None,
    label_param_dict: dict[str, list[float]] | None = None,

    plot_z: list[float] = [-3e-06, -9e-06],
    insulator: bool = False,

    length_limit_param: str | None = "conductor_all_length", 
    length_limit: float | None = None,

    # sweep
    sweep_dict: dict[str, str] | None = None,  # to get the path to the sweep folder
    display_left_out_lines : bool= True,

    only_iterations: list[int] | None = None,
    ):
    
    if len(y_axis_params) == 0 or len(plot_z) == 0:
        print("Skipping, since no valid parameters provided.")
        return
    
    modelnames, iterations,  left_out_lines_list = iteration_setup(
                                                        translation_dict = translation_dict,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending = ending, 
                                                        data_export = data_export,
                                                            
                                                        sweep_dict = sweep_dict,
                                                        sweep_parameters_dict = label_param_dict,
                                                        
                                                        only_iterations = only_iterations,
                                                    )

    # import dataframes for each model in the group
    list_df, meaned_values_float, unique_values_list = get_df_lists(
                                                        iterations = iterations,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending= ending, 
                                                        data_export= data_export,
                                                        sweep_dict = sweep_dict,
                                                            
                                                        left_out_lines_list = left_out_lines_list,
                                                        
                                                        params_to_mean = ["y"],
                                                        params_to_unique = ["z"],
                                                        )
    unique_z_values = unique_values_list.get("z", [])

    if sweep_dict is None:
        df_parameters = import_csv_to_df(input_folder, modelfolder, modelnames[0], "-parameters.csv", data_export)
    else:
        df_parameters = import_csv_to_df_sweep(0, input_folder, modelfolder, modelnames[0], sweep_dict, "-parameters.csv", data_export, left_out_lines_list[0])

    if insulator:
        plot_z, insulator_height = apply_insulator_height(df_parameters = df_parameters, plot_z = plot_z)
    
    plot_z = check_validity_of_z_values(plot_z, unique_z_values)

    if length_limit is None:
        length_limit = np.inf
        if length_limit_param is not None and length_limit_param in df_parameters["name"].values:
            mask = df_parameters["name"] == length_limit_param
            length_limit = 1.5 / 2 * float(df_parameters.loc[mask, "evaluated_value"].iloc[0])

    # plotting
    xparam = "x"
    for yparam in y_axis_params:
        if xparam not in list_df[0].columns:
            print(f"Skipping '{xparam}'.")
            continue
        if yparam not in list_df[0].columns:
            print(f"Skipping '{yparam}'.")
            continue

        # create figure and axis for plotting
        fig, ax = apf.get_fig_ax(fraction_textwidth=fraction)
            
        # Color
        Acolors = ["tab:red", "tab:orange", "tab:brown"]
        Bcolors = ["tab:blue", "tab:cyan", "tab:purple"]

        # Data
        for lidx, left_out_lines in enumerate(left_out_lines_list):
            for zidx, z_value in enumerate(plot_z):
                for midx, iteration in enumerate(iterations):
                    dfidx = lidx*len(iterations) + midx
                    df_plot = list_df[dfidx][[xparam, yparam]].copy()
                    mask = list_df[dfidx]["z"] == z_value
                    df_plot = df_plot[mask]
                    mask = df_plot[xparam].abs() <= length_limit
                    df_plot = df_plot[mask]

                    color = get_color(lidx, len(left_out_lines_list), zidx, len(plot_z), midx, len(iterations))

                    z_title_dict: dict[str, float] = {}
                    if len(plot_z) < 2:
                        z_title_dict["z"] = z_value
                        label_start = ""
                    else:
                        z_title_dict = {}
                        z_value_str = r"$z$ =" + f" {z_value}" + r" $[\mathrm{m}]$"
                        z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=False)
                        label_start = z_value_str

                    if label_param_dict is not None and not display_left_out_lines:
                        label_param_dict.pop("left_out_lines", None)

                    title_param_dict = {} if title_param_dict is None else title_param_dict
                    label, title = get_labels_and_title(
                                    translation_dict = translation_dict,

                                    label_start  = label_start,
                                    title_start = groupname,

                                    title_parameters = title_parameters,
                                    title_param_dict = meaned_values_float | z_title_dict | title_param_dict,

                                    label_parameters = label_parameters,
                                    label_param_dict = label_param_dict,

                                    # paths
                                    input_folder = input_folder, 
                                    modelfolder = modelfolder,
                                    modelname = modelnames[midx],
                                    data_export = data_export,
                                    sweep_dict = sweep_dict,

                                    iteration = iteration,
                                    left_out_lines = left_out_lines,
                                    midx = midx,

                                    show_modelname_in_title = False,
                                    show_modelname_in_label = len(set(modelnames)) > 1,
                                )

                    fig, ax = apf.standard_scatter_plot_df(
                                df = df_plot,
                                x = xparam,
                                y = yparam,

                                label = label,
                                color = color,
                                translation_dict = translation_dict,

                                fig = fig,
                                ax = ax,

                                title = title,
                                x_label = xparam,
                                y_label = yparam,
                                xstyle = 'prefix',
                                ystyle = 'prefix',
                            )

                # Theory            
                if "mf.B" in yparam and magnetic_theory is not None:
                    Bidx = THEORY_FORMULA[yparam]
                    color = Acolors[zidx % len(Acolors)]
                    fig, ax = magnetic_theory(fig, ax, df_parameters, xparam=xparam, Bidx=Bidx, z_pos = z_value, color = color)

                if "T (K)" in yparam and temperature_theory is not None:
                    colorA = Bcolors[zidx % len(Bcolors)]
                    colorB = Acolors[zidx % len(Acolors)]
                    fig, ax = temperature_theory(fig, ax, df_parameters, xparam=xparam, z_pos = z_value, colorA = colorA, colorB = colorB)

        fig, ax = apf.plot_background(fig, ax)
            
        fig, ax = apf.dynamic_legend(fig, ax, fraction=fraction)

        # remove any text in parentheses and leading/trailing whitespace
        clean_yparam = re.sub(r"\s*\(.*?\)", "", str(yparam)) 

        # create output folder 
        plot_output_folder = output_folder / modelfolder / "x axis"
        makedirs(plot_output_folder, exist_ok=True)
        groupname_clean = groupname.replace(r"\enquote", "").replace("\"", "_").replace("{", "").replace("}", "")
        iterations_str = "--" + "__".join(map(str, only_iterations)) if only_iterations is not None else ""
        apf.save_figure(fig, path = plot_output_folder / f"{groupname_clean}-{clean_yparam}_over_{xparam}--{"__".join(map(str, plot_z))}{iterations_str}")


##############################################################################

def xyplane_plot(
    translation_dict: dict[str, str],

    # paths
    output_folder: Path,
    input_folder: Path, 
    modelfolder: str, 
    groupname: str,
    modelnames: list[str], 
    ending: str = "-xy_exported_data.txt",
    data_export: str = "Data Export",

    # plotting
    fraction: float = 1.0,
    z_axis_params: list[str] = ["mf.Bx (T)", "mf.By (T)", "mf.Bz (T)", "T (K)"],

    title_parameters: list[str] | None = None,
    title_param_dict: dict[str, float] | None = None,

    sweep_parameters_dict: dict[str, list[float]] | None = None,

    length_limit_param: str | None = "conductor_all_length",
    length_limit: float | None = None,

    width_limit_param: str | None = "conductor_all_width",
    width_limit: float | None = None,

    # sweep
    sweep_dict: dict[str, str] | None = None, # to get the path to the sweep folder
    display_left_out_lines : bool= True,

    only_iterations: list[int] | None = None,
    ):
    if len(z_axis_params) == 0:
        print("No z-axis parameters provided for xy-plane plotting.")
        return
    
    modelnames, iterations,  left_out_lines_list = iteration_setup(
                                                        translation_dict = translation_dict,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending = ending, 
                                                        data_export = data_export,
                                                            
                                                        sweep_dict = sweep_dict,
                                                        sweep_parameters_dict = sweep_parameters_dict,
                                                        
                                                        only_iterations = only_iterations,
                                                    )
    # import dataframes for each model in the group
    list_df, meaned_values_float, unique_values_list = get_df_lists(
                                                        iterations = iterations,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending= ending, 
                                                        data_export= data_export,
                                                        sweep_dict = sweep_dict,
                                                            
                                                        left_out_lines_list = left_out_lines_list,
                                                        
                                                        params_to_mean = ["z"],
                                                        params_to_unique = None,
                                                        )

    if sweep_dict is None:
        df_parameters = import_csv_to_df(input_folder, modelfolder, modelnames[0], "-parameters.csv", data_export)
    else:
        df_parameters = import_csv_to_df_sweep(0, input_folder, modelfolder, modelnames[0], sweep_dict, "-parameters.csv", data_export, left_out_lines_list[0])

    if length_limit is None:
        length_limit = np.inf
        if length_limit_param is not None and length_limit_param in df_parameters["name"].values:
            mask = df_parameters["name"] == length_limit_param
            length_limit = 1.5 / 2 * float(df_parameters.loc[mask, "evaluated_value"].iloc[0])

    if width_limit is None:
        width_limit = np.inf
        if width_limit_param is not None and width_limit_param in df_parameters["name"].values:
            mask = df_parameters["name"] == width_limit_param
            width_limit = 1.5 / 2 * float(df_parameters.loc[mask, "evaluated_value"].iloc[0])

        if "02_00" in modelfolder:
            conB_width_offset = 0.0
            mask = df_parameters["name"] == "epilayer_width"
            conB_width_offset += 0.5 * df_parameters.loc[mask, "evaluated_value"].iloc[0]
            mask = df_parameters["name"] == "conductor_epilayer_distance"
            conB_width_offset += df_parameters.loc[mask, "evaluated_value"].iloc[0]
            mask = df_parameters["name"] == "conductor_B_width"
            conB_width_offset += 0.5 * df_parameters.loc[mask, "evaluated_value"].iloc[0]
            
            width_limit = 1.5 * conB_width_offset

    if width_limit is None or length_limit is None:
        raise ValueError("Both width_limit and length_limit must be provided or computable from parameters.")
    
    width_limit = max(width_limit, length_limit) # ensure width limit is at least as large as length limit
    length_limit = max(length_limit, width_limit) # ensure length limit is at least as large as width limit

    # plotting
    xparam = "x"
    yparam = "y"
    for zparam in z_axis_params:
        if xparam not in list_df[0].columns:
            print(f"Skipping '{xparam}'.")
            continue
        if yparam not in list_df[0].columns:
            print(f"Skipping '{yparam}'.")
            continue
        if zparam not in list_df[0].columns:
            print(f"Skipping '{zparam}'.")
            continue

        for lidx, left_out_lines in enumerate(left_out_lines_list):
            for midx, iteration in enumerate(iterations):
                dfidx = lidx*len(iterations) + midx
                df_plot = list_df[dfidx][[xparam, yparam, zparam]].copy()
                df_plot = df_plot.dropna()
                df_plot = df_plot[(df_plot[xparam].abs() <= length_limit) & (df_plot[yparam].abs() <= width_limit)]

                title_param_dict = {} if title_param_dict is None else title_param_dict
                label, title = get_labels_and_title(
                                translation_dict = translation_dict,

                                label_start  = "",
                                title_start = groupname,

                                title_parameters = title_parameters,
                                title_param_dict = meaned_values_float | title_param_dict,

                                label_parameters = None,
                                label_param_dict = None,

                                # paths
                                input_folder = input_folder, 
                                modelfolder = modelfolder,
                                modelname = modelnames[midx],
                                data_export = data_export,
                                sweep_dict = sweep_dict,

                                iteration = iteration,
                                left_out_lines = left_out_lines,
                                midx = midx,

                                show_modelname_in_title = len(set(modelnames)) > 1,
                                show_modelname_in_label = False,
                                )

                fig, ax = apf.standard_scatter_plot_df(
                            df = df_plot,
                            x = xparam,
                            y = yparam,
                            z = zparam,
                            translation_dict = translation_dict,
                            fraction_textwidth = fraction,
                            title = title,
                            x_label = xparam,
                            y_label = yparam,
                            z_label = zparam,
                            xstyle = 'prefix',
                            ystyle = 'prefix',
                            zstyle = 'prefix',
                            markersize = 0.01*apf.MARKERSIZE,
                        )
            
                # remove any text in parentheses and leading/trailing whitespace
                clean_zparam = re.sub(r"\s*\(.*?\)", "", str(zparam)) 

                model_label = modelnames[midx].split("-")[1] if "-" in modelnames[midx] else modelnames[midx]
                model_label = translation_dict[model_label] if translation_dict and model_label in translation_dict else model_label

                # create output folder 
                plot_output_folder = output_folder / modelfolder / "xy plane"
                makedirs(plot_output_folder, exist_ok=True)  
                z_value = meaned_values_float.get("z", None)
                iterations_str = "--" + "__".join(map(str, only_iterations)) if only_iterations is not None else ""
                apf.save_figure(fig, path = plot_output_folder / f"{model_label}-{clean_zparam}_over_{xparam}{yparam}_plane-{z_value}{iterations_str}")

##############################################################################

def angle_error_plot(
    translation_dict: dict[str, str],

    # paths
    output_folder: Path,
    input_folder: Path, 
    modelfolder: str, 
    groupname: str,
    modelnames: list[str], 
    ending: str = "-depth_exported_data.txt", 
    data_export: str = "Sweep Export",

    fraction: float = 1.0,
    title_parameters: list[str] | None = None,
    title_param_dict: dict[str, float] | None = None,

    label_parameters: list[str] | None = None,
    label_param_dict: dict[str, list[float]] | None = None,

    plot_z: list[float]  = [-3e-06, -9e-06],

    sweep_dict: dict[str, str] | None = None,  # to get the path to the sweep folder   

    display_statistics: bool = True,
    plot_rolling_mean: bool = False,
    plot_total_mean: bool = False,
    ):

    if len(modelnames) != 1:
        raise ValueError("For sweep plotting, only one modelname should be provided.")
        
    modelnames, iterations,  left_out_lines_list = iteration_setup(
                                                        translation_dict = translation_dict,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending = ending, 
                                                        data_export = data_export,
                                                            
                                                        sweep_dict = sweep_dict,
                                                        sweep_parameters_dict = label_param_dict,
                                                        
                                                        only_iterations = None,
                                                    )

    # import dataframes for each model in the group
    list_df, meaned_values_float, unique_values_list = get_df_lists(
                                                        iterations = iterations,
                                                        
                                                        input_folder = input_folder, 
                                                        modelfolder = modelfolder,
                                                        modelnames = modelnames, 
                                                        ending= ending, 
                                                        data_export= data_export,
                                                        sweep_dict = sweep_dict,
                                                            
                                                        left_out_lines_list = left_out_lines_list,
                                                        
                                                        params_to_mean = ["x", "y"],
                                                        params_to_unique = ["z"],
                                                        )
    unique_z_values = unique_values_list.get("z", [])
    print(f"Unique z values: {unique_z_values}")

    plot_z = check_validity_of_z_values(plot_z, unique_z_values)


    # plotting
    xparam = "Set angle (°)"
    yparam = "Angle Error (°)"

    # create figure and axis for plotting
    fig, ax = apf.get_fig_ax(fraction_textwidth=fraction)

    # Color
    cmap = mpl.colormaps['viridis'] # type: ignore

    df_statistics = pd.DataFrame(columns=["Left out lines", "z", "Mean Angle Error (°)", "Std Dev (°)"])
    # Data
    if True:
        df_mean = pd.DataFrame()
        for lidx, left_out_lines in enumerate(left_out_lines_list):
            for zidx, z_value in enumerate(plot_z):
                df_plot = pd.DataFrame()
                for midx, iteration in enumerate(iterations):
                    dfidx = lidx*len(iterations) + midx
                    df_temp = list_df[dfidx].copy()

                    mask = df_temp["z"] == z_value
                    Bx = float(df_temp.loc[mask, "mf.Bx (T)"].iloc[0])
                    By = float(df_temp.loc[mask, "mf.By (T)"].iloc[0])

                    is_angle = np.rad2deg(np.arctan2(By,Bx))
                    set_angle =  label_param_dict["Set angle (°)"][midx] if label_param_dict is not None else np.nan
                    angle_error = is_angle - set_angle

                    df_temp.loc[mask, "Is Angle (°)"] = is_angle
                    df_temp.loc[mask, "Set angle (°)"] = set_angle
                    df_temp.loc[mask, "Angle Error (°)"] = angle_error

                    df_plot = pd.concat([df_plot, df_temp[mask][[xparam, yparam]]], ignore_index=True)

                if plot_total_mean:
                    df_mean = pd.concat([df_mean, df_plot], ignore_index=True)

            mean_error = df_plot[yparam].mean()
            std_error = df_plot[yparam].std()
            max_error = df_plot[yparam].max()

            df_statistics = pd.concat([ 
                                df_statistics, 
                                pd.DataFrame({
                                    "Left out lines":       [left_out_lines],
                                    "z":                    [z_value],
                                    "Mean Angle Error (°)": [mean_error],
                                    "Std Dev (°)":          [std_error],
                                    "Max Angle Error (°)":  [max_error],
                                    })
                                ], ignore_index=True)

            
            z_title_dict: dict[str, float] = {}
            if len(plot_z) < 2:
                z_title_dict["z"] = z_value
                label_start = ""
            else:
                z_value_str = r"$z$ =" + f" {z_value}" + r" $[\mathrm{m}]$, "
                z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=False)
                label_start = z_value_str

            statistic_dict: dict[str, list[float]] = {}
            statistic_dict[r"$\mu$ $[^\circ]$"] = [np.nan]*len(left_out_lines_list)
            statistic_dict[r"$\sigma$ $[^\circ]$"] = [np.nan]*len(left_out_lines_list)
            if display_statistics:
                statistic_dict[r"$\mu$ $[^\circ]$"][lidx] = mean_error
                statistic_dict[r"$\sigma$ $[^\circ]$"][lidx] = std_error

            title_param_dict = {} if title_param_dict is None else title_param_dict
            label, title = get_labels_and_title(
                            translation_dict = translation_dict,
            
                            label_start  = label_start,
                            title_start = groupname,
            
                            title_parameters = title_parameters,
                            title_param_dict = meaned_values_float | z_title_dict | title_param_dict,
            
                            label_parameters = label_parameters,
                            label_param_dict = statistic_dict,
            
                            # paths
                            input_folder = input_folder, 
                            modelfolder = modelfolder,
                            modelname = modelnames[midx],
                            data_export = data_export,
                            sweep_dict = sweep_dict,
            
                            iteration = iteration,
                            left_out_lines = left_out_lines,
                            midx = lidx,

                            show_modelname_in_title = False,
                            show_modelname_in_label = False,
                            )

            color = get_color(lidx, len(left_out_lines_list), zidx, len(plot_z), 0, 1)
            
            fig, ax = apf.standard_scatter_plot_df(
                df = df_plot,
                x = xparam,
                y = yparam,
                z = None,

                label = label,
                color = color,
                translation_dict = translation_dict,

                fig = fig,
                ax = ax,

                title = title,
                x_label = xparam,
                y_label = yparam,
                z_label = None,

                xstyle = 'prefix',
                ystyle = 'prefix',
                )

            if plot_rolling_mean:
                df_plot['rolling_mean'] = df_plot[yparam].rolling(window=8, center=True).mean()
                ax.plot(df_plot[xparam], df_plot['rolling_mean'], label="rolling mean", color="tab:red")

    if plot_total_mean:
        df_mean_grouped = df_mean.groupby(xparam).mean().reset_index()
        ax.plot(df_mean_grouped[xparam], df_mean_grouped[yparam], label="mean", color="tab:blue")


    fig, ax = apf.plot_background(fig, ax)
    if label is not None:
        fig, ax = apf.dynamic_legend(fig, ax, fraction=fraction)
    
    # create output folder 
    plot_output_folder = output_folder / modelfolder / "angle error"
    makedirs(plot_output_folder, exist_ok=True)  
    apf.save_figure(fig, path = plot_output_folder / f"{modelnames[0]}-Angle Error-{"__".join(map(str, plot_z))}")

    return df_statistics

##############################################################################