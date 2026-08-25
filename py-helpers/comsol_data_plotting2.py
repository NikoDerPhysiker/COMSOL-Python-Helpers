# Author: Niko Bleidistel
# last change: 2026-08-19

##############################################################################
# import packages
##############################################################################


from os import makedirs

import pandas as pd
import numpy as np

import matplotlib.cm as cm
from matplotlib import ticker

from pathlib import Path
import importlib
import re

# custom packages
import advanced_plotting_functions as apf
_ = importlib.reload(apf)

import theory_fit_models as tfm
_ = importlib.reload(tfm)

import comsol_data_import as cdi
_ = importlib.reload(cdi)

# for type hints
from typing import Literal, Callable
import matplotlib.figure
import matplotlib.axes
from matplotlib.markers import MarkerStyle


##############################################################################
##############################################################################
# plotting helpers
##############################################################################
##############################################################################

def standard_scatter_plot(
        # plotting data
        x: list[float],
        y: list[float],
        z: list[float] | None = None,

        label: str | None = None,
        color: tuple[float, float, float] |str | None = apf.TITLECOLOR,
        markersize: float = apf.MARKERSIZE,
        translation_dict: dict[str, str] | None = None,

        # Figure and Axes
        fraction_textwidth: float = 1.0,
        fig: matplotlib.figure.Figure | None = None, 
        ax: matplotlib.axes.Axes | None = None,

        # labels and title
        title: str | None = None,
        x_label: str | None = None, 
        y_label: str | None = None,
        z_label: str | None = None,

        xstyle: Literal['sci', 'scientific', 'plain', 'prefix'] | None = 'plain',
        ystyle: Literal['sci', 'scientific', 'plain', 'prefix'] | None = 'plain',
        zstyle: Literal['sci', 'scientific', 'plain', 'prefix'] | None = 'prefix',
    ):
    """
    Create a standard scatter plot.
    Args:
        x (list[float])                             : x data
        y (list[float])                             : y data
        z (list[float] | None)                      : z data for color mapping (optional)
        label (str | None)                          : label for the data points (optional)
        color (str | None)                          : color for the data points (optional, ignored if z is provided)
        translation_dict (dict[str, str] | None)    : dictionary for translating labels (optional)

        fraction_textwidth (float)              : fraction of text width for figure size
        fig (matplotlib.figure.Figure | None)   : existing figure to plot on (optional)
        ax (matplotlib.axes.Axes | None)        : existing axes to plot on (optional)

        title (str | None)      : title of the plot (optional)
        x_label (str | None)    : label for the x-axis (optional)
        y_label (str | None)    : label for the y-axis (optional)
        z_label (str | None)    : label for the colorbar (optional, only used if z is provided)

        xstyle (Literal['sci', 'scientific', 'plain', 'prefix'] | None) : style for x-axis ticks (optional)
        ystyle (Literal['sci', 'scientific', 'plain', 'prefix'] | None) : style for y-axis ticks (optional)
        zstyle (Literal['sci', 'scientific', 'plain', 'prefix'] | None) : style for colorbar ticks (optional, only used if z is provided)

    Returns:
        tuple (Figure, Axes): the figure and axes objects of the plot
    """

    # create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = apf.get_fig_ax(fraction_textwidth=fraction_textwidth)

    # apply 'prefix' style to label
    if label is not None:
        labels = label.split("\n")
        label = ""
        for l in labels: 
            labelpart = apf.translate_and_prefix_label(l, translation_dict)
            label += labelpart + "\n"
        label = label.rstrip("\n")

    
    # plot
    if z is None:
        # mean y for multiple y values with same x
        if len(x) == len(y):
            unique_x = sorted(set(x))
            mean_y = [sum(y[i] for i in range(len(x)) if x[i] == ux) / sum(1 for i in range(len(x)) if x[i] == ux) for ux in unique_x]
            x = unique_x
            y = mean_y
        ax.scatter(x, y, label=label, color=color, s=markersize, marker=MarkerStyle('o'), rasterized=True)
    else:
        img = ax.scatter(x, y, c=z, cmap='viridis', label=label, s=markersize, marker=MarkerStyle('o'), rasterized=True)
        cbar = fig.colorbar(img, ax=ax)

        if translation_dict is not None and z_label is not None and z_label in translation_dict:
            z_label = translation_dict[z_label]

        zprefix = None
        if zstyle is not None:
            if zstyle == 'prefix':
                z_abs_max = max(abs(z_val) for z_val in z)
                zprefix, zsi_exponent, zexponent = apf.get_SI_prefix((z_abs_max, z_abs_max))
                z_scale_factor = 10 ** (-zsi_exponent)
                cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * z_scale_factor:g}"))
            else:
                cbar.ax.ticklabel_format(axis='y', style= zstyle, scilimits=(0,0), useMathText=True)
    
        if zstyle == 'prefix' and z_label is not None:
            if zprefix is not None:
                z_label = apf.set_prefix_in_label(string = z_label, prefix = zprefix)
            cbar.set_label(z_label, rotation=270, labelpad=10 + 2*apf.FONTSIZE,)

    # translate labels if translation_dict is provided
    if translation_dict is not None:
        if x_label is not None and x_label in translation_dict:
            x_label = translation_dict[x_label]
        if y_label is not None and y_label in translation_dict:
            y_label = translation_dict[y_label]

    # set background
    fig, ax = apf.plot_background(fig, ax, 
                                  title=title, 
                                  x_label=x_label, 
                                  y_label=y_label,
                                  xstyle = xstyle,
                                  ystyle = ystyle,
                                  )

    if label is not None:
        ax.legend(loc='best')

    return fig, ax

##############################################################################

def standard_scatter_plot_df(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str | None = None,
    **kwargs 
):
    """
    uses 'standard_scatter_plot()' to plot data from a pandas DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        x (str): The column name in the DataFrame for the x-axis data.
        y (str): The column name in the DataFrame for the y-axis data.
        z (str | None): The column name in the DataFrame for the z-axis data (optional).
        **kwargs: Additional keyword arguments to pass to standard_scatter_plot().
    """
    # convert the specified columns to float and then to lists
    x_data = df[x].astype(float).tolist()
    y_data = df[y].astype(float).tolist()
    z_data = df[z].astype(float).tolist() if z is not None else None

    # pass data and additional arguments to standard_scatter_plot()
    return standard_scatter_plot(
        x=x_data,
        y=y_data,
        z=z_data,
        **kwargs
    )

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
        min_distance:float = 3e-6,
        z_pos: float | None = None,
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

    ax.plot(x, y, label=fr"Biot-Savart with ${num_w} \times {num_h}$ conductors", color="tab:red")#, s=apf.MARKERSIZE)

    return fig, ax

##############################################################################

def add_temperature_theory_01_00(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        df_parameters: pd.DataFrame,
        xparam: str,
        z_pos: float = 0.0
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
        ax.plot(x, y, label="Theory: line source approximation", color="tab:blue")#, s=apf.MARKERSIZE)

    if False:
        # get y values
        y = [np.nan] * len(x)
        for i, x_val in enumerate(x):
            if xparam == "z":
                    z_plot = x_val
                    y_plot = 0.0
            if xparam == "y":
                    z_plot = z_pos 
                    y_plot = x_val
            
            T = tfm.temp_est_refraction_spreading(
                    df = df_layer,
                    power_per_length = power_per_length,
                    z = z_plot,
                    y = y_plot,
                    T_iso = T_iso,
                )
            
            y[i] = T - T_iso # store the relative temperature (T - T_iso) in the y array
            
        # plot
        ax.plot(x, y, label="Theory: extended source per layer,\n increased spreading per thermal refraction", color="tab:green")#, s=apf.MARKERSIZE)

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
            
            T = tfm.temp_est_integration_spreading(
                    df = df_layer,
                    power_per_length = power_per_length,
                    z = z_plot,
                    y = y_plot,
                    T_iso = T_iso,
                )
            
            y[i] = T - T_iso # store the relative temperature (T - T_iso) in the y array
            
        # plot
        ax.plot(x, y, label="Theory: extended source,\n        spreading per integration from top", color="tab:red")#, s=apf.MARKERSIZE)


    return fig, ax

##############################################################################
##############################################################################

def add_magnetic_theory_02_00(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        df_parameters: pd.DataFrame,
        xparam: str,
        Bidx: int,
        min_distance:float = 3e-6,
        z_pos: float | None = None,
        withsupply: bool = False,
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
        

    ax.plot(x, y, label=fr"Biot-Savart: 3 straight conductors", color="tab:red")#, s=apf.MARKERSIZE)

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

##############################################################################

def import_COMSOLTXT_to_df(input_folder: Path, modelfolder: str, modelname: str, ending: str, data_export: str = "Data Export", T_0: float = T_0):
    path = input_folder / modelfolder / modelname / data_export / f"{modelname}{ending}"
    header, df = cdi.read_comsol_export(str(path))

    # use relative temperatures
    if "T (K)" in df.columns:
        df["T (K)"] = df["T (K)"].apply(lambda T: T - T_0)

    return header, df

##############################################################################

def zaxis_plot(
        translation_dict: dict[str, str],

        # theory
        magnetic_theory: Callable,
        temperature_theory: Callable,

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
        ):
    group_length = len(modelnames)
    model_labels = [modelname.split("-")[-1] for modelname in modelnames]

    # import dataframes for each model in the group
    list_df = []
    x_values = set()
    y_values = set()
    for midx in range(group_length):
        # get all depth dataframes
        depth_header_data, df = import_COMSOLTXT_to_df(input_folder, modelfolder, modelnames[midx], ending, data_export)

        df["x"] = cdi.round_to_6_sig_digits(df["x"])
        x_values.update(df["x"].dropna())

        df["y"] = cdi.round_to_6_sig_digits(df["y"])
        y_values.update(df["y"].dropna())

        list_df.append(df)

    x_value = np.mean(list(x_values))
    if x_value < 1e-20:
        x_value = 0.0

    y_value = np.mean(list(y_values))
    if y_value < 1e-20:
        y_value = 0.0

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
        cmap = cm.get_cmap('viridis')
        
        for midx in range(group_length):
            df_plot = list_df[midx][[xparam, yparam]].copy()

            x_value_str = r"$\bm{x =" + f" {x_value}" + r" [\mathrm{m}]}$"
            x_value_str = apf.set_prefix_in_number_unit_string(x_value_str, bm=True)

            y_value_str = r"$\bm{y =" + f" {y_value}" + r" [\mathrm{m}]}$"
            y_value_str = apf.set_prefix_in_number_unit_string(y_value_str, bm=True)

            fig, ax = standard_scatter_plot_df(
                        df = df_plot,
                        x = xparam,
                        y = yparam,
                        z = None,

                        label = model_labels[midx],
                        color = cmap(midx / group_length),
                        translation_dict = translation_dict,

                        fig = fig,
                        ax = ax,

                        title = groupname + " (" + x_value_str + ", " + y_value_str + ")",
                        x_label = xparam,
                        y_label = yparam,
                    )

        # Theory
        if magnetic_theory is not None or temperature_theory is not None:
            df_parameters = import_csv_to_df(input_folder, modelfolder, modelnames[0], "-parameters.csv", data_export)
            
        if "mf.B" in yparam and magnetic_theory is not None:
            Bidx = THEORY_FORMULA[yparam]
            fig, ax = magnetic_theory(fig, ax, df_parameters, xparam=xparam, Bidx=Bidx, min_distance=3e-6, z_pos = None)

        if "T (K)" in yparam and temperature_theory is not None:
            fig, ax = temperature_theory(fig, ax, df_parameters, xparam=xparam)

        fig, ax = apf.plot_background(fig, ax)
        
        ax.legend(
            loc="upper center", 
            bbox_to_anchor=(0.5, -0.15/fraction), 
            ncol=2
            )
        
        # remove any text in parentheses and leading/trailing whitespace
        clean_yparam = re.sub(r"\s*\(.*?\)", "", str(yparam)) 

        # create output folder 
        output_folder = output_folder / modelfolder / "z axis"
        makedirs(output_folder, exist_ok=True)  

        apf.save_figure(fig, path = output_folder / f"{clean_yparam}_over_{xparam}")

##############################################################################

def yaxis_plot(
        translation_dict: dict[str, str],

        # theory
        magnetic_theory: Callable,
        temperature_theory: Callable,

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
        plot_z: list[float] = [-3e-06, -9e-06],
        insulator: bool = False,
        width_limit_param: str | None = "conductor_all_width" 
        ):
    
    group_length = len(modelnames)
    model_labels = [modelname.split("-")[-1] for modelname in modelnames]

    df_parameters = import_csv_to_df(input_folder, modelfolder, modelnames[0], "-parameters.csv", data_export) # assume constant values

    # import dataframes for each model in the group
    list_df = []
    x_values = set()
    z_values = set()
    for midx in range(group_length):
        # get all depth dataframes
        depth_header_data, df = import_COMSOLTXT_to_df(input_folder, modelfolder, modelnames[midx], ending, data_export)

        df["x"] = cdi.round_to_6_sig_digits(df["x"])
        x_values.update(df["x"].dropna())

        df["z"] = cdi.round_to_6_sig_digits(df["z"])
        z_values.update(df["z"].dropna())

        list_df.append(df)

    x_value = np.mean(list(x_values)) 
    if x_value < 1e-20:
        x_value = 0.0

    unique_z_values = sorted(list(z_values))

    if insulator:
        mask = df_parameters["name"] == "insulator_height"
        insulator_height = float(df_parameters.loc[mask, "evaluated_value"].iloc[0])
        plot_z = [z - insulator_height for z in plot_z]

    # Round the z values to 6 significant figures to avoid floating point issues when comparing with unique_z_values
    positive_values = np.where(plot_z == 0, 1e-20, np.abs(plot_z))
    exponent = np.floor(np.log10(positive_values)) # get magnitude 
    factor = 10 ** (5 - exponent) # shift factor
    plot_z = (np.round(np.array(plot_z) * factor) / factor).tolist() # round shifted values and shift back
    plot_z = (np.where(plot_z == 0.0, 0.0, plot_z)).tolist() 

    for z_value in plot_z:
        if z_value not in unique_z_values:
            print(f"{z_value} is not a valid z value.")

    width_limit = np.inf
    if width_limit_param is not None and width_limit_param in df_parameters["name"].values:
        mask = df_parameters["name"] == width_limit_param
        width_limit = 1.5 / 2 * float(df_parameters.loc[mask, "evaluated_value"].iloc[0])

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
            
        # Data
        cmap = cm.get_cmap('viridis')

        for z_value in plot_z:
            for midx in range(group_length):
                df_plot = list_df[midx][[xparam, yparam]].copy()
                mask = list_df[midx]["z"] == z_value
                df_plot = df_plot[mask]
                mask = df_plot[xparam].abs() <= width_limit
                df_plot = df_plot[mask]

                x_value_str = r"$\bm{x =" + f" {x_value}" + r" [\mathrm{m}]}$"
                x_value_str = apf.set_prefix_in_number_unit_string(x_value_str, bm=True)

                z_value_str = r"$\bm{z =" + f" {z_value}" + r" [\mathrm{m}]}$"
                z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=True)

                if len(plot_z) < 2:
                    label = model_labels[midx]
                    title = groupname + " (" + x_value_str + "," + z_value_str + ")"
                else:
                    label = model_labels[midx] + " at " + z_value_str
                    title = groupname + " (" + x_value_str + ")"

                fig, ax = standard_scatter_plot_df(
                            df = df_plot,
                            x = xparam,
                            y = yparam,
                            z = None,

                            label = label,
                            color = cmap(midx / group_length / len(plot_z) + plot_z.index(z_value) / len(plot_z) / group_length),
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
                fig, ax = magnetic_theory(fig, ax, df_parameters, xparam=xparam, Bidx=Bidx, min_distance=3e-6, z_pos = z_value)

            if "T (K)" in yparam and temperature_theory is not None:
                fig, ax = temperature_theory(fig, ax, df_parameters, xparam=xparam, z_pos = z_value)

            fig, ax = apf.plot_background(fig, ax)
            
            ax.legend(
                loc="upper center", 
                bbox_to_anchor=(0.5, -0.15/fraction), 
                ncol=2
                )
        
        # remove any text in parentheses and leading/trailing whitespace
        clean_yparam = re.sub(r"\s*\(.*?\)", "", str(yparam)) 

        # create output folder 
        output_folder = output_folder / modelfolder / "y axis"
        makedirs(output_folder, exist_ok=True)  

        apf.save_figure(fig, path = output_folder / f"{clean_yparam}_over_{xparam}")

##############################################################################

def xaxis_plot(
        translation_dict: dict[str, str],

        # theory
        magnetic_theory: Callable,
        temperature_theory: Callable,

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
        plot_z: list[float] = [-3e-06, -9e-06],
        insulator: bool = False,
        length_limit_param: str | None = "conductor_all_length" 
        ):
    
    group_length = len(modelnames)
    model_labels = [modelname.split("-")[-1] for modelname in modelnames]

    df_parameters = import_csv_to_df(input_folder, modelfolder, modelnames[0], "-parameters.csv", data_export) # assume constant values

    # import dataframes for each model in the group
    list_df = []
    y_values = set()
    z_values = set()
    for midx in range(group_length):
        # get all depth dataframes
        depth_header_data, df = import_COMSOLTXT_to_df(input_folder, modelfolder, modelnames[midx], ending, data_export)

        df["y"] = cdi.round_to_6_sig_digits(df["y"])
        y_values.update(df["y"].dropna())

        df["z"] = cdi.round_to_6_sig_digits(df["z"])
        z_values.update(df["z"].dropna())

        list_df.append(df)

    y_value = np.mean(list(y_values)) 
    if y_value < 1e-20:
        y_value = 0.0

    unique_z_values = sorted(list(z_values))

    if insulator:
        mask = df_parameters["name"] == "insulator_height"
        insulator_height = float(df_parameters.loc[mask, "evaluated_value"].iloc[0])
        plot_z = [z - insulator_height for z in plot_z]

    # Round the z values to 6 significant figures to avoid floating point issues when comparing with unique_z_values
    positive_values = np.where(plot_z == 0, 1e-20, np.abs(plot_z))
    exponent = np.floor(np.log10(positive_values)) # get magnitude 
    factor = 10 ** (5 - exponent) # shift factor
    plot_z = (np.round(np.array(plot_z) * factor) / factor).tolist() # round shifted values and shift back
    plot_z = (np.where(plot_z == 0.0, 0.0, plot_z)).tolist() 

    for z_value in plot_z:
        if z_value not in unique_z_values:
            print(f"{z_value} is not a valid z value.")

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
            
        # Data
        cmap = cm.get_cmap('viridis')

        for z_value in plot_z:
            for midx in range(group_length):
                df_plot = list_df[midx][[xparam, yparam]].copy()
                mask = list_df[midx]["z"] == z_value
                df_plot = df_plot[mask]
                mask = df_plot[xparam].abs() <= length_limit
                df_plot = df_plot[mask]

                y_value_str = r"$\bm{y =" + f" {y_value}" + r" [\mathrm{m}]}$"
                y_value_str = apf.set_prefix_in_number_unit_string(y_value_str, bm=True)

                z_value_str = r"$\bm{z =" + f" {z_value}" + r" [\mathrm{m}]}$"
                z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=True)

                if len(plot_z) < 2:
                    label = model_labels[midx]
                    title = groupname + " (" + y_value_str + "," + z_value_str + ")"
                else:
                    label = model_labels[midx] + " at " + z_value_str
                    title = groupname + " (" + y_value_str + ")"

                fig, ax = standard_scatter_plot_df(
                            df = df_plot,
                            x = xparam,
                            y = yparam,
                            z = None,

                            label = label,
                            color = cmap(midx / group_length / len(plot_z) + plot_z.index(z_value) / len(plot_z) / group_length),
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
                fig, ax = magnetic_theory(fig, ax, df_parameters, xparam=xparam, Bidx=Bidx, min_distance=3e-6, z_pos = z_value)

            if "T (K)" in yparam and temperature_theory is not None:
                fig, ax = temperature_theory(fig, ax, df_parameters, xparam=xparam, z_pos = z_value)

            fig, ax = apf.plot_background(fig, ax)
            
            ax.legend(
                loc="upper center", 
                bbox_to_anchor=(0.5, -0.15/fraction), 
                ncol=2
                )
        
        # remove any text in parentheses and leading/trailing whitespace
        clean_yparam = re.sub(r"\s*\(.*?\)", "", str(yparam)) 

        # create output folder 
        output_folder = output_folder / modelfolder / "x axis"
        makedirs(output_folder, exist_ok=True)  

        apf.save_figure(fig, path = output_folder / f"{clean_yparam}_over_{xparam}")


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
        length_limit_param: str | None = "conductor_all_length",
        width_limit_param: str | None = "conductor_all_width" 
        ):
    
    group_length = len(modelnames)
    model_labels = [modelname.split("-")[-1] for modelname in modelnames]

    df_parameters = import_csv_to_df(input_folder, modelfolder, modelnames[0], "-parameters.csv", data_export) # assume constant values

    # import dataframes for each model in the group
    list_df = []
    z_values = set()
    for midx in range(group_length):
        # get all depth dataframes
        depth_header_data, df = import_COMSOLTXT_to_df(input_folder, modelfolder, modelnames[midx], ending, data_export)

        df["z"] = cdi.round_to_6_sig_digits(df["z"])
        z_values.update(df["z"].dropna())

        list_df.append(df)

    z_value = np.mean(list(z_values)) 
    if z_value < 1e-20:
        z_value = 0.0


    length_limit = np.inf
    if length_limit_param is not None and length_limit_param in df_parameters["name"].values:
        mask = df_parameters["name"] == length_limit_param
        length_limit = 1.5 / 2 * float(df_parameters.loc[mask, "evaluated_value"].iloc[0])

    width_limit = np.inf
    if width_limit_param is not None and width_limit_param in df_parameters["name"].values:
        mask = df_parameters["name"] == width_limit_param
        width_limit = 1.5 / 2 * float(df_parameters.loc[mask, "evaluated_value"].iloc[0])

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

        for midx in range(group_length):
                df_plot = list_df[midx][[xparam, yparam, zparam]].copy()
                df_plot = df_plot.dropna()
                df_plot = df_plot[(df_plot[xparam].abs() <= length_limit) & (df_plot[yparam].abs() <= width_limit)]

                z_value_str = r"$\bm{z =" + f" {z_value}" + r" [\mathrm{m}]}$"
                z_value_str = apf.set_prefix_in_number_unit_string(z_value_str, bm=True)

                title = groupname + " (" + model_labels[midx] + ", " + z_value_str + ")"

                fig, ax = standard_scatter_plot_df(
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
                            markersize = 0.001*apf.MARKERSIZE,
                        )
        
        # remove any text in parentheses and leading/trailing whitespace
        clean_zparam = re.sub(r"\s*\(.*?\)", "", str(zparam)) 

        # create output folder 
        output_folder = output_folder / modelfolder / "xy plane"
        makedirs(output_folder, exist_ok=True)  

        apf.save_figure(fig, path = output_folder / f"{model_labels[midx]}-{clean_zparam}_over_{xparam}{yparam}_plane-{z_value}")

##############################################################################