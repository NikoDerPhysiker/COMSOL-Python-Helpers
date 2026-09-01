# Author: Niko Bleidistel
# last change: 2026-08-18

# Hint: This is an old file, which is not used anymore. The new file is 'comsol_data_plotting2.py'. Which has fundamentaly changed.

##############################################################################
# import packages
##############################################################################

import re
import io
import pandas as pd
from pathlib import Path
from IPython.display import display


from matplotlib import ticker

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# For type hints
from typing import Any, Literal, cast

import matplotlib.pyplot as plt

import importlib
# custom packages
import plot_functions as pfs
importlib.reload(pfs)

import time_logging as tl          
importlib.reload(tl)

# For type hints
import matplotlib.axes
import matplotlib.figure
from matplotlib.markers import MarkerStyle


##############################################################################
##############################################################################
# interpolation functions
##############################################################################
##############################################################################

if False: # old function, not used anymore, needs fixing (and inefficient)
    def filter_and_interpolate_df(df: pd.DataFrame,
                                grid_dict: dict | None = None,
                                limit_dict: dict | None = None,
                                ):
        """
        Filters the DataFrame using a safety margin to prevent edge explosions,
        performs fast spatial interpolation, and clips back to strict limits.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be filtered and interpolated.
            grid_dict (dict | None): A dictionary specifying the exact coordinates for interpolation.
            limit_dict (dict | None): A dictionary specifying the min and max limits for each column.
        
        Returns:
            (pd.DataFrame): The filtered and interpolated DataFrame.
        """

        if df is None:
            raise ValueError("The input DataFrame 'df' is None.")
        tl.log_message(f"Starting filtering and interpolation of DataFrame.")
            
        filtered_df = df.copy()
        tl.log_message(f"Initial DataFrame shape: {len(filtered_df)} rows, {len(filtered_df.columns)} columns.")
        
        # 1. Apply initial filtering with a SAFETY MARGIN to prevent edge effects during interpolation.
        if limit_dict:
            for column, limits in limit_dict.items():
                if column in filtered_df.columns:
                    lower, upper = sorted(limits)
                    span = upper - lower
                    # 10% safety margin added to the outer boundaries
                    margin = span * 0.10 if span > 0 else 1e-6
                    
                    filtered_df = filtered_df[
                        (filtered_df[column] >= (lower - margin)) & 
                        (filtered_df[column] <= (upper + margin))
                    ]
            tl.log_message(f"Filtering complete. DataFrame shape after filtering: {len(filtered_df)} rows, {len(filtered_df.columns)} columns.")
        if filtered_df.empty:
            print("Warning: No data points remain after applying safety limits.")
            tl.log_message(f"WARNING: No data points remain after applying safety limits.")
            return filtered_df

        if not grid_dict:
            # If no grid specified, clip to strict limits immediately and return
            if limit_dict:
                for column, limits in limit_dict.items():
                    if column in filtered_df.columns:
                        lower, upper = sorted(limits)
                        filtered_df = filtered_df[(filtered_df[column] >= lower) & (filtered_df[column] <= upper)]
            tl.log_message(f"No grid specified. Returning filtered DataFrame with shape: {len(filtered_df)} rows, {len(filtered_df.columns)} columns.")
            return filtered_df

        # Automatically detect coordinate columns (x, y, z) that are NOT defined in grid_dict
        possible_coords = ['x', 'y', 'z']
        preserved_axes = [col for col in possible_coords if col in filtered_df.columns and col not in grid_dict]
        
        interp_features = list(grid_dict.keys())
        
        target_columns = [
            col for col in filtered_df.columns 
            if col not in interp_features and col not in preserved_axes
        ]

        if not target_columns:
            raise ValueError("No target columns left for interpolation.")

        # Build the evaluation grid for the specified grid features
        grid_arrays = [np.array(grid_dict[col]) for col in interp_features]
        meshgrid = np.meshgrid(*grid_arrays, indexing='ij')
        flat_grid = np.vstack([m.flatten() for m in meshgrid]).T
        
        results = []
        tl.log_message(f"Starting interpolation.")
        
        # CASE 1: There are missing coordinates in grid_dict (like 'z'), which must be preserved fully
        if preserved_axes:
            filtered_df = filtered_df.sort_values(by=preserved_axes)
            
            fit_coords_cols = interp_features + preserved_axes
            source_coords = filtered_df[fit_coords_cols].values
            
            for point in flat_grid:
                point_df = pd.DataFrame(index=filtered_df.index)
                
                for idx, col in enumerate(interp_features):
                    point_df[col] = point[idx]
                    
                for col in preserved_axes:
                    point_df[col] = filtered_df[col]
                    
                query_points = np.hstack([
                    np.tile(point, (len(filtered_df), 1)), 
                    filtered_df[preserved_axes].values
                ])
                    
                for target_col in target_columns:
                    values = filtered_df[target_col].values
                    
                    try:
                        interpolator = LinearNDInterpolator(source_coords, values) # type: ignore
                        point_df[target_col] = interpolator(query_points)
                    except Exception:
                        interpolator = NearestNDInterpolator(source_coords, values) # type: ignore
                        point_df[target_col] = interpolator(query_points)
                    
                results.append(point_df)
            
            if not results:
                final_df = pd.DataFrame(columns=df.columns)
            else:
                final_df = pd.concat(results, ignore_index=True)
            
        # CASE 2: All coordinates are explicitly specified in grid_dict
        else:
            source_coords = filtered_df[interp_features].values
            result_dict = {col: flat_grid[:, i] for i, col in enumerate(interp_features)}
            final_df = pd.DataFrame(result_dict)
            
            for target_col in target_columns:
                values = filtered_df[target_col].values
                try:
                    interpolator = LinearNDInterpolator(source_coords, values) # type: ignore
                    final_df[target_col] = interpolator(flat_grid)
                except Exception:
                    interpolator = NearestNDInterpolator(source_coords, values) # type: ignore
                    final_df[target_col] = interpolator(flat_grid)
        tl.log_message(f"Interpolation complete. DataFrame shape after interpolation: {len(final_df)} rows, {len(final_df.columns)} columns.")

        # 2. FINAL CLIPPING: Crop back to the strict limits requested by the user
        if limit_dict and not final_df.empty:
            for column, limits in limit_dict.items():
                if column in final_df.columns:
                    lower, upper = sorted(limits)
                    final_df = final_df[
                        (final_df[column] >= lower) & (final_df[column] <= upper)
                    ]
        tl.log_message(f"Final clipping complete. DataFrame shape after clipping: {len(final_df)} rows, {len(final_df.columns)} columns.")
        return final_df
        
    ##############################################################################

    def mask_and_interpolate_data(df: pd.DataFrame,
                                x_params: list[str],
                                y_params: list[str],
                                grid_dict: dict | None = None,
                                limit_dict: dict | None = None,
                                ):
        """
        Import Data from a COMSOL export txt file, filter and interpolate in prepatration for plotting.

        Args:
            df (pandas.DataFrame): The DataFrame containing the COMSOL data.
            x_params (list[str]): The parameters to use for the x-axis.
            y_params (list[str]): The parameters to use for the y-axis.
            grid_dict (dict | None, optional): Dictionary defining the grid for interpolation. Defaults to None.
            limit_dict (dict | None, optional): Dictionary defining the limits for filtering. Defaults to None.
        
        Returns:
            (pandas.DataFrame): The filtered and interpolated DataFrame.
        """
        tl.log_message(f"Starting mask and interpolation of DataFrame.")
        # mask before interpolation to reduce computational load
        keep_columns = ['x', 'y', 'z'] + x_params + y_params
        actual_columns_2keep = list(dict.fromkeys(keep_columns))                                     # ensure that the list of columns to keep does not contain duplicates
        df_interpol = df[actual_columns_2keep]                                                       # filter the DataFrame to keep only the relevant columns

        # filtering and interpolation of the DataFrame 
        df_curves = filter_and_interpolate_df(
            df=df_interpol, 
            grid_dict=grid_dict, 
            limit_dict=limit_dict,
            )
        tl.log_message(f"Mask and interpolation complete. DataFrame shape after processing: {len(df_curves)} rows, {len(df_curves.columns)} columns.")
        return df_curves

##############################################################################
##############################################################################
# plotting functions
##############################################################################
##############################################################################

def plot_comsol_data(df: pd.DataFrame,
                     header_data: dict,
                     x_column: str,
                     y_column: str,
                     x_label: str|None = None,
                     y_label: str|None = None,
                     title: str|None = None,
                     add_title_info: bool = True,
                     label: str|None = None,
                     marker = 'o', 
                     color = 'tab:blue', 
                     labelcolor = 'black', 
                     fig:matplotlib.figure.Figure|None = None, 
                     ax:matplotlib.axes.Axes|None = None,
                     show_legend: bool = True,
                     legend_loc: str = 'best',
                     xscale: str|None = None,
                     yscale: str|None = None,
                     xstyle: Literal['sci', 'scientific', 'plain', 'prefix'] = "prefix",
                     ystyle: Literal['sci', 'scientific', 'plain', 'prefix'] = "prefix",
                     grid: bool = True,
                     ):
    """
    Plots the COMSOL data from the DataFrame using the provided header information for labeling and titling the plot.
    
    Args:
        df (pd.DataFrame):                       pandas DataFrame containing the numerical data to be plotted
        header_data (dict):                      dictionary containing the header information ("Model", "Date", etc.)
        x_column (str, optional):                name of the column in the DataFrame to be used for x axis
        y_column (str, optional):                name of the column in the DataFrame to be used for y axis
        x_label (str | None, optional):          label for x axis
        y_label (str | None, optional):          label for y axis
        title (str | None, optional):            title of the plot
        add_title_info (bool, optional):         whether to add model and date information from header_data to the title
        label (str | None, optional):            label for the data points (for legend)
        marker (str, optional):                  marker style for the data points
        color (str, optional):                   color for the data points and error bars
        labelcolor (str, optional):              color for the axis labels
        fig (plt.Figure | None, optional):       matplotlib figure to plot on (if None, a new figure is created)
        ax (plt.Axes | None, optional):          matplotlib axis to plot on (if None, a new axis is created)
        show_legend (bool, optional):            whether to show the legend
        legend_loc (str, optional):              location of the legend
        xscale (str | None, optional):           scale for x axis (e.g. 'linear', 'log')
        yscale (str | None, optional):           scale for y axis (e.g. 'linear', 'log')
        grid (bool, optional):                   whether to show grid

    Returns:
        (plt.Figure, plt.Axes):                the matplotlib figure and axis objects containing the plot

    Raises:
        ValueError: If any of the input parameters are of incorrect type or if required columns are not found in the DataFrame or header data.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame containing the numerical data to be plotted.")
    if not isinstance(header_data, dict):
        raise ValueError("header_data must be a dictionary containing the header information (e.g. 'Model', 'Date').")
    if not isinstance(x_column, str):
        raise ValueError("x_column must be a string representing the name of the column in the DataFrame to be used for x axis.")
    if not isinstance(y_column, str):
        raise ValueError("y_column must be a string representing the name of the column in the DataFrame to be used for y axis.")

    if x_label is None:
        x_label = x_column
    if y_label is None:
        y_label = y_column
    if label is None:
        label = y_column

    if add_title_info:
        try: 
            model = header_data["Model"].strip('.mph').replace("_", " ")
            date = header_data["Date"]
            titleprefix = f"{model} - {date}"
        except KeyError:
            print("Model and/or Date information not found in header_data.")
            titleprefix = None

        if titleprefix is None:
            plot_title = title
        else:
            if title is None:
                plot_title = titleprefix
            else:
                plot_title = f"{title} \n ({titleprefix})"
    else:
        plot_title = title

    fig, ax = pfs.u_plot_scatter_with_error_bars(
        df[x_column].astype(float).tolist(),
        df[y_column].astype(float).tolist(),
        x_label=x_label,
        y_label=y_label,
        title=plot_title,
        label=label,
        marker=marker,
        color=color,
        labelcolor=labelcolor,
        fig=fig,
        ax=ax,
        show_legend=show_legend,
        legend_loc=legend_loc,
        xscale=xscale,
        yscale=yscale,
        xstyle= xstyle,
        ystyle= ystyle,
        grid=grid,
        plot_error=False,
    )
    return fig, ax

##############################################################################
##############################################################################

def add_param_and_translate(
            param_name: str,
            df_curves: pd.DataFrame,
            df_param: pd.DataFrame,
            label: str | None = None,
            translation_dict: dict | None = None,
            header_data: dict | None = None,
            ):
        """
        Adds a parameter value to the label string, translating the parameter name if a translation dictionary is provided.

        Args:
            param_name (str): The name of the parameter to add to the label.
            df_curves (pd.DataFrame): The DataFrame containing the curve data.
            df_param (pd.DataFrame): The DataFrame containing the parameter data.
            label (str | None): The current label string to which the parameter value will be added.
            translation_dict (dict | None, optional): A dictionary for translating parameter names to more descriptive labels. Defaults to None.
            header_data (dict | None, optional): A dictionary containing header information, used for adding units to the label. Defaults to None.
        
        Returns:
            (str): The updated label string with the parameter value added.
        """
        
        if param_name in df_curves.columns:
            value = df_curves[param_name].iloc[0]
        elif param_name in df_param.columns:
            value = df_param[param_name].iloc[0]
        else:
            tl.log_message(f"Warning: '{param_name}' not found in df_curves or df_param.")
        
        
        if translation_dict and param_name in translation_dict:
            translated_name = translation_dict[param_name]
            bracket_match = re.match(r".*\[(.*)\]$", translated_name)
            if bracket_match:
                unit = bracket_match.group(1)
                translated_name = translated_name.replace(f" [{unit}]", "")
            else:
                unit = None
        else:
            translated_name = param_name

        translated_name = translated_name.replace("-axis", "") 
        
        if label is None:
            label = ""
        else:
            label += "\n"

        siprefix, xsi_exponent, exponent_diff = pfs.get_SI_prefix((value, value))  # Get the SI prefix for the value
        scaled_value = value * 10 ** (-xsi_exponent)

        label += f"{translated_name} = {scaled_value:.3g}"

        if translation_dict and param_name in translation_dict:
            label += f" [{siprefix}{unit}]"
        
        return label

##############################################################################

def standard_plot(output_folder: str | None,
                  x_param: str,
                  y_param: str,
                  header_data: dict,
                  df_curves: pd.DataFrame,
                  df_param: pd.DataFrame,
                  title_params: list[str] | None = None,
                  sweep_params: list[str] | None = None,
                  title: str | None = None,
                  add_title_info: bool = True,
                  custom_label: str | None = None,
                  translation_dict: dict | None = None,
                  fig = None, 
                  ax = None,
                  color='tab:blue',
                  save_plot = True
                  ):    
    """
    Standard plot function for COMSOL data of magnetic field simulations.

    Args:
        output_folder (str):                        The folder where the plot will be saved.
        x_param (str):                              The parameter to use for the x-axis.
        y_param (str):                              The parameter to use for the y-axis.
        header_data (dict):                         The header data extracted from the COMSOL export file.
        df_curves (pd.DataFrame):                   The filtered and interpolated DataFrame ready for plotting (should contain the columns x_param and y_param).
        df_param (pd.DataFrame):                    The original DataFrame containing the parameter values which will be used for the plot title ('root.current_A (A)' and 'root.current_B (A)').
        title_params (list[str], optional):         The parameters to include in the plot title. Defaults to None.
        sweep_params (list[str], optional):         The parameters that were swept (i.e. in interpolation or parameter sweep), which will be used for the legend label if custom_label is not provided. Defaults to None.
        title (str, optional):                      A custom title for the plot. If not provided, the title will be generated based on title_params. Defaults to None.
        add_title_info (bool, optional):            Whether to add model and date information from header_data to the title. Defaults to True.
        custom_label (str, optional):               A custom label for the legend. If not provided, the label will be generated based on sweep_params. Defaults to None.
        translation_dict (dict, optional):          A dictionary for translating parameter names to more descriptive labels for the axes. Defaults to None.
        fig (matplotlib.figure.Figure, optional):   A matplotlib figure object to plot on. If None, a new figure will be created. Defaults to None.
        ax (matplotlib.axes.Axes, optional):        A matplotlib axes object to plot on. If None, a new axes will be created. Defaults to None.
        color (str, optional):                      The color of the plot elements. Defaults to 'tab:blue'.
        save_plot (bool, optional):                 Whether to save the plot as a PNG file. Defaults to True.
    
    Returns:
        (matplotlib.figure.Figure, matplotlib.axes.Axes): The matplotlib figure and axes objects containing the plot.
    """

    # create label for legend
    if custom_label:
        label = custom_label
    elif sweep_params:
        label = ""
        for sweep_param in sweep_params:
            label = add_param_and_translate(
                param_name=sweep_param,
                df_curves=df_curves,
                df_param=df_param,
                label=label,
                translation_dict=translation_dict,
                header_data=header_data,
            )
        label = label.strip()  # type: ignore
    else:
        label = None

        
    # create title for plot
    if title_params:
        if title is None:
            title = ""
        title += "\n"
        for title_param in title_params:
            title = add_param_and_translate(
                param_name=title_param,
                df_curves=df_curves,
                df_param=df_param,
                label=title,
                translation_dict=translation_dict,
                header_data=header_data,
            )
    else:
        title = title

    # extract descriptions
    if translation_dict:
        try: 
            x_discription = translation_dict.get(x_param)
        except Exception as e:
            print(f"Error retrieving x_param description: {e}")
            x_discription = x_param

        try:
            y_discription = translation_dict.get(y_param)
        except Exception as e:
            print(f"Error retrieving y_param description: {e}")
            y_discription = y_param

    else:
        x_discription = x_param
        y_discription = y_param

    # plot the data
    fig, ax = plot_comsol_data(
        df=df_curves,
        header_data = header_data,
        x_column = x_param,
        y_column = y_param,
        x_label = x_discription,
        y_label = y_discription,
        title = title,
        add_title_info = add_title_info,
        label = label,
        marker = 'o', 
        color = color, 
        labelcolor = 'black', 
        fig = fig, 
        ax = ax,
        show_legend = True,
        legend_loc = 'best',
        xscale = None,
        yscale = None,
        xstyle = 'prefix',
        ystyle = 'prefix',
        grid = True,
        )
    
    if save_plot:
        if not output_folder:
            raise ValueError("output_folder must be provided if save_plot is True.")
        modelname = str(header_data.get('Model')).replace(".mph", "")
        output_path = Path(output_folder) / f"{modelname}_{x_param}_vs_{y_param}.png"
        fig.savefig(str(output_path), dpi=300)
    
    return fig, ax

##############################################################################
##############################################################################

def plane_plot(
        output_folder: str | None,
        x_param: str,
        y_param: str,
        z_param: str,
        header_data: dict,
        df: pd.DataFrame,
        df_param: pd.DataFrame,
        title_params: list[str] | None = None,
        sweep_params: list[str] | None = None,
        title: str | None = None,
        add_title_info: bool = True,
        custom_label: str | None = None,
        translation_dict: dict | None = None,
        save_plot = True,
        fig = None, 
        ax = None,
        colormap='viridis',
        labelcolor = 'black',
        xscale = None,
        yscale = None,
        xstyle = 'prefix',
        ystyle = 'prefix',
        zstyle = 'prefix',
        grid = True,
        ):    
    """
    Standard plot function for COMSOL data of magnetic field simulations.

    Args:
        output_folder (str):                        The folder where the plot will be saved.
        x_param (str):                              The parameter to use for the x-axis.
        y_param (str):                              The parameter to use for the y-axis.
        z_param (str):                              The parameter plotted via colormap.
        header_data (dict):                         The header data extracted from the COMSOL export file.
        df (pd.DataFrame):                          The DataFrame containing the data to be plotted.
        df_param (pd.DataFrame):                    The original DataFrame containing the parameter values which will be used for the plot title ('root.current_A (A)' and 'root.current_B (A)').
        title_params (list[str], optional):         The parameters to include in the plot title. Defaults to None.
        sweep_params (list[str], optional):         The parameters that were swept (i.e. in interpolation or parameter sweep), which will be used for the legend label if custom_label is not provided. Defaults to None.
        title (str, optional):                      A custom title for the plot. If not provided, the title will be generated based on title_params. Defaults to None.
        add_title_info (bool, optional):            Whether to add model and date information from header_data to the title. Defaults to True.
        custom_label (str, optional):               A custom label for the legend. If not provided, the label will be generated based on sweep_params. Defaults to None.
        translation_dict (dict, optional):          A dictionary for translating parameter names to more descriptive labels for the axes. Defaults to None.
        fig (matplotlib.figure.Figure, optional):   A matplotlib figure object to plot on. If None, a new figure will be created. Defaults to None.
        ax (matplotlib.axes.Axes, optional):        A matplotlib axes object to plot on. If None, a new axes will be created. Defaults to None.
        color (str, optional):                      The color of the plot elements. Defaults to 'tab:blue'.
        save_plot (bool, optional):                 Whether to save the plot as a PNG file. Defaults to True.
    
    Returns:
        (matplotlib.figure.Figure, matplotlib.axes.Axes): The matplotlib figure and axes objects containing the plot.
    """

    # create label for legend
    if custom_label:
        label = custom_label
    elif sweep_params:
        label = ""
        for sweep_param in sweep_params:
            label = add_param_and_translate(
                param_name=sweep_param,
                df_curves=df,
                df_param=df_param,
                label=label,
                translation_dict=translation_dict,
                header_data=header_data,
            )
        label = label.strip()  # type: ignore
    else:
        label = z_param
        if translation_dict and z_param in translation_dict:
            label = translation_dict[z_param]
            # bracket_match = re.match(r".*\[(.*)\]$", translated_name)
            # if bracket_match:
            #     unit = bracket_match.group(1)
            #     translated_name = translated_name.replace(f" [{unit}]", "")
            # else:
            #     unit = None

            # value = df[z_param].max()
            # siprefix, xsi_exponent, exponent_diff = pf.get_SI_prefix((value, value))  # Get the SI prefix for the value
            # label = f"{translated_name} [{siprefix}{unit}]"
        
    # create title for plot
    if title_params:
        if title is None:
            title = ""
        title += "\n"
        for title_param in title_params:
            title = add_param_and_translate(
                param_name=title_param,
                df_curves=df,
                df_param=df_param,
                label=title,
                translation_dict=translation_dict,
                header_data=header_data,
            )
    else:
        title = title

    # extract descriptions
    if translation_dict:
        try: 
            x_discription = translation_dict.get(x_param)
        except Exception as e:
            print(f"Error retrieving x_param description: {e}")
            x_discription = x_param

        try:
            y_discription = translation_dict.get(y_param)
        except Exception as e:
            print(f"Error retrieving y_param description: {e}")
            y_discription = y_param

    else:
        x_discription = x_param
        y_discription = y_param

    if add_title_info:
        try: 
            model = header_data["Model"].strip('.mph').replace("_", " ")
            date = header_data["Date"]
            titleprefix = f"{model} - {date}"
        except KeyError:
            print("Model and/or Date information not found in header_data.")
            titleprefix = None

        if titleprefix is None:
            plot_title = title
        else:
            if title is None:
                plot_title = titleprefix
            else:
                plot_title = f"{title} \n ({titleprefix})"
    else:
        plot_title = title

    if fig is None or ax is None:
        fig, ax = plt.subplots()


    x = df[x_param].to_numpy(dtype=np.float64)
    y = df[y_param].to_numpy(dtype=np.float64)
    z = df[z_param].to_numpy(dtype=np.float64)


    img = ax.scatter(x, y, c=z, cmap=colormap, s=2, marker=MarkerStyle('o'))

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(str(label), color=labelcolor)

    zprefix = None
    if zstyle is not None:
        if zstyle == 'prefix':
            z_abs_max = df[z_param].abs().max()
            zprefix, zsi_exponent, zexponent = pfs.get_SI_prefix((z_abs_max, z_abs_max))
            z_scale_factor = 10 ** (-zsi_exponent)
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * z_scale_factor:g}"))
        else:
            cbar.ax.ticklabel_format(axis='y', style=cast(Any, zstyle), scilimits=(0,0), useMathText=True)

    if zstyle == 'prefix' and label is not None:
        if zprefix is not None:
            label = pfs.set_prefix_in_label(string = label, prefix = zprefix)

    cbar.set_label(str(label), color=labelcolor)


    xprefix = None
    yprefix = None
    if xstyle is not None:
        if xstyle == 'prefix':
            x_abs_max = df[x_param].abs().max()
            xprefix, xsi_exponent, xexponent = pfs.get_SI_prefix((x_abs_max, x_abs_max))
            scale_factor = 10 ** (-xsi_exponent)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * scale_factor:g}"))
        else:
            ax.ticklabel_format(axis='x', style=cast(Any, xstyle), scilimits=(0,0), useMathText=True)
    if ystyle is not None:
        if ystyle == 'prefix':
            y_abs_max = df[y_param].abs().max()
            yprefix, ysi_exponent, yexponent = pfs.get_SI_prefix((y_abs_max, y_abs_max))
            scale_factor = 10 ** (-ysi_exponent)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * scale_factor:g}"))
        else:
            ax.ticklabel_format(axis='y', style=cast(Any, ystyle), scilimits=(0,0), useMathText=True)

    if xstyle == 'prefix' and x_discription is not None:
        if xprefix is not None:
            x_discription = pfs.set_prefix_in_label(string = x_discription, prefix = xprefix)
    if ystyle == 'prefix' and y_discription is not None:
        if yprefix is not None:
            y_discription = pfs.set_prefix_in_label(string = y_discription, prefix = yprefix)

    if x_discription is not None:
        ax.set_xlabel(x_discription)
    if y_discription is not None:
        ax.set_ylabel(y_discription, color=labelcolor)

    if title is not None:
        ax.set_title(title, fontweight='semibold')

    if xscale is not None:
        ax.set_xscale(cast(Any, xscale))
    if yscale is not None:
        ax.set_yscale(cast(Any, yscale))

    if grid:
        ax.grid(True, which='both', linestyle='--')

    
    if save_plot:
        if not output_folder:
            raise ValueError("output_folder must be provided if save_plot is True.")
        modelname = str(header_data.get('Model')).replace(".mph", "")
        
        clean_z_param = re.sub(r"\s*\(.*?\)", "", str(z_param))

        output_path = Path(output_folder) / f"{modelname}-{clean_z_param}_over_{x_param}{y_param}-plane.png"
        fig.savefig(str(output_path), dpi=300)
    
    return fig, ax

##############################################################################
##############################################################################

def polar_plot(
        df: pd.DataFrame,
        angle_column: str,
        r_column: str,
        angle_label: str | None = "Angle [°]",
        title: str = "Polar Plot",
        marker: str = 'o',
        color: str = 'tab:blue',
        labelcolor: str = 'black',
        label: str | None = None,
        fig: matplotlib.figure.Figure | None = None,
        ax: matplotlib.axes.Axes | None = None,
        show_legend: bool = False,
        legend_loc: str = 'best',
        grid: bool = True,
        ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Create a polar plot from the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        angle_column (str): The name of the column in df that contains the angle values (in degrees).
        r_column (str): The name of the column in df that contains the radius values.
        angle_label (str, optional): Label for the angle axis. Defaults to "Angle [°]".
        r_label (str, optional): Label for the radius axis. Defaults to "Radius".
        title (str, optional): Title of the plot. Defaults to "Polar Plot".
        marker (str, optional): Marker style for the data points. Defaults to 'o'.
        color (str, optional): Color for the data points. Defaults to 'tab:blue'.
        labelcolor (str, optional): Color for the axis labels. Defaults to 'black'.
        label (str | None, optional): Label for the data points (for legend). Defaults to None.
        fig (matplotlib.figure.Figure | None, optional): Matplotlib figure to plot on. If None, a new figure is created. Defaults to None.
        ax (matplotlib.axes.Axes | None, optional): Matplotlib axes to plot on. If None, a new axes is created. Defaults to None.
        show_legend (bool, optional): Whether to show the legend. Defaults to False.
        legend_loc (str, optional): Location of the legend. Defaults to 'best'.
        grid (bool, optional): Whether to show grid lines. Defaults to True.
    """
    angle = df[angle_column]
    r = df[r_column]

    if fig is None or ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    if ax is not None:
        ax.scatter(angle, np.abs(r), marker=cast(Any, marker), color=color, label=label)

    if angle_label is not None:
        ax.set_xlabel(angle_label, color=labelcolor)

    if title is not None:
        ax.set_title(title, fontweight='semibold')

    if show_legend and label is not None:
        ax.legend(loc=legend_loc)
    
    if grid:
        ax.grid(True)

    return fig, ax