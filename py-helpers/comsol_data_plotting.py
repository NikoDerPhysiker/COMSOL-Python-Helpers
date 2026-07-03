# Author: Niko Bleidistel
# last change: 2026-06-30

##############################################################################
# import packages
##############################################################################

import re
import io
import pandas as pd
from pathlib import Path
from IPython.display import display


import numpy as np
from scipy.interpolate import LinearNDInterpolator


import importlib
# custom packages
import plot_functions as pf
importlib.reload(pf)

# For type hints
import matplotlib.axes
import matplotlib.figure


##############################################################################
##############################################################################
# functions
##############################################################################
##############################################################################

def find_all_files_in_folder(folder_path: str = ".", file_extension: str = "txt"):
    """
    Finds all files with a specific extension in a given folder and its subfolders.


    Args:
        folder_path (str, optional):        The path to the folder where the search should be performed. Relative and absolute paths are supported. (default is "." = current directory).
        file_extension (str, optional):     The file extension to look for (default is "txt").

    Returns:
        (list[str]):    A list of file paths that match the specified extension.

    Raises:
        ValueError:     If folder_path is not a string or if file_extension is not a string.
    """

    if not isinstance(folder_path, (str)):
        raise ValueError("folder_path must be a string or representing the path to the folder.")
    if not isinstance(file_extension, str):
        raise ValueError("file_extension must be a string representing the file extension to look for.")

    # use rglob to find all files with the specified extension in the folder and its subfolders
    file_paths = [str(p) for p in Path(folder_path).rglob(f"*.{file_extension}")]

    return file_paths

##############################################################################
##############################################################################

def read_comsol_export(file_path: str):
    """
    Reads a COMSOL data export file and extracts header information into variables and the numerical data into a pandas DataFrame.
    The function processes the file line by line, distinguishing between header lines (starting with '%') and data lines.

    Args:
        file_path (str): The path to the COMSOL data export file.

    Returns:
        (dict, pandas.DataFrame): A tuple containing the header information as key-value pairs and the numerical data as a pandas DataFrame.

    Raises:
        ValueError: If the file_path is not a string.
    """
    if not isinstance(file_path, (str)):
        raise ValueError("file_path must be a string representing the path to the COMSOL data export file.")

    header_data = {}
    data_lines = []
    column_names = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines() # read all lines at once

    for line in lines:
        line_str = line.strip() # remove leading/trailing whitespace and newline characters

        # regex to split on two or more spaces or on a single space after a closing parenthesis or before 'root.'
        regex = r"\s{2,}|(?<=\))\s+|\s+(?=root\.)"
        
        # extract header information from lines starting with '%'
        if line_str.startswith("%"):
            # delete the leading '%' and strip again to clean up the line
            clean_line = line_str[1:].strip()

            # Split the line into key and value at the first occurrence of ':'
            if ":" in clean_line:
                key, val = clean_line.split(":", 1)
                header_data[key.strip()] = val.strip()

            # if line start with '%' but does not contain ':', it treated as a list of column names
            elif clean_line.startswith("x") or clean_line.startswith("Position"):
                
                # Split the line into column names using regex to handle multiple spaces as delimiters
                column_names = re.split(regex, clean_line)
        
        # if the line does not start with '%', it is treated as a data line and added to the list of data lines 
        else:
            # Only add non-empty lines to the data_lines list
            if line_str:
                data_lines.append(line_str)

    # Create a DataFrame from the data lines using pandas
    data_string = "\n".join(data_lines)                 # Join the list of data lines into a single string for pandas to read
    df = pd.read_csv(
        io.StringIO(data_string),                       # Use StringIO to read the data string as if it were a file
        sep=regex,                                      # Use regex 
        names=column_names,
        engine='python',
    )

    return header_data, df

##############################################################################
##############################################################################
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

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
        
    filtered_df = df.copy()
    
    # 1. Apply initial filtering with a SAFETY MARGIN to speed up the interpolator
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

    if filtered_df.empty:
        print("Warning: No data points remain after applying safety limits.")
        return filtered_df

    if not grid_dict:
        # If no grid specified, clip to strict limits immediately and return
        if limit_dict:
            for column, limits in limit_dict.items():
                if column in filtered_df.columns:
                    lower, upper = sorted(limits)
                    filtered_df = filtered_df[(filtered_df[column] >= lower) & (filtered_df[column] <= upper)]
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

    # 2. FINAL CLIPPING: Crop back to the strict limits requested by the user
    if limit_dict and not final_df.empty:
        for column, limits in limit_dict.items():
            if column in final_df.columns:
                lower, upper = sorted(limits)
                final_df = final_df[
                    (final_df[column] >= lower) & (final_df[column] <= upper)
                ]

    return final_df


if False:
    def filter_and_interpolate_df_old(df: pd.DataFrame,
                                  grid_dict: dict|None = None,
                                  limit_dict: dict|None = None,
                                  exclude_from_interpolation: list|None = None,
                                  ):
        """
        Filters the DataFrame based on specified limits and then performs interpolation to create a new DataFrame with values at exact coordinates.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be filtered and interpolated.
            grid_dict (dict|None): A dictionary specifying the exact coordinates for interpolation.
            limit_dict (dict|None): A dictionary specifying the min and max limits for each column.
            exclude_from_interpolation (list|None): A list of columns to exclude from interpolation.

        Returns:
            pd.DataFrame: The filtered and interpolated DataFrame.
        """
        if exclude_from_interpolation is None:
            exclude_from_interpolation = []

        filtered_df = df.copy()

        # apply min,max limits on dataframe
        if limit_dict:
            for column, limits in limit_dict.items():
                if len(limits) != 2:
                    raise ValueError(f"Limits for column '{column}' must be a list of two values: [lower_limit, upper_limit].")
                if column in filtered_df.columns and len(limits) == 2:
                    lower, upper = sorted(limits)
                    filtered_df = filtered_df[
                    (filtered_df[column] >= lower) & (filtered_df[column] <= upper)
                ]
                else:
                    raise ValueError(f"Column '{column}' not found in DataFrame.")

        # if after limit filtering no data points remain, return the empty DataFrame
        if filtered_df.empty:
            print("Warning: No data points remain after applying the upper and lower limits.")
            return filtered_df

        # if no exact coordinates are specified, return the filtered DataFrame without interpolation
        if not grid_dict:
            print("Info: No exact coordinates specified in grid_dict, therefore no interpolation was performed.")
            return filtered_df

        # interpolation features, the columns with defined exact coordinates.
        interp_features = list(grid_dict.keys())

        # columns which are excluded from interpolation and kept as is.
        excluded_cols = [col for col in exclude_from_interpolation if col in filtered_df.columns]

        # columns which are neither interpolation features nor excluded, these are the target columns for interpolation
        target_columns = [
            col for col in filtered_df.columns 
            if col not in interp_features and col not in excluded_cols
        ]

        if not target_columns:
            raise ValueError("No target columns left for interpolation, after subtracting interpolation grid and excluded columns.")

        # create a meshgrid of the exact coordinates defined in grid_dict, this will be the basis for interpolation
        grid_arrays = [np.array(grid_dict[col]) for col in interp_features]
        meshgrid = np.meshgrid(*grid_arrays, indexing='ij')
        flat_grid = np.vstack([m.flatten() for m in meshgrid]).T

        # Interpolation:
        results = []

        # in case there are exluded columns
        if excluded_cols:
            # sort the filtered DataFrame by the excluded columns to ensure a consistent order for interpolation
            filtered_df = filtered_df.sort_values(by=excluded_cols)

            # for each point in the defined grid 
            for point in flat_grid:
                # we interpolate the interp_features at this point
                # the actual coordinates of the data points in the filtered DataFrame, which will be used for interpolation
                points_coords = filtered_df[interp_features].values

                # creat a new DataFrame for the interpolated point, initially with the same index as the filtered DataFrame to keep track of the original data points
                point_df = pd.DataFrame(index=filtered_df.index)

                # assign the exact coordinates of the current point to the corresponding columns in the new DataFrame
                for idx, col in enumerate(interp_features):
                    point_df[col] = point[idx]

                # add the excluded columns as is to the new DataFrame
                for col in excluded_cols:
                    point_df[col] = filtered_df[col]

                # for each target column, creat an interpolator
                for target_col in target_columns:
                    values = filtered_df[target_col].values
                    coords_with_excluded = filtered_df[interp_features + excluded_cols].values
                    target_interpolator = LinearNDInterpolator(coords_with_excluded, values) #type: ignore

                    # creat query points for interpolation, which consist of the exact coordinates
                    query_points = np.hstack([
                        np.tile(point, (len(filtered_df), 1)), 
                        filtered_df[excluded_cols].values
                    ])

                    point_df[target_col] = target_interpolator(query_points)

                # get rid of any rows with NaN
                point_df = point_df.dropna(subset=target_columns)
                results.append(point_df)

            # concatenate all the interpolated points into a single DataFrame
            if not results:
                return pd.DataFrame(columns=df.columns)
            return pd.concat(results, ignore_index=True)

        # if there are no excluded columns, we can directly interpolate 
        else:
            result_dict = {col: flat_grid[:, i] for i, col in enumerate(interp_features)}
            result_df = pd.DataFrame(result_dict)
            points = filtered_df[interp_features].values

            for target_col in target_columns:
                values = filtered_df[target_col].values
                interpolator = LinearNDInterpolator(points, values) #type: ignore
                result_df[target_col] = interpolator(flat_grid)

            return result_df
    
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

    # mask before interpolation to reduce computational load
    keep_columns = ['x', 'y', 'z'] + x_params + y_params
    actual_columns_2keep = list(dict.fromkeys(keep_columns))                                     # ensure that the list of columns to keep does not contain duplicates
    df_interpol = df[actual_columns_2keep]                                                       # filter the DataFrame to keep only the relevant columns
    df_interpol.head()
    # filtering and interpolation of the DataFrame with the new function signature
    df_curves = filter_and_interpolate_df(
        df=df_interpol, 
        grid_dict=grid_dict, 
        limit_dict=limit_dict,
        )
    
    return df_curves

##############################################################################

def plot_comsol_data(df: pd.DataFrame,
                     header_data: dict,
                     x_column: str,
                     y_column: str,
                     x_label: str|None = None,
                     y_label: str|None = None,
                     title: str|None = None,
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
                     xstyle: str|None = "sci",
                     ystyle: str|None = "sci",
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

    fig, ax = pf.u_plot_scatter_with_error_bars(
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
            label: str | None,
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
            print(f"Warning: '{param_name}' not found in df_curves or df_param.")
            return label  # return the original label if the parameter is not found
        
        if translation_dict and param_name in translation_dict:
            translated_name = translation_dict[param_name]
        else:
            translated_name = param_name
        
        if label is None:
            label = ""
        else:
            label += "\n"

        label += f"{translated_name} = {value:.2e}"

        if header_data and param_name in ['x', 'y', 'z']:
            label += f" [{header_data.get('Length unit')}]"
        
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
        xstyle = 'sci',
        ystyle = 'sci',
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