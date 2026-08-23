# Author: Niko Bleidistel
# last change: 2026-08-19

##############################################################################
# import packages
##############################################################################

import pandas as pd
import importlib

from matplotlib import ticker

# custom packages
import advanced_plotting_functions as apf
_ = importlib.reload(apf)

# for type hints
from typing import Literal
import matplotlib.figure
import matplotlib.axes

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
        ax.scatter(x, y, label=label, color=color, s=markersize)
    else:
        img = ax.scatter(x, y, c=z, cmap='viridis', label=label, s=markersize)
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
