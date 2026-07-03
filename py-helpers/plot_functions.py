# Author: Niko Bleidistel
# last change: 2026-06-15

# Description:
# This module provides functions for plotting data with error bars and optional fitting, 
# specifically designed to work with data that may include uncertainties (using the 'uncertainties' package). 
# The main function, `u_plot_scatter_with_error_bars`, allows for flexible plotting of scatter points with error bars, 
# while the extended function `u_plot_scatter_with_error_bars_and_fit` adds the capability to fit a model to the data and plot the fit. 
# Additionally, the `u_twin_plot` function enables plotting two datasets with different y-axes on the same x-axis using twin axes. 
# The functions are designed to be user-friendly and customizable, with options for labels, colors, legends, scales, and more.


##############################################################################
# Import necessary libraries and define type hints
##############################################################################

import uncertainties as unc
import numpy as np
import matplotlib.pyplot as plt

# For type hints
from typing import Any, cast

import matplotlib.axes
import matplotlib.figure

from uncertainties.core import Variable, AffineScalarFunc
from typing import Union
UncFloat = Union[Variable, AffineScalarFunc]

##############################################################################
# base function
##############################################################################

def u_plot_scatter_with_error_bars(x: list[UncFloat] | list[float], 
                                   y: list[UncFloat] | list[float], 
                                   x_label: str | None = None, 
                                   y_label: str | None = None, 
                                   title: str | None = None,
                                   label: str | None = None, 
                                   marker: str = 'o', 
                                   color: str = 'tab:blue', 
                                   labelcolor: str = 'black', 
                                   fig: matplotlib.figure.Figure | None = None, 
                                   ax: matplotlib.axes.Axes | None = None,
                                   show_legend: bool = True,
                                   legend_loc: str = 'best',
                                   xscale: str | None = None,
                                   yscale: str | None = None,
                                   xstyle: str | None = None,
                                   ystyle: str | None = None,
                                   grid: bool = True,
                                   plot_error: bool = True,
                                   fix_title_spacing: bool = False,
                                   ):
    """
    Plots a scatter plot with error bars for x and y.
    
    Parameters:
        x (list[unc.ufloat] | list[float]):  list of x data (can be list of ufloats or regular floats)
        y (list[unc.ufloat] | list[float]):  list of y data (can be list of ufloats or regular floats)
        x_label (str, optional):             label for x axis
        y_label (str, optional):             label for y axis
        title (str, optional):               title of the plot
        label (str, optional):               label for the data points (for legend)
        marker (str, optional):              marker style for the data points
        color (str, optional):               color for the data points and error bars
        labelcolor (str, optional):          color for the axis labels
        fig (plt.Figure | None, optional):   matplotlib figure to plot on (if None, a new figure is created)
        ax (plt.Axes | None, optional):      matplotlib axis to plot on (if None, a new axis is created)
        show_legend (bool, optional):        whether to show the legend
        legend_loc (str, optional):          location of the legend
        xscale (str | None, optional):       scale for x axis (e.g. 'linear', 'log')
        yscale (str | None, optional):       scale for y axis (e.g. 'linear', 'log')
        xstyle (str | None, optional):       style for x axis ticks (e.g. 'plain', 'sci')
        ystyle (str | None, optional):       style for y axis ticks (e.g. 'plain', 'sci')
        grid (bool, optional):               whether to show grid
        plot_error (bool, optional):         whether to plot error bars (if False, errors are ignored and only scatter points are plotted)

    Returns:
        (plt.Figure, plt.Axes): A tuple containing the matplotlib figure and axis objects for the plot. 
            - If fig and ax were provided as input, they will be returned unchanged.
            - If fig and ax were None, a new figure and axis will be created and returned.


    Raises:
        ValueError: If any of the input parameters are of incorrect type or if x and y data are not provided or have different lengths.
    """
    if not isinstance(x, list):
        raise ValueError("x must be a list of data points (can be list of ufloats or regular floats).")
    if not isinstance(y, list):
        raise ValueError("y must be a list of data points (can be list of ufloats or regular floats).")
    if fig is not None and not isinstance(fig, matplotlib.figure.Figure):
        raise ValueError("fig must be a matplotlib Figure object or None.")
    if ax is not None and not isinstance(ax, matplotlib.axes.Axes):
        raise ValueError("ax must be a matplotlib Axes object or None.")
    if not isinstance(show_legend, bool):
        raise ValueError("show_legend must be a boolean indicating whether to show the legend.")
    if not isinstance(grid, bool):
        raise ValueError("grid must be a boolean indicating whether to show grid.")
    if not isinstance(plot_error, bool):
        raise ValueError("plot_error must be a boolean indicating whether to plot error bars.")
    
    if x is None or y is None:
        raise ValueError("x and y data must be provided")
    if x is not None and y is not None and len(x) != len(y):
        raise ValueError("x and y data must have the same length")
    
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    if plot_error:
        # Extract nominal values: use .n if it exists, otherwise cast to float
        x_nom = [cast(Any, xi).n if hasattr(xi, 'n') else float(cast(Any, xi)) for xi in x]
        # Extract standard deviations: use .s if it exists, otherwise default to 0.0
        x_err = np.array([cast(Any, xi).s if hasattr(xi, 's') else 0.0 for xi in x])
        
        # Print warning if at least one element had no uncertainty attributes
        if any(not hasattr(xi, 's') or not hasattr(xi, 'n') for xi in x):
            print("Some or all x data converted to ufloat with zero uncertainty")

        # Repeat the exact same logic for y data
        y_nom = [cast(Any, yi).n if hasattr(yi, 'n') else float(cast(Any, yi)) for yi in y]
        y_err = np.array([cast(Any, yi).s if hasattr(yi, 's') else 0.0 for yi in y])
        
        if any(not hasattr(yi, 's') or not hasattr(yi, 'n') for yi in y):
            print("Some or all y data converted to ufloat with zero uncertainty")
        
        # Only provide errorbars to matplotlib if there are non-zero uncertainties.
        xerr_arg = x_err if np.any(~np.isclose(x_err, 0.0)) else None
        yerr_arg = y_err if np.any(~np.isclose(y_err, 0.0)) else None

        # Plot the data with error bars. If xerr_arg or yerr_arg is None, matplotlib will simply ignore the error bars for that axis.
        ax.errorbar(x_nom, y_nom, xerr=cast(Any, xerr_arg), yerr=cast(Any, yerr_arg), label=label,  fmt=marker, color=color, ecolor=color, capsize=5)

    else:
        # If errors are not plotted, only extract the nominal values
        x_nom = [cast(Any, xi).n if hasattr(xi, 'n') else float(cast(Any, xi)) for xi in x]
        y_nom = [cast(Any, yi).n if hasattr(yi, 'n') else float(cast(Any, yi)) for yi in y]

        # plot the data without error bars
        ax.scatter(x_nom, y_nom, label=label, color=color, marker=cast(Any, marker), s=8)


    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label, color=labelcolor)

    if title is not None:
        if fix_title_spacing:
            ax.set_title(
                title, 
                fontweight='semibold', 
                fontfamily='monospace',
                loc='center',
                multialignment='left'
            )
        else:
            ax.set_title(title, fontweight='semibold')

    if label is not None and show_legend:
        ax.legend(loc=legend_loc)
    if xscale is not None:
        ax.set_xscale(cast(Any, xscale))
    if yscale is not None:
        ax.set_yscale(cast(Any, yscale))
    if xstyle is not None:
        ax.ticklabel_format(axis='x', style=cast(Any, xstyle), scilimits=(0,0), useMathText=True)
    if ystyle is not None:
        ax.ticklabel_format(axis='y', style=cast(Any, ystyle), scilimits=(0,0), useMathText=True)
    if grid:
        ax.grid(True, which='both', linestyle='--')
    

    return fig, ax

##############################################################################
# extended function with fit
##############################################################################

def u_plot_scatter_with_error_bars_and_fit(x: list[UncFloat] | list[float],
                                           y: list[UncFloat] | list[float],
                                           x_label: str | None=None, 
                                           y_label: str | None=None, 
                                           title: str | None=None,
                                           label: str | None=None, 
                                           marker: str='o', 
                                           color: str='tab:blue', 
                                           labelcolor: str='black', 
                                           fig: matplotlib.figure.Figure | None=None, 
                                           ax: matplotlib.axes.Axes | None=None,
                                           show_legend: bool=True,
                                           legend_loc: str='best',
                                           xscale: str | None=None,
                                           yscale: str | None=None,
                                           grid: bool=True,
                                           plot_error: bool=False,
                                           fit_model = None,
                                           fit_params: tuple | None=None,
                                           fit_color: str='tab:red',
                                           fit_label: str | None=None,
                                           fit_uncertainties: bool=False,
                                           ):
    """
    Plots a scatter plot with error bars for x and y, and optionally fits a model to the data and plots the fit.
    
    Parameters:
    :x (list[unc.ufloat] | list[float]):        list of x data (can be list of ufloats or regular floats)
    :y (list[unc.ufloat] | list[float]):        list of y data (can be list of ufloats or regular floats)
    :x_label (str | None, optional):            label for x axis
    :y_label (str | None, optional):            label for y axis
    :title (str | None, optional):              title of the plot
    :label (str | None, optional):              label for the data points (for legend)
    :marker (str, optional):                    marker style for the data points
    :color (str, optional):                     color for the data points and error bars
    :labelcolor (str, optional):                color for the axis labels
    :fig (plt.Figure | None, optional):         matplotlib figure to plot on (if None, a new figure is created)
    :ax (plt.Axes | None, optional):            matplotlib axis to plot on (if None, a new axis is created)
    :show_legend (bool, optional):              whether to show the legend
    :legend_loc (str, optional):                location of the legend
    :xscale (str | None, optional):             scale for x axis (e.g. 'linear', 'log')
    :yscale (str | None, optional):             scale for y axis (e.g. 'linear', 'log')
    :grid (bool, optional):                     whether to show grid
    :plot_error (bool, optional):               whether to plot error bars (if False, errors are ignored and only scatter points are plotted)
    :fit_model (callable | None, optional):     function to fit to the data (should take x and fit parameters as input)
    :fit_params (tuple | None, optional):       parameters for the fit model (should be a tuple of parameters to pass to fit_model)
    :fit_color (str, optional):                 color for the fit line
    :fit_label (str | None, optional):          label for the fit line (for legend)
    :fit_uncertainties (bool, optional):        whether to plot the uncertainties of the fit parameters as a shaded area around the fit line (only works if fit_params are ufloats with nonzero uncertainties)

    Returns:
    :fig: the matplotlib figure object containing the plot
    :ax: the matplotlib axis object containing the plot
    
    Raises:
    :ValueError: If any of the input parameters are of incorrect type or if x and y data are not provided or have different lengths, or if fit_model is provided without fit_params, or if fit_params are not a tuple.
    """

    fig, ax = u_plot_scatter_with_error_bars(x, y, x_label, y_label, title, label, marker, color, labelcolor, 
                                             fig, ax, show_legend=False, legend_loc=legend_loc, xscale=xscale, 
                                             yscale=yscale, grid=grid, plot_error=plot_error
                                             )

    if fit_model is not None and fit_params is not None:
        if len(x) > 0 and hasattr(x[0], 'n'):
            x_nom = [cast(Any, xi).n if hasattr(xi, 'n') else float(cast(Any, xi)) for xi in x]
        else:
            x_nom = np.asarray(x, dtype=float)
        x_fit = np.linspace(np.min(x_nom), np.max(x_nom), 100)
        y_fit = fit_model(x_fit, *fit_params)
        ax.plot(x_fit, y_fit, color=fit_color, label=fit_label)

        if fit_uncertainties and any(hasattr(p, 's') and p.s > 0 for p in fit_params):
            y_fit_upper = fit_model(x_fit, *(p.n + p.s if hasattr(p, 's') else p for p in fit_params))
            y_fit_lower = fit_model(x_fit, *(p.n - p.s if hasattr(p, 's') else p for p in fit_params))
            ax.fill_between(x_fit, y_fit_lower, y_fit_upper, color=fit_color, alpha=0.3)

    if label is not None and show_legend:
        ax.legend(loc=legend_loc)

    return fig, ax

##############################################################################
# extended function with twin axes
##############################################################################

def u_twin_plot(x, 
              y1, y2, 
              x_label, 
              y1_label=None, y2_label=None, 
              title=None,
              label1=None, label2=None,
              color1 = 'tab:blue', color2 = 'tab:red',
              labelcolor1 = 'tab:blue', labelcolor2 = 'tab:red',
              fig=None,
              ax1=None, ax2=None,
              show_legend=True,
              legend_loc='best',
              grid=False,
              plot_error = False,
              ):
    """
    Plots two sets of data (x, y1) and (x, y2) on the same x axis but with different y axes (twin axes).
    If fig, ax1 and ax2 are not provided, a new figure and twin axes will be created. If any of fig, ax1 or ax2 is provided, all three must be provided.
    
    Parameters:
    :x: list of x data (can be list of ufloats or regular floats
    :y1: list of y data for the first dataset (can be list of ufloats or regular floats)
    :y2: list of y data for the second dataset (can be list of ufloats or regular floats)
    :x_label: label for x axis
    :y1_label: label for y axis of the first dataset
    :y2_label: label for y axis of the second dataset
    :title: title of the plot
    :label1: label for the first dataset (for legend)
    :label2: label for the second dataset (for legend)
    :color1: color for the first dataset
    :color2: color for the second dataset
    :labelcolor1: color for the y axis label of the first dataset
    :labelcolor2: color for the y axis label of the second dataset
    :fig: matplotlib figure to plot on (if None, a new figure is created)
    :ax1: matplotlib axis for the first dataset (if None, a new axis is created)
    :ax2: matplotlib axis for the second dataset (if None, a new twin axis is created)
    :show_legend: whether to show the legend
    :legend_loc: location of the legend
    :grid: whether to show grid
    :plot_error: whether to plot error bars (if True, y1 and y2 should be lists of ufloats with nonzero uncertainties, otherwise errors will be ignored)

    Returns:
    :fig: the matplotlib figure object containing the plot
    :ax1: the matplotlib axis object for the first dataset
    :ax2: the matplotlib axis object for the second dataset

    Raises:
    :ValueError: If any of the input parameters are of incorrect type or if x, y1 and y2 data are not provided or have different lengths, or if fig, ax1 and ax2 are not all provided when any of them is provided.
    """
    
    if fig is not None or ax1 is not None or ax2 is not None:
        if fig is None or ax1 is None or ax2 is None:
            raise ValueError("fig, ax1 and ax2 must all be provided if any of them is provided")
    if fig is None and ax1 is None and ax2 is None:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

    u_plot_scatter_with_error_bars(x, 
                                   y1,
                                   x_label=x_label,
                                   y_label=y1_label,
                                   title=title,
                                   label=label1,
                                   color=color1,
                                   labelcolor=labelcolor1,
                                   fig=fig,
                                   ax=ax1,
                                   show_legend=False,
                                   grid=grid,
                                   plot_error = plot_error,
                                   )
    u_plot_scatter_with_error_bars(x, 
                                   y2,
                                   x_label=x_label,
                                   y_label=y2_label,
                                   title=title,
                                   label=label2,
                                   color=color2,
                                   labelcolor=labelcolor2,
                                   fig=fig,
                                   ax=ax2,
                                   show_legend=False,
                                   grid=grid,
                                   plot_error = plot_error,
                                   )
    if show_legend:
        assert ax1 is not None
        assert ax2 is not None
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc=legend_loc)
    return fig, ax1, ax2

##############################################################################
# end of file
##############################################################################