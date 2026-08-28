# Author: Niko Bleidistel
# last change: 2026-08-19

##############################################################################
# import packages
##############################################################################

import math
import re

import pandas as pd

# for plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for type hinting
from typing import Literal, cast

import matplotlib.axes
import matplotlib.figure
from matplotlib.markers import MarkerStyle
from matplotlib.backends.backend_agg import FigureCanvasAgg

from pathlib import Path


##############################################################################
##############################################################################
# constants
##############################################################################
##############################################################################

FONTSIZE = 12           # in pt, of the latex document
TEXTWIDTH = 455.24411   # in pt, of the latex document
TEXTHEIGHT = 635.25946  # in pt, of the latex document

MARKERSIZE = 5         # of the plot markers
TITLECOLOR = (0 / 255, 51 / 255, 102 / 255) # color of the title = FAU-Blau

##############################################################################
# style settings
##############################################################################

# import scienceplots         # for the 'science' and 'nature' styles
# _ = scienceplots.__file__   # to avoid linter warning about unused import

# plt.style.use([
#     'science',
#     'nature'
#     ])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Latin Modern Roman"], 
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{lmodern} \usepackage{upgreek} \usepackage{xfrac} \usepackage{bm} \usepackage{csquotes}",
})

plt.rcParams.update({
    "font.size": FONTSIZE,                
    "axes.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "xtick.labelsize": 0.8 * FONTSIZE,
    "ytick.labelsize": 0.8 * FONTSIZE,
})

##############################################################################
##############################################################################
# string formaters
##############################################################################
##############################################################################

def get_SI_prefix(limits: tuple[float, float], use_u_as_micro: bool = False, bm: bool = False) -> tuple[str | None, int, int]:
        """
        Returns the SI prefix for a given numeric value.
        
        Args:
            limits (tuple[float, float]): A tuple containing the minimum and maximum values.
            use_u_as_micro (bool): Whether to use 'u' as the symbol for micro or 'μ'.
            bm (bool): Whether to use bold math formatting for the SI prefix.
        
        Returns:
            tuple[str|None, int, int]: A tuple containing the SI prefix, the SI exponent, and the exponent difference to the magnitude of the input values.
        """
        # get the maximum absolute value from the limits
        xmax = max(abs(x) for x in limits)

        # get magnitude of the maximum value
        x_base_exponent = math.floor(math.log10(xmax)) if xmax > 0 else 0
        
        # Round down to the nearest multiple of 3 for SI prefix
        xsi_exponent = (x_base_exponent // 3) * 3 
        exponent_diff = x_base_exponent - xsi_exponent

        # Define the mapping of SI prefixes to their corresponding exponents
        PREFIX_TO_EXPONENT = {
            # Large values (positive exponents)
            r"\mathrm{Q}": 30,   # Quetta
            r"\mathrm{R}": 27,   # Ronna
            r"\mathrm{Y}": 24,   # Yotta
            r"\mathrm{Z}": 21,   # Zetta
            r"\mathrm{E}": 18,   # Exa
            r"\mathrm{P}": 15,   # Peta
            r"\mathrm{T}": 12,   # Tera
            r"\mathrm{G}": 9,    # Giga
            r"\mathrm{M}": 6,    # Mega
            r"\mathrm{k}": 3,    # Kilo
            r"\mathrm{h}": 2,    # Hecto
            r"\mathrm{da}": 1,   # Deca

            # zero exponent (no prefix)
            "": 0,     # No prefix
            
            # Small values (negative exponents)
            r"\mathrm{d}": -1,   # Deci
            r"\mathrm{c}": -2,   # Centi
            r"\mathrm{m}": -3,   # Milli
            r"\mathrm{u}": -6,   # Micro (often written as µ)
            r"\mathrm{n}": -9,   # Nano
            r"\mathrm{p}": -12,  # Pico
            r"\mathrm{f}": -15,  # Femto
            r"\mathrm{a}": -18,  # Atto
            r"\mathrm{z}": -21,  # Zepto
            r"\mathrm{y}": -24,  # Yocto
            r"\mathrm{r}": -27,  # Ronto
            r"\mathrm{q}": -30,   # Quekto
        }
        if not use_u_as_micro:
            # PREFIX_TO_EXPONENT["μ"] = -6   # Use 'μ' for micro instead of 'u'
            PREFIX_TO_EXPONENT[r"\mathrm{\upmu}"] = -6   # Use 'μ' for micro instead of 'u'
            del PREFIX_TO_EXPONENT[r"\mathrm{u}"]    # Remove 'u' from the dictionary

        # Returns the key, or None if the value doesn't exist
        siprefix = next((k for k, v in PREFIX_TO_EXPONENT.items() if v == xsi_exponent), None)
        if bm and siprefix is not None:
            siprefix = r"\bm{" + siprefix.strip("$") + "}"  # Add bold math formatting to the SI prefix
        return siprefix, xsi_exponent, exponent_diff

##############################################################################

def prefixes_notation(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, axis: Literal['x', 'y']):
    """
    Adjusts the axis labels of a matplotlib plot to use SI prefixes based on the data limits.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        axis (Literal['x', 'y']): The axis to adjust ('x' or 'y').
    
    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, str|None, int]: A tuple containing the updated figure and axes objects, the SI prefix used, and the SI exponent.
    """
    fig.canvas.draw() 
    if axis == 'x':
        xlimits = ax.get_xlim()
        siprefix, xsi_exponent, exponent_diff = get_SI_prefix(xlimits)
        scale_factor = 10 ** (-xsi_exponent)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * scale_factor:g}"))
    elif axis == 'y':
        ylimits = ax.get_ylim()
        siprefix, xsi_exponent, exponent_diff = get_SI_prefix(ylimits)
        scale_factor = 10 ** (-xsi_exponent)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y * scale_factor:g}"))
    return fig, ax, siprefix, xsi_exponent

##############################################################################

def set_prefix_in_label(string: str, prefix: str):
        """
        Inserts the SI prefix into the string behind the first occurrence of an opening rectangular bracket.
        If no rectangular bracket is found, it searches an opening round bracket.
        If no bracket is found, the prefix is appended to the end of the string.

        Args:
            string (str): The input string where the prefix will be inserted.
            prefix (str): The SI prefix to insert.
        
        Returns:
            str: The modified string with the SI prefix inserted.
        """
        # if re.search(r"$\[\w", string):
            # return re.sub(r"\[(?=\w)", f"[{prefix}", string)
        if r"$[" in string:
            return string.replace("$[", f"$[{prefix}")
        # elif re.search(r"$\(\w", string):
            # return re.sub(r"\((?=\w)", f"({prefix}", string)
        elif r"$(" in string:
            return string.replace("$(", f"$({prefix}")
        else:
            return f"{string} [{prefix}1]"

##############################################################################

def set_prefix_in_number_unit_string(string: str, bm: bool = False) -> str:
    """
    Searches the pattern '= number [' in the string. Then it extracts the number and uses  
    Args:
        string (str): The input string containing the number and unit.
        bm (bool): Whether to use bold math formatting for the SI prefix. Default is False.
    """
    pattern = r"(=\s*)(-?\d+\.?\d*(?:[eE][-+]?\d+)?)(?=\s*\[)"
    # matches the pattern '= number [' in the string, where 'number' can be an integer or a float (including scientific notation).
    # (=\s*) matches the equal sign followed by optional whitespace and captures it as group 1.
    # (-?\d+\.?\d*(?:[eE][-+]?\d+)?) matches the number (including optional scientific notation) and captures it as group 2.
    # # -?\d+\.?\d* matches an optional negative sign, followed by one or more digits, an optional decimal point, and zero or more digits.
    # # (?:[eE][-+]?\d+)? matches an optional scientific notation part, which consists of 'e' or 'E', an optional sign, and one or more digits.
    # (?=\s*\[) is a positive lookahead that ensures the number is followed by optional whitespace and an opening square bracket, but does not include it in the match.
    
    match = re.search(pattern, string)
    if match:
        prefix_part = match.group(1) # the '=' and any optional whitespace before the number
        number_str = match.group(2)  # the number in float format or scientific notation
        number = float(number_str) # convert matched number from string to float
        
        # get the prefix
        siprefix, xsi_exponent, exponent_diff = get_SI_prefix((number, number), bm=bm)
        
        if siprefix is not None:
            # scale the number according to the SI prefix exponent
            scaled_number = number * (10 ** (-xsi_exponent))
            scaled_str = f"{scaled_number:g}"
            
            # replace the matched pattern in the string with the scaled number
            string = re.sub(pattern, rf"{prefix_part}{scaled_str}", string, count=1)
            
            # insert the SI prefix into the string behind opening rectangular bracket 
            string = string.replace("[", f"[{siprefix}")
            
    return string

##############################################################################
##############################################################################

def get_formula_unit(string: str) -> tuple[str, str]:
    """
    Searches for a formula and a unit in the given string. 
    The formula is expected to be enclosed in dollar signs '$', 
    and the unit is expected to be enclosed in either square brackets '[]' or round brackets '()'.

    Args:
        string (str): The input string containing the formula and unit.

    Returns:
        tuple[str, str]: A tuple containing the extracted formula and unit. 
                         If no unit is found, returns empty strings for unit.
                         If no formula is found, returns the entire string but the unit as the formula.
    """
    value = string.strip()

    # 1. Search for square brackets at the end of the string, e.g., [m]
    unit_match = re.search(r"(\[.*?\])$", value)
    
    # 2. Fallback: Search for round brackets at the end, e.g., (m)
    if not unit_match:
        unit_match = re.search(r"(\(.*?\))$", value)
        
    if unit_match:
        full_unit_string = unit_match.group(1)
        # Remove the outer brackets (works for both [ ] and ( ))
        unit = full_unit_string[1:-1].strip()
    else:
        full_unit_string = ""
        unit = ""
        
    # Get formula from the string, e.g., "$x$" or "$|\vec{B}|$"
    math_blocks = re.findall(r"(\$.*?\$)", value)
    if math_blocks:
        short = math_blocks[-1]
    else:
        # Fallback: if no math block is found, use the whole string without the unit
        short = value
        if full_unit_string:
            short = short.replace(full_unit_string, "")
        short = short.strip("[]()").strip()

    return short, unit

##############################################################################

def translate_and_prefix_label(label: str, translation_dict: dict[str, str] | None = None, bm: bool = False) -> str:
    """
    Translates the variable in the label using the provided translation dictionary and applies SI prefix notation.

    The label is expected to be in the format "variable = value [unit]".
    If a translation dictionary is provided and the variable is found in it, the variable will be replaced with its corresponding translation.
    If there is a unit in the (now translated) variable -- marked by either round or square brackets -- and no unit behind the value, the unit will be moved behind the value.
    The function also applies SI prefix notation to the value, meaning that the magnitude of the value will be adjusted to a suitable SI prefix, and the prefix will be added to the unit.

    Args:
        label (str):    The input label string, expected to be in the format "variable = value [unit]".
        translation_dict (dict[str, str] | None): A dictionary for translating variable names. Default is None.
        bm (bool): Whether to use bold math notation for the label. Default is False.
    Returns:
        str: The translated and prefixed label.
    """

    # split the label into variable and rest
    if '=' not in label:
        if translation_dict is not None and label in translation_dict:
            label = translation_dict[label]
        return label
    else:
        label_parts = label.strip().split('=')  # split the label at '=' 
        variable = label_parts[0].strip()       # get the variable name
        rest = '='.join(label_parts[1:]) if len(label_parts) > 1 else ''  # get the rest of the label after '='

    if translation_dict is not None and variable in translation_dict:
        variable = translation_dict[variable]
   
    # get the short name and unit for the variable
    short, unit = get_formula_unit(variable)

    # if the rest of the label does not contain a unit and a unit is available, append the unit to the rest of the label
    _ , unit_rest = get_formula_unit(rest)
    if unit_rest == '' and unit != '':
        rest = f"{rest} [{unit}]"

    short_label = f"{short} = {rest}"
    return set_prefix_in_number_unit_string(short_label, bm=bm)


##############################################################################
##############################################################################
# plotting helper
##############################################################################
##############################################################################

def calc_figure_size(fraction: float = 1.0, width_pt: float = TEXTWIDTH, subplots: tuple = (1, 1)) -> tuple[float, float]:
    """
    Calculates the exact figure size in inches based on LaTeX textwidth.

    Args:
        fraction (float): Fraction of the LaTeX textwidth to use for the figure width. Default is 1.0 (full width).
        width_pt (float): Width of the LaTeX textwidth in points. Default is TEXTWIDTH.
        subplots (tuple): A tuple specifying the number of rows and columns of subplots. Default is (1, 1).
    """
    # 1. pt to inch conversion factor
    inches_per_pt = 1 / 72.27
    
    # 2. figure width in inches
    fig_width_in = width_pt * inches_per_pt * fraction

    # 3. golden ratio for the height (aesthetically pleasing standard ratio)
    golden_ratio = (5**0.5 - 1) / 2
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in

##############################################################################
##############################################################################

def get_fig_ax(fraction_textwidth: float = 1.0, subplots: tuple = (1, 1)) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Creates a matplotlib figure and axes with a specified width fraction of the LaTeX textwidth.
    (The textwidth of the LaTeX document is defined via the constant 'TEXTWIDTH' in this file.)

    Args:
        fraction_textwidth (float): Fraction of the LaTeX textwidth to use for the figure width. Default is 1.0 (full width).
        subplots (tuple): A tuple specifying the number of rows and columns of subplots. Default is (1, 1).

    Returns:
        tuple (Figure, Axes): A tuple containing the created matplotlib figure and axes objects.
    """
    fig_width_in, fig_height_in = calc_figure_size(fraction=fraction_textwidth, width_pt=TEXTWIDTH, subplots=subplots)
    fig, ax = plt.subplots(*subplots, figsize=(fig_width_in, fig_height_in))
    return fig, ax

##############################################################################

def get_fig_ax_page(
        fraction_textwidth: float = 1.0,
        fraction_textheight: float = 1.0,
        subplots: tuple = (1, 1)
        ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Creates a matplotlib figure and axes with specified width and height fractions of the LaTeX textwidth and textheight.
    
    Args:
        fraction_textwidth (float): Fraction of the LaTeX textwidth to use for the figure width. Default is 1.0 (full width).
        fraction_textheight (float): Fraction of the LaTeX textheight to use for the figure height. Default is 1.0 (full height).
        subplots (tuple): A tuple specifying the number of rows and columns of subplots. Default is (1, 1).

    Returns:
        tuple (Figure, Axes): A tuple containing the created matplotlib figure and axes objects.
    """
    
    fig_width_in, _ = calc_figure_size(fraction=fraction_textwidth, width_pt=TEXTWIDTH, subplots=(1, 1))
    fig_height_in, _ = calc_figure_size(fraction=fraction_textheight, width_pt=TEXTHEIGHT, subplots=(1, 1))
    fig, ax = plt.subplots(*subplots, figsize=(fig_width_in, fig_height_in))
    return fig, ax

##############################################################################
##############################################################################

def plot_background(
        # Figure and Axes
        fig: matplotlib.figure.Figure | None = None, 
        ax: matplotlib.axes.Axes | None = None,

        # labels and title
        title: str | None = None,
        x_label: str | None = None, 
        y_label: str | None = None,
        xstyle: Literal['sci', 'scientific', 'plain', 'prefix'] | None = 'prefix',
        ystyle: Literal['sci', 'scientific', 'plain', 'prefix'] | None = 'prefix',
        ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Sets up the background of a matplotlib plot, including figure and axes, labels, title, and grid.

    Args:
        fig (matplotlib.figure.Figure | None): The matplotlib figure object.
        ax (matplotlib.axes.Axes | None): The matplotlib axes object.

        title (str | None): The title of the plot. Default is None.
        x_label (str | None): The label for the x-axis. Default is None.
        y_label (str | None): The label for the y-axis. Default is None.
        xstyle (Literal['sci', 'scientific', 'plain', 'prefix'] | None): The style for the x-axis tick labels. Default is 'prefix'.
        ystyle (Literal['sci', 'scientific', 'plain', 'prefix'] | None): The style for the y-axis tick labels. Default is 'prefix'.

    Returns:
        tuple (Figure, Axes): A tuple containing the updated matplotlib figure and axes objects.
    """


    # get the figure and axis if not provided
    if fig is None or ax is None:
        raise ValueError("Figure and axis must be provided.")

    # apply 'prefix' style to axis
    xprefix = None
    yprefix = None
    if xstyle is not None:
        if xstyle == 'prefix':
            fig, ax, xprefix, xexponent = prefixes_notation(fig, ax, 'x')
        else:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(axis='x', style= xstyle, scilimits=(0,0), useMathText=True)
    if ystyle is not None:
        if ystyle == 'prefix':
            fig, ax, yprefix, yexponent = prefixes_notation(fig, ax, 'y')
        else:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(axis='y', style=ystyle, scilimits=(0,0), useMathText=True)

    # get current label if not provided
    if x_label is None:
        x_label = ax.get_xlabel()
    if y_label is None:
        y_label = ax.get_ylabel()

    # apply 'prefix' style to labels
    if xstyle == 'prefix' and x_label is not None:
        if xprefix is not None:
            x_label = set_prefix_in_label(string = x_label, prefix = xprefix)
    if ystyle == 'prefix' and y_label is not None:
        if yprefix is not None:
            y_label = set_prefix_in_label(string = y_label, prefix = yprefix)

    # set axis labels
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    # show title, legend and grid
    if title is not None:
        if "(" in title:
            main_title = title.split("(")[0].strip()
            subtitle = "(" + title.split("(")[1].strip()
        else:
            main_title = title
            subtitle = ""

        ax.set_title(rf"\textbf{{{main_title}}}"+"\n\n"+rf"{{{subtitle}}}", color=TITLECOLOR, pad=20)

    ax.set_axisbelow(True)  # Ensure grid is below other plot elements
    ax.grid(True, which='both', linestyle='--', zorder=0)
    
    return fig, ax

##############################################################################
##############################################################################

def dynamic_legend(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    fraction: float = 1.0,
    width_tolerance: float = 1.10,
):
    """
    Creates a dynamic legend for a matplotlib plot, adjusting the number of columns based on the plot width and number of legend entries (even or odd).
    It also sorts the legend entries: If scatter and line plots alternate, placing scatter entries on the left and line entries on the right.

    Args:
        fig (matplotlib.figure.Figure):     The matplotlib figure object.
        ax (matplotlib.axes.Axes):          The matplotlib axes object.
        fraction (float):                   Fraction of the page width to use for the legend. Default is 1.0 (full width).
        width_tolerance (float):            Tolerance factor for the maximum allowed legend width relative to the figure width. Default is 1.10.
    
    Returns:
        tuple (Figure, Axes): A tuple containing the updated matplotlib figure and axes objects.
    """
    # 1. Handles und Labels extrahieren
    handles, labels = ax.get_legend_handles_labels()
    n_entries = len(handles)

    if n_entries == 0:
        return fig, ax

    # 2. Regel 3: Sortierung nach Typ (Scatter links, Lines rechts), falls abwechselnd
    is_scatter = [isinstance(h, PathCollection) for h in handles]

    is_alternating = False
    if n_entries >= 2:
        is_alternating = all(is_scatter[i] != is_scatter[i + 1] for i in range(n_entries - 1))

    if is_alternating:
        scatter_items = [(h, l) for h, l, s in zip(handles, labels, is_scatter) if s]
        line_items = [(h, l) for h, l, s in zip(handles, labels, is_scatter) if not s]

        sorted_items = scatter_items + line_items
        handles = [item[0] for item in sorted_items]
        labels = [item[1] for item in sorted_items]

    # 3. Regel 1: Start-Spaltenanzahl passend zur Paritaet von n_entries waehlen
    ncol_start = 2 if n_entries % 2 == 0 else 1

    # "Plotbreite" = Gesamtbreite der Figure inkl. Achsenbeschriftung etc.
    fig_width_pts = fig.get_size_inches()[0] * 72
    allowed_max_width_pts = fig_width_pts * width_tolerance

    # Vorhandene Legende entfernen, damit sie die Messung nicht verfaelscht
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()

    # 4. Regel 2: Fuer jede erlaubte (paritaetskonforme) Spaltenanzahl die
    # Legende probeweise rendern und die tatsaechliche Breite messen
    optimal_ncol = ncol_start
    for test_ncol in range(ncol_start, n_entries + 1, 2):
        trial_legend = ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15 / fraction),
            ncol=test_ncol,
        )
        fig.canvas.draw()
        # fig.canvas ist zur Laufzeit ein Agg-basierter Canvas (Agg, TkAgg,
        # Qt5Agg, ...); FigureCanvasBase selbst deklariert get_renderer()
        # nicht, daher der cast fuer den Type Checker.
        renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
        legend_width_pts = trial_legend.get_window_extent(renderer).width / fig.dpi * 72
        trial_legend.remove()

        if legend_width_pts <= allowed_max_width_pts:
            optimal_ncol = test_ncol
        else:
            break

    # 5. Finale Legende setzen
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15 / fraction),
        ncol=optimal_ncol,
    )

    return fig, ax

##############################################################################
##############################################################################

def save_figure(fig: matplotlib.figure.Figure, path: Path | str):
    """
    Saves the given figure to a PDF file in the specified output folder.
    
    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure object to save.
        path (Path | str): The path where the figure will be saved.
        fraction (float): The fraction of the page width to use for the figure.
        subplots (tuple): The number of rows and columns of subplots.
    """
    fig.savefig(
        str(path)+".pdf",
        bbox_inches='tight',
        dpi=300, # set the resolution for "rasterized" elements (i.e. scatter-points) in the figure
        backend='pdf',
        )
    plt.show()
    # plt.close(fig)  # Close the figure after saving to free up memory

##############################################################################
##############################################################################

def standard_scatter_plot(
        # plotting data
        x: list[float],
        y: list[float],
        z: list[float] | None = None,

        label: str | None = None,
        color: tuple[float, float, float] |str | None = TITLECOLOR,
        markersize: float = MARKERSIZE,
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
        fig, ax = get_fig_ax(fraction_textwidth=fraction_textwidth)

    # apply 'prefix' style to label
    if label is not None:
        labels = label.split("\n")
        label = ""
        for l in labels: 
            labelpart = translate_and_prefix_label(l, translation_dict)
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

        divider = make_axes_locatable(ax)
        # size="5%" bestimmt die Breite der Colorbar, pad=0.15 den Abstand zum Hauptplot
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = fig.colorbar(img, cax=cax)

        if translation_dict is not None and z_label is not None and z_label in translation_dict:
            z_label = translation_dict[z_label]

        zprefix = None
        if zstyle is not None:
            if zstyle == 'prefix':
                z_abs_max = max(abs(z_val) for z_val in z)
                zprefix, zsi_exponent, zexponent = get_SI_prefix((z_abs_max, z_abs_max))
                z_scale_factor = 10 ** (-zsi_exponent)
                cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * z_scale_factor:g}"))
            else:
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                cbar.ax.yaxis.set_major_formatter(formatter)
                cbar.ax.ticklabel_format(axis='y', style= zstyle, useMathText=True)
    
        if zstyle == 'prefix' and z_label is not None:
            if zprefix is not None:
                z_label = set_prefix_in_label(string = z_label, prefix = zprefix)
            cbar.set_label(z_label, rotation=270, labelpad=20 + 2*FONTSIZE,)

    # translate labels if translation_dict is provided
    if translation_dict is not None:
        if x_label is not None and x_label in translation_dict:
            x_label = translation_dict[x_label]
        if y_label is not None and y_label in translation_dict:
            y_label = translation_dict[y_label]

    if z is not None:
        # increase the height of the figure to match the aspect ratio of the data
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        data_aspect_ratio = abs(y_max - y_min) / abs(x_max - x_min)

        # get the current figure width in inches
        fig = ax.get_figure()
        breite, _ = fig.get_size_inches()

        # set the height of the figure based on the data aspect ratio
        fig.set_size_inches(breite, breite * data_aspect_ratio)
        ax.set_aspect('equal', adjustable='box')

        # set equal locator (ticks) for x and y axes
        equal_locator = ticker.AutoLocator()
        ax.xaxis.set_major_locator(equal_locator)
        ax.yaxis.set_major_locator(equal_locator)

    # set background
    fig, ax = plot_background(fig, ax, 
                                  title=title, 
                                  x_label=x_label, 
                                  y_label=y_label,
                                  xstyle = xstyle,
                                  ystyle = ystyle,
                                  )

    if label is not None:
        dynamic_legend(fig, ax, fraction=fraction_textwidth)

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
            - label (str | None); label for the data points (optional)
            - color (str | None); color for the data points (optional, ignored if z is provided)
            - translation_dict (dict[str, str] | None); dictionary for translating labels (optional)
    
            - fraction_textwidth (float); fraction of text width for figure size
            - fig (matplotlib.figure.Figure | None); existing figure to plot on (optional)
            - ax (matplotlib.axes.Axes | None); existing axes to plot on (optional)
    
            - title (str | None); title of the plot (optional)
            - x_label (str | None); label for the x-axis (optional)
            - y_label (str | None); label for the y-axis (optional)
            - z_label (str | None); label for the colorbar (optional, only used if z is provided)
    
            - xstyle (Literal['sci', 'scientific', 'plain', 'prefix'] | None); style for x-axis ticks (optional)
            - ystyle (Literal['sci', 'scientific', 'plain', 'prefix'] | None); style for y-axis ticks (optional)
            - zstyle (Literal['sci', 'scientific', 'plain', 'prefix'] | None); style for colorbar ticks (optional, only used if z is provided)
    
        Returns:
            tuple (Figure, Axes): the figure and axes objects of the plot
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