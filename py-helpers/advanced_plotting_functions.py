# Author: Niko Bleidistel
# last change: 2026-08-19

##############################################################################
# import packages
##############################################################################

import math
import re

# for plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker

# for type hinting
from typing import Literal
import matplotlib.axes
import matplotlib.figure


##############################################################################
##############################################################################
# constants
##############################################################################
##############################################################################

FONTSIZE = 12           # in pt, of the latex document
TEXTWIDTH = 455.24411   # in pt, of the latex document
TEXTHEIGHT = 535.6748   # in pt, of the latex document

MARKERSIZE = 25         # of the plot markers
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
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{lmodern} \usepackage{upgreek} \usepackage{xfrac}",
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
# new features
##############################################################################
##############################################################################

def calc_figure_size(fraction: float = 1.0, width_pt: float = TEXTWIDTH, subplots: tuple = (1, 1)):
    """
    Calculates the exact figure size in inches based on LaTeX textwidth.
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

def get_SI_prefix(limits: tuple[float, float], use_u_as_micro: bool = False) -> tuple[str | None, int, int]:
        """
        Returns the SI prefix for a given numeric value.
        
        Args:
            limits (tuple[float, float]): A tuple containing the minimum and maximum values.
            use_u_as_micro (bool): Whether to use 'u' as the symbol for micro or 'μ'.
        
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
            "Q": 30,   # Quetta
            "R": 27,   # Ronna
            "Y": 24,   # Yotta
            "Z": 21,   # Zetta
            "E": 18,   # Exa
            "P": 15,   # Peta
            "T": 12,   # Tera
            "G": 9,    # Giga
            "M": 6,    # Mega
            "k": 3,    # Kilo
            "h": 2,    # Hecto
            "da": 1,   # Deca

            # zero exponent (no prefix)
            "": 0,     # No prefix
            
            # Small values (negative exponents)
            "d": -1,   # Deci
            "c": -2,   # Centi
            "m": -3,   # Milli
            "u": -6,   # Micro (often written as µ)
            "n": -9,   # Nano
            "p": -12,  # Pico
            "f": -15,  # Femto
            "a": -18,  # Atto
            "z": -21,  # Zepto
            "y": -24,  # Yocto
            "r": -27,  # Ronto
            "q": -30,   # Quekto
        }
        if not use_u_as_micro:
            # PREFIX_TO_EXPONENT["μ"] = -6   # Use 'μ' for micro instead of 'u'
            # PREFIX_TO_EXPONENT[r"$\\mu$"] = -6   # Use 'μ' for micro instead of 'u'
            PREFIX_TO_EXPONENT[r"$\\upmu$"] = -6   # Use 'μ' for micro instead of 'u'
            del PREFIX_TO_EXPONENT["u"]    # Remove 'u' from the dictionary

        # Returns the key, or None if the value doesn't exist
        siprefix = next((k for k, v in PREFIX_TO_EXPONENT.items() if v == xsi_exponent), None)
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
        if re.search(r"\[\w", string):
            return re.sub(r"\[(?=\w)", f"[{prefix}", string)
        elif re.search(r"\(\w", string):
            return re.sub(r"\((?=\w)", f"({prefix}", string)
        else:
            return f"{string} [{prefix}"

##############################################################################

def set_prefix_in_number_unit_string(string: str):
    """
    Searches the pattern '= number [' in the string. Then it extracts the number and uses  
    """
    pattern = r"=\s*(-?\d+\.?\d*)(?:\s*\[)?"
    match = re.search(pattern, string)
    if match:
        number = float(match.group(1))
        siprefix, xsi_exponent, exponent_diff = get_SI_prefix((number, number))
        scaled_number = number * (10 ** (-xsi_exponent))

        if siprefix is not None:
            string = re.sub(pattern, f"= {scaled_number} [{siprefix}", string)
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

def translate_and_prefix_label(label: str, translation_dict: dict[str, str] | None = None) -> str:
    """
    Translates the variable in the label using the provided translation dictionary and applies SI prefix notation.

    The label is expected to be in the format "variable = value [unit]".
    If a translation dictionary is provided and the variable is found in it, the variable will be replaced with its corresponding translation.
    If there is a unit in the (now translated) variable -- marked by either round or square brackets -- and no unit behind the value, the unit will be moved behind the value.
    The function also applies SI prefix notation to the value, meaning that the magnitude of the value will be adjusted to a suitable SI prefix, and the prefix will be added to the unit.

    Args:
        label (str):    The input label string, expected to be in the format "variable = value [unit]".
        translation_dict (dict[str, str] | None): A dictionary for translating variable names. Default is None.

    Returns:
        str: The translated and prefixed label.
    """

    # split the label into variable and rest
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
    return set_prefix_in_number_unit_string(short_label)


##############################################################################
##############################################################################
# plotting helper
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
            ax.ticklabel_format(axis='x', style= xstyle, scilimits=(0,0), useMathText=True)
    if ystyle is not None:
        if ystyle == 'prefix':
            fig, ax, yprefix, yexponent = prefixes_notation(fig, ax, 'y')
        else:
            ax.ticklabel_format(axis='y', style=ystyle, scilimits=(0,0), useMathText=True)

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
        ax.set_title(rf"\textbf{{{title}}}", color=TITLECOLOR)

    ax.grid(True, which='both', linestyle='--')
    
    return fig, ax

##############################################################################
##############################################################################

