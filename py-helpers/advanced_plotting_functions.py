# Author: Niko Bleidistel
# last change: 2026-08-19

"""
call "finalize_layout_and_save_figure(fig, ax, fraction=1.0), path = ..., )"
to apply the style provided by the functions in this file to a matplotlib figure and axes.

use "standard_scatter_plot(...)" or its dataframe wrapper "standard_scatter_plot_df(...)" to automatically apply the "prefix style" to labels.
label are best provided in one of the following forms
- "variable = value [unit]"
- "variable [unit] = value"
to allow automatic SI prefixing of the unit based on the magnitude of the value. LaTeX math blocks should be handled correctly.
Hint: The equal sign hints that a prefix should be applied to the unit, while the absence of an equal sign will leave the label unchanged.
"""

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

from matplotlib.transforms import Bbox
import matplotlib.transforms as mtransforms

# for type hinting
from typing import Literal, cast

import matplotlib.axes
import matplotlib.figure
import matplotlib.text
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

MARKERSIZE = 4         # of the plot markers

MAINCOLOR = (0/255, 51/255, 102/255)            # color of the title = FAU-Blau
SECONDARYCOLOR = (190/255, 205/255, 220/255)    # Light blue
BACKGROUNDCOLOR = (225/255, 225/255, 225/255)   # Light gray

LEFT_LABEL_SPACE = 1.20 # inches, space to the left of the y-axis label for long labels with units
RIGHT_LABEL_SPACE = 1.20

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
    "axes.titlesize": 1.2 * FONTSIZE,                
    "axes.labelsize": 1.0 * FONTSIZE,
    "legend.fontsize": 0.9 * FONTSIZE,
    "xtick.labelsize": 0.8 * FONTSIZE,
    "ytick.labelsize": 0.8 * FONTSIZE,
})

plt.rcParams.update({
    "axes.titlecolor": MAINCOLOR,           # color of the title
    "axes.labelcolor": "black",             # color of the x and y axis labels
    "axes.edgecolor": MAINCOLOR,            # color of the axes lines
    "xtick.color": "black",                 # color of the x-axis tick labels
    "ytick.color": "black",                 # color of the y-axis tick labels
    "axes.linewidth": 1.0,                  # linewidth of the axes lines

    "legend.facecolor": "white",            # background color of the legend box
    "legend.edgecolor": MAINCOLOR,          # frame color of the legend box
    "legend.frameon": True,                 # whether to draw a frame around the legend
    "legend.labelcolor": "black",           # color of the text in the legend
    "legend.linewidth": 1.0,                # linewidth of the frame around the legend
})

plt.rcParams.update({
    'legend.columnspacing': 1.0,  # horizontal distance between columns in the legend
    'legend.labelspacing':  0.5,  # vertical distance between rows in the legend
    'legend.handletextpad': 0.7,  # distance between symbol and text
})

##############################################################################
##############################################################################
# string formaters
##############################################################################
##############################################################################

# A bracketed unit block: [ ... ] or ( ... ), optionally wrapped in a single
# math block ($...$).
_UNIT_BLOCK = r"\$\[[^\]]*\]\$|\[[^\]]*\]|\$\([^)]*\)\$|\([^)]*\)"

# Same as _UNIT_BLOCK, but without the $-wrapped variants.
_PLAIN_UNIT_BLOCK = r"\[[^\]]*\]|\([^)]*\)"

# A number: optional '+' or '-' sign, decimal part, optional exponent.
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

_PLAIN_UNIT_BLOCK_RE = re.compile(_PLAIN_UNIT_BLOCK)
_UNIT_BLOCK_AT_END_RE = re.compile(rf"(?P<unit>{_UNIT_BLOCK})\s*$")
_NUMBER_AND_UNIT_RE = re.compile(rf"(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT_BLOCK})?")
_NUMERIC_VALUE_ONLY_RE = re.compile(rf"{_NUMBER}\s*(?:{_UNIT_BLOCK})?")
_MATH_BLOCK_RE = re.compile(r"\$[^$]+\$")
_EXPLICIT_MATH_RE = re.compile(r"\$.*?\$")


# =============================================================================
# Internal helpers
# =============================================================================

def _strip_math_wrap(text: str) -> tuple[str, bool]:
    """
    Recurring pattern: "is `text` wrapped in exactly one $...$ block (no
    further '$' inside)?"

    Returns (inner_content, True) if so, otherwise (text.strip(), False).
    """
    value = text.strip()
    if value.startswith("$") and value.endswith("$") and "$" not in value[1:-1]:
        return value[1:-1].strip(), True
    return value, False


def _normalize_unit_token(value: str) -> str:
    """Strip LaTeX wrappers, brackets, and $ so units can be compared regardless of formatting."""
    value = value.strip().replace("$", "")
    value = value.replace(r"\mathrm{", "").replace(r"\text{", "").replace(r"\bm{", "")
    value = value.replace(r"\upmu", "μ").replace(r"\mu", "μ")
    for ch in "{}[]()":
        value = value.replace(ch, "")
    return value.strip()


def _unit_already_has_prefix(unit_text: str, prefix: str) -> bool:
    """True only when `unit_text` already includes `prefix` as a real SI prefix, not just the bare unit."""
    if not unit_text or not prefix:
        return False
    norm_unit = _normalize_unit_token(unit_text)
    norm_prefix = _normalize_unit_token(prefix)
    if not norm_prefix or norm_unit == norm_prefix:
        return False
    return norm_unit.startswith(norm_prefix) and len(norm_unit) > len(norm_prefix)


def _prefix_unit_text(unit_text: str, prefix: str) -> str:
    """Insert `prefix` inside a unit's brackets/math wrapper, preserving the formatting used."""
    if not unit_text or not prefix:
        return unit_text

    value = unit_text.strip()
    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()

    if value.startswith("[") and value.endswith("]"):
        return f"[{prefix}{value[1:-1]}]"
    if value.startswith("(") and value.endswith(")"):
        return f"({prefix}{value[1:-1]})"
    return f"[{prefix}{value}]"


def _split_formula_and_unit(string: str) -> tuple[str, str]:
    """Split a label-like symbol from an optional trailing unit."""
    value = string.strip()
    if not value:
        return "", ""

    unit = ""
    formula_part = value

    unit_match = _UNIT_BLOCK_AT_END_RE.search(value)
    if unit_match:
        unit_text = unit_match.group("unit").strip()
        inner, is_math = _strip_math_wrap(unit_text)
        unit = inner if is_math else unit_text.strip("[]() ")
        formula_part = value[:unit_match.start()].strip()

    if not formula_part:
        return "", unit.strip()

    math_block_match = _MATH_BLOCK_RE.search(formula_part)
    if math_block_match:
        return math_block_match.group(0).strip(), unit.strip()

    return formula_part.strip("[]() ").strip(), unit.strip()


def _is_numeric_value_only(text: str) -> bool:
    """True only for a bare numeric value (with optional unit), not for mixed text like '..., automatic'."""
    value = text.strip()
    if not value:
        return False
    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return bool(_NUMERIC_VALUE_ONLY_RE.fullmatch(value))


# =============================================================================
# Public functions
# =============================================================================

def set_prefix_in_label(string: str, prefix: str):
    """Insert the SI prefix into the first bracketed unit block without duplicating it on repeated calls."""
    if not string or not prefix:
        return string

    match = _PLAIN_UNIT_BLOCK_RE.search(string)
    if match is None:
        return f"{string} [{prefix}1]"

    full_match = match.group(0)
    inner = full_match[1:-1]
    if _unit_already_has_prefix(inner, prefix):
        return string

    updated = _prefix_unit_text(full_match, prefix)
    return string[:match.start()] + updated + string[match.end():]


def clean_label_text(text: str) -> str:
    """Strip an accidental outer math wrapper from plain text, leaving numeric values untouched."""
    value = (text or "").strip()
    if not value:
        return value

    value, _ = _strip_math_wrap(value)
    if "=" not in value:
        return value

    left, right = value.split("=", 1)
    left, right = left.strip(), right.strip()
    if re.search(_NUMBER, right):
        return value
    return f"{left} = {right}".strip()


def set_prefix_in_number_unit_string(string: str, bm: bool = False) -> str:
    """
    Insert the appropriate SI prefix into a string containing a numeric value and an optional unit, while preserving any existing LaTeX math formatting.
    1. Rule: If the string does not contain an '=', return the original string unchanged.
    2. Rule: Else split at the first '=' and process the right part to identify a numeric value and an optional unit.
    3. Rule: If a numeric value is found and already wrapped in a single math block, return the original string unchanged.
    4. Rule: If no unit is found, wrap the numeric value in a math block and return the updated string.
    5. Rule: If a unit and a value is found, determine the appropriate SI prefix based on the magnitude of the number.
    6. Rule: If there is no suitable SI prefix or if the unit already contains the prefix, return the original string unchanged.
    7. Rule: Else insert the SI prefix into the unit text, preserving any existing brackets and formatting.
    8. Rule: If there is a trailing comma (+ optional text) after the numeric value, place the math block before the comma and keep the trailing text outside the math block.
    """
    if '=' not in string:
        return string

    left, right = string.split('=', 1)
    left = left.strip()
    value_part = right.strip()

    _, is_single_math_wrap = _strip_math_wrap(value_part)
    if is_single_math_wrap:
        return string

    match = _NUMBER_AND_UNIT_RE.search(value_part)
    if match is None or match.group("number") == "":
        return string

    if match.start() > 0 and value_part[match.start() - 1] == "$" and match.end() < len(value_part) and value_part[match.end()] == "$":
        # If the number is already wrapped in a single math block, return the original string without further processing,
        # to avoid creating display style math blocks with double "$$...$$".
        return string

    number_str = match.group("number")
    unit_text = match.group("unit")
    number = float(number_str)

    if unit_text is None:
        replacement = f"${number_str}$"
        updated_value_part = value_part[:match.start()] + replacement + value_part[match.end():]
        updated_value_part = updated_value_part.strip()

        if match.start() == 0 and value_part[match.end():].strip().startswith(","):
            updated_value_part = replacement + value_part[match.end():].strip()
        return f"{left} = {updated_value_part}".strip()

    siprefix, xsi_exponent, _ = get_SI_prefix((number, number), bm=bm)
    if siprefix is None or _unit_already_has_prefix(unit_text, siprefix):
        return string

    scaled_number = number * (10 ** (-xsi_exponent))
    scaled_str = f"{scaled_number:g}"
    replacement = f"{scaled_str}{_prefix_unit_text(unit_text, siprefix)}"
    updated_value_part = value_part[:match.start()] + replacement + value_part[match.end():]
    updated_value_part = updated_value_part.strip()

    if match.start() == 0 and value_part[match.end():].strip().startswith(","):
        updated_value_part = f"${replacement}$" + value_part[match.end():].strip()

    return f"{left} = {updated_value_part}".strip()


def translate_and_prefix_label(label: str, translation_dict: dict[str, str] | None = None, bm: bool = False) -> str:
    """Translate the label and apply SI prefixing without disturbing LaTeX math blocks."""
    if '=' not in label:
        translated = label
        if translation_dict is not None and label in translation_dict:
            translated = translation_dict[label]
        return clean_label_text(translated).strip()

    left, right = label.split('=', 1)
    variable = left.strip()
    rest = right.strip()

    if translation_dict is not None and variable in translation_dict:
        variable = translation_dict[variable]

    explicit_math_label = bool(_EXPLICIT_MATH_RE.search(variable or ""))

    if explicit_math_label:
        unit_match = _UNIT_BLOCK_AT_END_RE.search(variable.strip())
        if unit_match:
            unit_text, _ = _strip_math_wrap(unit_match.group("unit"))
            variable = variable[:unit_match.start()].rstrip()
            if rest and not _PLAIN_UNIT_BLOCK_RE.search(rest):
                rest = f"{rest}{unit_text}".strip()
        result = f"{variable.strip()} = {rest}".strip()
    else:
        short, unit = _split_formula_and_unit(variable)
        if not short:
            short = variable.strip()

        if unit and not _PLAIN_UNIT_BLOCK_RE.search(rest):
            rest = f"{rest} [{unit}]".strip()

        result = f"{short} = {rest}".strip()

    result = set_prefix_in_number_unit_string(result, bm=bm).strip()

    if '=' in result:
        left_expr, right_expr = result.split('=', 1)
        left_expr = left_expr.strip()
        right_expr = right_expr.strip()

        # Only wrap if the entire right-hand side is a pure numeric value.
        # Mixed text like 'z = -3.5[μm], automatic' is left untouched.
        if right_expr and _is_numeric_value_only(right_expr):
            already_single_wrapped = (
                right_expr.startswith("$") and right_expr.endswith("$") and "$" not in right_expr[1:-1]
            )
            if not already_single_wrapped:
                right_stripped = right_expr
                if right_stripped.startswith("$") and right_stripped.endswith("$"):
                    right_stripped = right_stripped[1:-1].strip()
                if not right_stripped.startswith("$") and not right_stripped.endswith("$"):
                    result = f"{left_expr} = ${right_stripped}$".strip()

    return result


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
        siprefix = r"\bm{" + siprefix + "}"  # Add bold math formatting to the SI prefix
    return siprefix, xsi_exponent, exponent_diff


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
##############################################################################
# plot position helper
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

    if title is not None:
        ax.set_title(title, pad=20,)

    ax.set_axisbelow(True)  # Ensure grid is below other plot elements
    ax.grid(True, which='both', linestyle='--', zorder=0)
    
    return fig, ax

##############################################################################
##############################################################################

# def set_wrapped_title(
#         fig: matplotlib.figure.Figure,
#         ax: matplotlib.axes.Axes,
#         title: str | None = None,
#         y: float = 1.0,               # y-position of the main title (in figure coordinates)
#         gap_in: float = 0.2,          # gap between main title and subtitle
#         linespacing: float | None = None,  # gap between lines in the subtitle (None = auto)
#         width_tolerance: float = 1.0,
#     ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list[matplotlib.text.Text]]:
#     """
#     Sets the title as TWO separate text artists:
#       1. Main title (before "("), bold, in axes.titlesize, on top.
#       2. Subtitle (bracket content without the brackets), regular FONTSIZE,
#          below it -- wrapped at commas if needed.

#     Returns a list of both created text artists (empty if no title was set),
#     so that finalize_layout/save_figure can correctly include their height.

#     If linespacing is None, it is chosen automatically: a compact value for
#     a single-line subtitle, and a larger, textsize-dependent value once the
#     subtitle wraps onto multiple lines (larger fonts need proportionally
#     more line spacing to stay readable).
#     """
#     if title is None:
#         title = ax.get_title()
#         ax.set_title("")

#     if not title:
#         return fig, ax, []

#     titlecolor = plt.rcParams.get("axes.titlecolor", MAINCOLOR)
#     titlesize = plt.rcParams.get("axes.titlesize", FONTSIZE)
#     textsize = plt.rcParams.get("font.size", FONTSIZE)

#     if isinstance(titlesize, str):
#         titlesize = FONTSIZE

#     if isinstance(textsize, str):
#         textsize = FONTSIZE

#     if "(" not in title:
#         main_artist = fig.suptitle(rf"\textbf{{{title}}}", y=y)
#         return fig, ax, [main_artist]

#     main_title = title.split("(", 1)[0].strip()
#     subtitle_content = title.split("(", 1)[1].rsplit(")", 1)[0].strip()

#     fig_width_in = fig.get_size_inches()[0]
#     max_width_in = fig_width_in * width_tolerance

#     fig.canvas.draw()
#     renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()

#     def render_width_in(text: str, fontsize: float) -> float:
#         # invisible text artist used only to measure the rendered width
#         t = fig.text(0, 0, rf"{{{text}}}", alpha=0.0, fontsize=fontsize)
#         fig.canvas.draw()
#         bbox = t.get_window_extent(renderer)
#         t.remove()
#         return bbox.width / fig.dpi

#     segments = [s.strip() for s in subtitle_content.split(",")]

#     if len(segments) <= 1 or render_width_in(subtitle_content, textsize) <= max_width_in:
#         wrapped_content = subtitle_content
#     else:
#         # Special rule: if the first segment contains no "$" (i.e. plain
#         # text such as "Round coil and grid" instead of a formula), check
#         # whether isolating it on its own line is actually needed -- only
#         # do so if the remaining segments would not fit on one line anyway.
#         isolate_first = False
#         if "$" not in segments[0] and len(segments) > 1:
#             rest_text = ", ".join(segments[1:])
#             if render_width_in(rest_text, textsize) > max_width_in:
#                 isolate_first = render_width_in(segments[0] + ",", textsize) <= max_width_in

#         lines: list[list[str]] = [[segments[0]]] if isolate_first else [[]]
#         remaining = segments[1:] if isolate_first else segments

#         # Greedy packing: fit as many segments as possible per line
#         for seg in remaining:
#             candidate_line = lines[-1] + [seg]
#             candidate_text = ", ".join(candidate_line)

#             if lines[-1] and render_width_in(candidate_text, textsize) > max_width_in:
#                 lines.append([seg])
#             else:
#                 lines[-1] = candidate_line

#         joined_lines = [", ".join(line) for line in lines]
#         for j in range(len(joined_lines) - 1):
#             joined_lines[j] += ","
#         wrapped_content = "\n".join(joined_lines)

#     # 1. Set the main title (bold, titlesize) and measure its position
#     main_artist = fig.suptitle(rf"\textbf{{{main_title}}}", y=y, fontsize=titlesize, color=titlecolor)
#     fig.canvas.draw()
#     main_bbox = main_artist.get_window_extent(renderer)
#     main_bottom_in = main_bbox.y0 / fig.dpi

#     # 2. Compute subtitle y-position: gap_in below the main title
#     fig_height_in = fig.get_size_inches()[1]
#     sub_y_fig = (main_bottom_in - gap_in) / fig_height_in

#     # 3. Auto-determine linespacing if not explicitly provided:
#     #    a single-line subtitle can stay compact, but a wrapped
#     #    (multi-line) subtitle needs more room, scaled by textsize so
#     #    larger fonts get proportionally more spacing.
#     # min_linespacing = 3e-4
#     min_linespacing = 0.03
#     print(f"min_linespacing: {min_linespacing}")
#     if linespacing is None:
#         n_lines = wrapped_content.count("\n") + 1
#         if n_lines > 1:
#             linespacing = min_linespacing + 0.03 * textsize
#         else:
#             linespacing = min_linespacing

#     # For a multi-line subtitle: verticalalignment='top' ensures sub_y_fig
#     # marks the TOP edge of the subtitle block

#     sub_artist = fig.text(
#         0.5, sub_y_fig+linespacing, wrapped_content,
#         ha='center', va='top',
#         fontsize=textsize,
#         # linespacing=linespacing,
#         color=titlecolor,
#     )

#     fig.canvas.draw()

#     return fig, ax, [main_artist, sub_artist]

import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.text
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import cast, Any

# Konstanten (falls nicht global definiert)
MAINCOLOR = "black"
FONTSIZE = 10

def set_wrapped_title(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    title: str | None = None,
    gap_to_plot_in: float = 0.2,       # Abstand vom Plot-Rand bis zur Unterkante des Subtitels
    gap_in: float = 0.15,               # Abstand von Oberkante Subtitel bis Unterkante Haupttitel
    linespacing: float | None = None,
    width_tolerance: float = 1.0,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list[matplotlib.text.Text]]:
    """
    Positioniert den Titel dynamisch von unten nach oben über dem Plot-Fenster.
    
    Unterkante Subtitel = Oberkante Plot + gap_to_plot_in
    Unterkante Haupttitel = Oberkante Subtitel + gap_in
    """
    if title is None:
        title = ax.get_title()
        ax.set_title("")

    if not title:
        return fig, ax, []

    titlecolor = plt.rcParams.get("axes.titlecolor", MAINCOLOR)
    titlesize = plt.rcParams.get("axes.titlesize", FONTSIZE)
    textsize = plt.rcParams.get("font.size", FONTSIZE)

    if isinstance(titlesize, str): titlesize = FONTSIZE
    if isinstance(textsize, str): textsize = FONTSIZE

    fig_width_in, fig_height_in = fig.get_size_inches()
    max_width_in = fig_width_in * width_tolerance

    fig.canvas.draw()
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()

    # --- Hilfsfunktion für Textbreitenmessung ---
    def render_width_in(text: str, fontsize: float) -> float:
        t = fig.text(0, 0, rf"{{{text}}}", alpha=0.0, fontsize=fontsize)
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer)
        t.remove()
        return bbox.width / fig.dpi

    # --- Text-Splitting und Wrapping ---
    if "(" not in title:
        main_title = title
        subtitle_content = ""
    else:
        main_title = title.split("(", 1)[0].strip()
        subtitle_content = title.split("(", 1)[1].rsplit(")", 1)[0].strip()

    # Wrapping-Logik für Subtitle
    wrapped_content = subtitle_content
    if subtitle_content:
        segments = [s.strip() for s in subtitle_content.split(",")]
        if len(segments) > 1 and render_width_in(subtitle_content, textsize) > max_width_in:
            isolate_first = False
            if "$" not in segments[0]:
                rest_text = ", ".join(segments[1:])
                if render_width_in(rest_text, textsize) > max_width_in:
                    isolate_first = render_width_in(segments[0] + ",", textsize) <= max_width_in

            lines: list[list[str]] = [[segments[0]]] if isolate_first else [[]]
            remaining = segments[1:] if isolate_first else segments

            for seg in remaining:
                candidate_line = lines[-1] + [seg]
                if lines[-1] and render_width_in(", ".join(candidate_line), textsize) > max_width_in:
                    lines.append([seg])
                else:
                    lines[-1] = candidate_line

            joined_lines = [", ".join(line) for line in lines]
            for j in range(len(joined_lines) - 1):
                joined_lines[j] += ","
            wrapped_content = "\n".join(joined_lines)

    # --- Absolute Positionierung von unten nach oben ---
    
    # 1. Bestimme die Oberkante des Plot-Fensters (ax) in Inches
    ax_bbox = ax.get_window_extent(renderer)
    plot_top_in = ax_bbox.y1 / fig.dpi

    artists = []

    if wrapped_content:
        # 2. Berechne exakte Y-Position für die Unterkante des Subtitels
        sub_bottom_in = plot_top_in + gap_to_plot_in
        sub_y_fig = sub_bottom_in / fig_height_in

        # Zeichne Subtitel (va='bottom' verankert die Unterkante)
        sub_artist = fig.text(
            0.5, sub_y_fig, wrapped_content,
            ha='center', va='bottom',
            fontsize=textsize,
            color=titlecolor,
        )
        artists.append(sub_artist)
        
        # Bestimme die tatsächliche Oberkante des gezeichneten Subtitels
        fig.canvas.draw()
        sub_bbox = sub_artist.get_window_extent(renderer)
        sub_top_in = sub_bbox.y1 / fig.dpi
        
        # Basis für den Haupttitel ist die Oberkante des Subtitels
        main_bottom_in = sub_top_in + gap_in
    else:
        # Wenn kein Subtitel existiert, nutzt der Haupttitel den direkten Plot-Abstand
        main_bottom_in = plot_top_in + gap_to_plot_in

    # 3. Berechne Y-Position für die Unterkante des Haupttitels
    main_y_fig = main_bottom_in / fig_height_in

    # Zeichne Haupttitel (va='bottom' verankert die Unterkante)
    main_artist = fig.suptitle(
        rf"\textbf{{{main_title}}}", 
        y=main_y_fig, 
        fontsize=titlesize, 
        color=titlecolor,
        va='bottom'
    )
    artists.insert(0, main_artist)

    # Finaler Render-Call, damit die neuen Positionen aktiv sind
    fig.canvas.draw()

    return fig, ax, artists


##############################################################################

def dynamic_legend(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    fraction: float = 1.0,
    width_tolerance: float = 1.00,
    max_columns: int = 3,
    use_parity: bool = False,
):
    """
    Creates a dynamic legend for a matplotlib plot, adjusting the number of columns based on the plot width and number of legend entries (even or odd).
    It also sorts the legend entries: If scatter and line plots alternate, placing scatter entries on the left and line entries on the right.

    Args:
        fig (matplotlib.figure.Figure):     The matplotlib figure object.
        ax (matplotlib.axes.Axes):          The matplotlib axes object.
        fraction (float):                   Fraction of the page width to use for the legend.
        max_columns (int):                  Maximum number of columns in the legend.
        width_tolerance (float):            Tolerance factor for the maximum allowed legend width relative to the figure width.

    Returns:
        tuple (Figure, Axes): A tuple containing the updated matplotlib figure and axes objects.
    """
    # 1. Get the current legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    n_entries = len(handles)

    if n_entries == 0:
        return fig, ax

    # 2. Rule 1: Check if the legend entries alternate between scatter and line plots
    # and change the order of the legend entries so that scatter entries are on the left and line entries are on the right.
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

    # 3. Rule 2: Check the parity of the number of legend entries if flag is set.
    if use_parity:
        ncol_start = 2 if n_entries % 2 == 0 else 1
        test_ncol_range = range(ncol_start, n_entries + 1, 2)
    else:
        ncol_start = 1
        test_ncol_range = range(ncol_start, n_entries + 1, 1)

    # limit the number of columns to the maximum allowed
    if n_entries % (max_columns + 1) == 0:
        max_columns = max_columns + 1
    test_ncol_range = [ncol for ncol in test_ncol_range if ncol <= max_columns]

    # get the figure width (= total width of the figure including axes labels, etc.) in points and calculate the allowed maximum width for the legend
    fig_width_pts = fig.get_size_inches()[0] * 72
    allowed_max_width_pts = fig_width_pts * width_tolerance

    # Remove the existing legend so it does not distort the measurement
    existing_legend = ax.get_legend()
    if existing_legend is not None:
        existing_legend.remove()

    # 4. Rule 2: For each allowed (parity-conforming) number of columns, render the
    # legend as a test and measure the actual width
    optimal_ncol = ncol_start
    for test_ncol in test_ncol_range:
        trial_legend = ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15 / fraction),
            ncol=test_ncol,
        )
        fig.canvas.draw()
        # fig.canvas is at runtime an Agg-based canvas (Agg, TkAgg, Qt5Agg, ...);
        # FigureCanvasBase itself does not declare get_renderer(), hence the cast for the type checker.
        renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
        legend_width_pts = trial_legend.get_window_extent(renderer).width / fig.dpi * 72
        trial_legend.remove()

        if legend_width_pts <= allowed_max_width_pts:
            optimal_ncol = test_ncol
        else:
            break

    # 5. Finally, render the legend with the optimal number of columns
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15 / fraction),
        ncol=optimal_ncol,
    )

    return fig, ax

##############################################################################

def measure_title_stack_height_in(
        fig: matplotlib.figure.Figure,
        title_artists: list[matplotlib.text.Text],
        axis_gap_in: float = 0.05,
        title_gap_in: float = 0.2,
    ) -> float:
    """
    Measures the total vertical space required to stack the given title
    artists (e.g. [main_title, subtitle]) plus the gaps between them and
    the gap to the axes top. Artist HEIGHT does not depend on its current
    y-position, so this can be measured before the figure is resized and
    the axes repositioned, and reused consistently by finalize_layout
    (to reserve the right amount of height).
    """
    if not title_artists:
        return 0.0

    fig.canvas.draw()
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()

    total_in = axis_gap_in
    for i, artist in enumerate(title_artists):
        bbox = artist.get_window_extent(renderer)
        total_in += bbox.height / fig.dpi
        if i < len(title_artists) - 1:
            total_in += title_gap_in

    return total_in

##############################################################################

def finalize_layout(
        fig, ax,
        left_in: float = LEFT_LABEL_SPACE,
        right_in: float = RIGHT_LABEL_SPACE,
        aspect_ratio: float | None = None,
        title_artists: list[matplotlib.text.Text] | None = None,
        title_axis_gap_in: float = 0.05,
        title_gap_in: float = 0.2,
    ):
    if aspect_ratio is None:
        aspect_ratio = (5 ** 0.5 - 1) / 2

    fig.canvas.draw()
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()

    fig_width_in, _ = fig.get_size_inches()
    axes_width_in = fig_width_in - left_in - right_in
    axes_height_in = axes_width_in * aspect_ratio

    extra_artists = []
    legend = ax.get_legend()
    if legend is not None:
        extra_artists.append(legend)

    tight_bbox = fig.get_tightbbox(renderer, bbox_extra_artists=extra_artists)
    ax_bbox = ax.get_window_extent(renderer)

    top_in = max(0.0, (tight_bbox.y1 - ax_bbox.y1) / fig.dpi)
    bottom_in = max(0.0, (ax_bbox.y0 - tight_bbox.y0) / fig.dpi)

    # Use the position-independent measured title stack height instead of
    # the (unreliable) current bbox position of the title artists.
    if title_artists:
        title_height_in = measure_title_stack_height_in(
            fig, title_artists, title_axis_gap_in, title_gap_in
        )
        top_in = max(top_in, title_height_in)

    new_fig_height_in = axes_height_in + top_in + bottom_in
    fig.set_size_inches(fig_width_in, new_fig_height_in, forward=True)

    left = left_in / fig_width_in
    width = axes_width_in / fig_width_in
    bottom = bottom_in / new_fig_height_in
    height = axes_height_in / new_fig_height_in

    ax.set_position([left, bottom, width, height])
    fig.canvas.draw()
    return fig, ax

##############################################################################

def position_ylabel_left(fig, ax, x_in: float = (LEFT_LABEL_SPACE - 0.5)):
    """
    Positions the y-axis label at a fixed horizontal position (in inches
    from the left edge of the figure). The tick numbers remain unchanged,
    directly on the axis. Must be called AFTER finalize_layout, since it
    requires the final axis position.
    """
    fig_width_in, _ = fig.get_size_inches()
    ax_pos = ax.get_position()

    x_fig = x_in / fig_width_in
    x_axes = (x_fig - ax_pos.x0) / ax_pos.width

    ax.yaxis.set_label_coords(x_axes, 0.5)
    return fig, ax

##############################################################################

def position_legend(fig, ax, gap_in: float = 0.05):
    """
    Positions the legend under the x-axis label with a fixed gap (in inches).
    Must be called AFTER finalize_layout.
    """
    legend = ax.get_legend()
    if legend is None:
        return fig, ax

    fig.canvas.draw()
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()

    xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
    bottom_of_xlabel_in = xlabel_bbox.y0 / fig.dpi

    fig_height_in = fig.get_size_inches()[1]
    anchor_y_fig = (bottom_of_xlabel_in - gap_in) / fig_height_in

    legend.set_bbox_to_anchor((0.5, anchor_y_fig), transform=cast(mtransforms.Transform, fig.transFigure))
    fig.canvas.draw()
    return fig, ax

##############################################################################

def save_figure(fig, ax, path, pad_inches: float = 0.05, title_artists: list[matplotlib.text.Text] | None = None):
    """
    Saves the figure to a PDF file. The width is fixed to the current
    figure width; the height is cropped tightly around the actual content
    (axes, legend, and title artists), plus pad_inches of padding.
    """
    fig.canvas.draw()
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()

    fig_width_in, _ = fig.get_size_inches()

    extra_artists = []
    legend = ax.get_legend()
    if legend is not None:
        extra_artists.append(legend)

    tight_bbox = fig.get_tightbbox(renderer, bbox_extra_artists=extra_artists)

    y0 = tight_bbox.y0 - pad_inches
    y1 = tight_bbox.y1 + pad_inches

    # Title artists are not reliably included in get_tightbbox -> account
    # for them explicitly
    for artist in (title_artists or []):
        bbox = artist.get_window_extent(renderer)
        y1 = max(y1, bbox.y1 / fig.dpi + pad_inches)

    x0, x1 = 0.0, fig_width_in
    bbox = Bbox.from_extents(x0, y0, x1, y1)
    fig.savefig(str(path) + ".pdf", bbox_inches=bbox, dpi=300, backend='pdf')
    plt.show(fig)
    # plt.close(fig)

##############################################################################

def finalize_layout_and_save_figure(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        path: str | Path,
        fraction: float = 1.0,
        aspect_ratio: float | None = None,
        title_axis_gap_in: float = 0.05,
        title_gap_in: float = 0.2,
        skip_ylabel_positioning: bool = False
    ):
    """
    Applies the standard layout adjustments (background, legend, title, axes labels) and saves the figure to a PDF file.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to be adjusted and saved.
        ax (matplotlib.axes.Axes): The axes to be adjusted and saved.
        path (str | Path): The path where the figure will be saved.
        fraction (float): The fraction of the legend to be displayed.
        title_axis_gap_in (float): The gap between the title and the axes in inches.
        title_gap_in (float): The gap between the title and the content in inches.
    """

    fig, ax = plot_background(fig, ax)
    fig, ax = dynamic_legend(fig, ax, fraction=fraction)

    fig, ax, title_artists = set_wrapped_title(fig, ax)
    fig, ax = finalize_layout(
        fig, ax, aspect_ratio=aspect_ratio, title_artists=title_artists,
        title_axis_gap_in=title_axis_gap_in, title_gap_in=title_gap_in
    )

    if not skip_ylabel_positioning:
        fig, ax = position_ylabel_left(fig, ax)
    fig, ax = position_legend(fig, ax)

    save_figure(fig, ax, path, title_artists=title_artists)

##############################################################################
##############################################################################
# Plotting helper
##############################################################################
##############################################################################

def standard_scatter_plot(
        # plotting data
        x: list[float],
        y: list[float],
        z: list[float] | None = None,

        label: str | None = None,
        color: tuple[float, float, float] |str | None = MAINCOLOR,
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