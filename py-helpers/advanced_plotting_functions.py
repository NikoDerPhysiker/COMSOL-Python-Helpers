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

MAINCOLOR = (0/255, 51/255, 102/255)            # color of the title = FAU-Blau
SECONDARYCOLOR = (190/255, 205/255, 220/255)    # Light blue
BACKGROUNDCOLOR = (225/255, 225/255, 225/255)   # Light gray

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











# def get_SI_prefix(limits: tuple[float, float], use_u_as_micro: bool = False, bm: bool = False) -> tuple[str | None, int, int]:
#         """
#         Returns the SI prefix for a given numeric value.
        
#         Args:
#             limits (tuple[float, float]): A tuple containing the minimum and maximum values.
#             use_u_as_micro (bool): Whether to use 'u' as the symbol for micro or 'μ'.
#             bm (bool): Whether to use bold math formatting for the SI prefix.
        
#         Returns:
#             tuple[str|None, int, int]: A tuple containing the SI prefix, the SI exponent, and the exponent difference to the magnitude of the input values.
#         """
#         # get the maximum absolute value from the limits
#         xmax = max(abs(x) for x in limits)

#         # get magnitude of the maximum value
#         x_base_exponent = math.floor(math.log10(xmax)) if xmax > 0 else 0
        
#         # Round down to the nearest multiple of 3 for SI prefix
#         xsi_exponent = (x_base_exponent // 3) * 3 
#         exponent_diff = x_base_exponent - xsi_exponent

#         # Define the mapping of SI prefixes to their corresponding exponents
#         PREFIX_TO_EXPONENT = {
#             # Large values (positive exponents)
#             r"\mathrm{Q}": 30,   # Quetta
#             r"\mathrm{R}": 27,   # Ronna
#             r"\mathrm{Y}": 24,   # Yotta
#             r"\mathrm{Z}": 21,   # Zetta
#             r"\mathrm{E}": 18,   # Exa
#             r"\mathrm{P}": 15,   # Peta
#             r"\mathrm{T}": 12,   # Tera
#             r"\mathrm{G}": 9,    # Giga
#             r"\mathrm{M}": 6,    # Mega
#             r"\mathrm{k}": 3,    # Kilo
#             r"\mathrm{h}": 2,    # Hecto
#             r"\mathrm{da}": 1,   # Deca

#             # zero exponent (no prefix)
#             "": 0,     # No prefix
            
#             # Small values (negative exponents)
#             r"\mathrm{d}": -1,   # Deci
#             r"\mathrm{c}": -2,   # Centi
#             r"\mathrm{m}": -3,   # Milli
#             r"\mathrm{u}": -6,   # Micro (often written as µ)
#             r"\mathrm{n}": -9,   # Nano
#             r"\mathrm{p}": -12,  # Pico
#             r"\mathrm{f}": -15,  # Femto
#             r"\mathrm{a}": -18,  # Atto
#             r"\mathrm{z}": -21,  # Zepto
#             r"\mathrm{y}": -24,  # Yocto
#             r"\mathrm{r}": -27,  # Ronto
#             r"\mathrm{q}": -30,   # Quekto
#         }
#         if not use_u_as_micro:
#             # PREFIX_TO_EXPONENT["μ"] = -6   # Use 'μ' for micro instead of 'u'
#             PREFIX_TO_EXPONENT[r"\mathrm{\upmu}"] = -6   # Use 'μ' for micro instead of 'u'
#             del PREFIX_TO_EXPONENT[r"\mathrm{u}"]    # Remove 'u' from the dictionary

#         # Returns the key, or None if the value doesn't exist
#         siprefix = next((k for k, v in PREFIX_TO_EXPONENT.items() if v == xsi_exponent), None)
#         if bm and siprefix is not None:
#             siprefix = r"\bm{" + siprefix + "}"  # Add bold math formatting to the SI prefix
#         return siprefix, xsi_exponent, exponent_diff

# ##############################################################################

# def prefixes_notation(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, axis: Literal['x', 'y']):
#     """
#     Adjusts the axis labels of a matplotlib plot to use SI prefixes based on the data limits.

#     Args:
#         fig (matplotlib.figure.Figure): The matplotlib figure object.
#         ax (matplotlib.axes.Axes): The matplotlib axes object.
#         axis (Literal['x', 'y']): The axis to adjust ('x' or 'y').
    
#     Returns:
#         tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, str|None, int]: A tuple containing the updated figure and axes objects, the SI prefix used, and the SI exponent.
#     """
#     fig.canvas.draw() 
#     if axis == 'x':
#         xlimits = ax.get_xlim()
#         siprefix, xsi_exponent, exponent_diff = get_SI_prefix(xlimits)
#         scale_factor = 10 ** (-xsi_exponent)
#         ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * scale_factor:g}"))
#     elif axis == 'y':
#         ylimits = ax.get_ylim()
#         siprefix, xsi_exponent, exponent_diff = get_SI_prefix(ylimits)
#         scale_factor = 10 ** (-xsi_exponent)
#         ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y * scale_factor:g}"))
#     return fig, ax, siprefix, xsi_exponent

# ##############################################################################

# def _contains_prefix_in_unit(unit_text: str, prefix: str) -> bool:
#     """Return True only when the unit already includes a real SI prefix, not just the base unit itself."""
#     if not unit_text or not prefix:
#         return False

#     def normalize(value: str) -> str:
#         value = value.strip().replace("$", "")
#         value = value.replace(r"\mathrm{", "")
#         value = value.replace(r"\text{", "")
#         value = value.replace(r"\bm{", "")
#         value = value.replace(r"\upmu", "μ").replace(r"\mu", "μ")
#         value = value.replace("{", "").replace("}", "")
#         value = value.replace("[", "").replace("]", "")
#         value = value.replace("(", "").replace(")", "")
#         return value.strip()

#     norm_unit = normalize(unit_text)
#     norm_prefix = normalize(prefix)
#     if not norm_prefix:
#         return False
#     if norm_unit == norm_prefix:
#         return False
#     return norm_unit.startswith(norm_prefix) and len(norm_unit) > len(norm_prefix) 
#     # Return True only if the unit starts with the prefix and is longer than the prefix itself, 
#     # indicating that it's a real SI prefix and not just the base unit. 


# def set_prefix_in_label(string: str, prefix: str):
#     """Insert the SI prefix into the first bracketed unit block without duplicating it on repeated calls."""
#     if not string or not prefix:
#         return string

#     # The regex pattern matches either a square bracketed block [ ... ] or a round bracketed block ( ... ).
#     match = re.search(r"\[[^\]]*\]|\([^)]*\)", string)

#     # if no match is found, append the prefix in square brackets at the end of the string
#     if match is None:
#         return f"{string} [{prefix}1]"

#     # If a match is found, check if the unit already contains the prefix
#     full_match = match.group(0)
#     inner = full_match[1:-1]
#     if _contains_prefix_in_unit(inner, prefix):
#         return string # Return the original string if the unit already contains the prefix

#     # If the unit does not contain the prefix, insert the prefix at the beginning of the unit
#     updated = full_match.replace(inner, f"{prefix}{inner}", 1)
#     return string[:match.start()] + updated + string[match.end():]

# ##############################################################################

# def _insert_prefix_into_unit(unit_text: str, prefix: str) -> str:
#     """Insert the SI prefix into the unit text, preserving any existing brackets and formatting."""
#     if not unit_text or not prefix:
#         return unit_text

#     # Remove leading and trailing whitespace
#     value = unit_text.strip() 

#     # If the unit is already wrapped in a single math block, strip the outer $...$ for processing
#     if value.startswith("$") and value.endswith("$"): 
#         value = value[1:-1].strip()

#     # If the unit is already wrapped in brackets, insert the prefix inside the brackets
#     if value.startswith("[") and value.endswith("]"):
#         return f"[{prefix}{value[1:-1]}]"
#     if value.startswith("(") and value.endswith(")"):
#         return f"({prefix}{value[1:-1]})"

#     # If the unit is not wrapped in brackets, simply insert the prefix at the beginning of the unit text
#     return f"[{prefix}{value}]"


# def set_prefix_in_number_unit_string(string: str, bm: bool = False) -> str:
#     """
#     Insert the appropriate SI prefix into a string containing a numeric value and an optional unit, while preserving any existing LaTeX math formatting.
#     1. Rule: If the string does not contain an '=', return the original string unchanged.
#     2. Rule: Else split at the first '=' and process the right part to identify a numeric value and an optional unit.
#     3. Rule: If a numeric value is found and already wrapped in a single math block, return the original string unchanged.
#     4. Rule: If no unit is found, wrap the numeric value in a math block and return the updated string.
#     5. Rule: If a unit and a value is found, determine the appropriate SI prefix based on the magnitude of the number.
#     6. Rule: If there is no suitable SI prefix or if the unit already contains the prefix, return the original string unchanged.
#     7. Rule: Else insert the SI prefix into the unit text, preserving any existing brackets and formatting.
#     8. Rule: If the original string was wrapped in a math block, wrap the updated string in a math block as well.
#     9. Rule: If there is a trailing comma (+ optional text) after the numeric value, place the math block before the comma and keep the trailing text outside the math block. 
#     """
#     if '=' not in string:
#         return string

#     # Split the string into left and right parts at the first '=', and strip leading/trailing whitespace from both parts
#     left, right = string.split('=', 1) 
#     left = left.strip()
#     value_part = right.strip()

#     # check if its a math block
#     if (value_part.startswith("$") and value_part.endswith("$")):
#         inner = value_part[1:-1]
#         # if there is no math block inside, return the string as is 
#         if "$" not in inner: 
#             return string
#         else: # if there is a math block inside, flag it
#             outer_math_wrapped = False  

#     # searches two parts:
#     # 1. a number (with optional sign, decimal point, and scientific notation)
#     # 2. an optional unit in brackets (square or round) and/or in math mode "$...$"
#     match = re.search(
#         r"(?P<number>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?P<unit>\$\[[^\]]*\]\$|\[[^\]]*\]|\$\([^)]*\)\$|\([^)]*\))?",
#         value_part,
#     )

#     # If no match is found or if the number part is empty, return the original string
#     if match is None or match.group("number") == "":
#         return string

#     if match.start() > 0 and value_part[match.start() - 1] == "$" and match.end() < len(value_part) and value_part[match.end()] == "$":
#         # If the number is already wrapped in a single math block, return the original string without further processing, 
#         # to avoid creating display style math blocks with double "$$...$$".
#         return string
#     if match.start() == 1 and value_part.startswith("$") and value_part.endswith("$") and "$" not in value_part[1:-1]:
#         return string

#     # Extract the matched number and unit, convert the number to a float
#     number_str = match.group("number")
#     unit_text = match.group("unit")
#     number = float(number_str)

#     # If no unit is found, wrap the number in a math block and return the updated string, keeping any trailing comma outside the math block.
#     if unit_text is None:
#         replacement = f"${number_str}$"
#         updated_value_part = value_part[:match.start()] + replacement + value_part[match.end():]
#         updated_value_part = updated_value_part.strip() # remove any leading/trailing whitespace from the updated value part

#         if match.start() == 0 and value_part[match.end():].strip().startswith(","):
#             # Keep a trailing comma outside the math block, e.g. 'z = $-3.5$, automatic'
#             updated_value_part = replacement + value_part[match.end():].strip()
#         return f"{left} = {updated_value_part}".strip()

#     # If a unit is found, determine the appropriate SI prefix 
#     siprefix, xsi_exponent, _ = get_SI_prefix((number, number), bm=bm)
#     if siprefix is None or _contains_prefix_in_unit(unit_text, siprefix):
#         # if no suitable SI prefix is found or if the unit already contains the prefix, return the original string
#         return string

#     # Scale the number according to the SI prefix exponent and insert the prefix into the unit text
#     scaled_number = number * (10 ** (-xsi_exponent))
#     scaled_str = f"{scaled_number:g}"
#     replacement = f"{scaled_str}{_insert_prefix_into_unit(unit_text, siprefix)}"
#     updated_value_part = value_part[:match.start()] + replacement + value_part[match.end():]
#     updated_value_part = updated_value_part.strip()

#     # If the original string was wrapped in a math block, wrap the updated string in a math block as well.
#     if outer_math_wrapped:
#         updated_value_part = f"${updated_value_part}$"

#     # If the number is at the start of the value part and there is trailed by optional whitespace and a comma, 
#     # create a math block and append the trailing comma and text outside the math block, e.g. 'z = $-3.5[μm]$, automatic' 
#     elif match.start() == 0 and value_part[match.end():].strip().startswith(","):
#         updated_value_part = f"${replacement}$" + value_part[match.end():].strip()

#     return f"{left} = {updated_value_part}".strip()

# ##############################################################################
# ##############################################################################

# def get_formula_unit(string: str) -> tuple[str, str]:
#     """Extract the math symbol and any trailing unit from a label-like string."""
#     value = string.strip()
#     if not value:
#         return "", ""

#     unit = ""
#     formula_search_part = value

#     # Strip a final bracketed unit only when it belongs to the label tail, not to the math formula itself.
#     unit_match = re.search(r"(?P<unit>\$\[[^\]]*\]\$|\[[^\]]*\]|\$\([^)]*\)\$|\([^)]*\))\s*$", value)
#     if unit_match:
#         unit_text = unit_match.group("unit").strip()
#         if unit_text.startswith("$") and unit_text.endswith("$") and "$" not in unit_text[1:-1]:
#             unit = unit_text[1:-1].strip()
#         else:
#             unit = unit_text.strip("[]() ")
#         formula_search_part = value[:unit_match.start()].strip()

#     if not formula_search_part:
#         return "", unit.strip()

#     # Preserve translated math blocks exactly as written.
#     math_block_match = re.search(r"\$[^$]+\$", formula_search_part)
#     if math_block_match:
#         return math_block_match.group(0).strip(), unit.strip()

#     plain = formula_search_part.strip("[]() ").strip()
#     return plain.strip(), unit.strip()


# ##############################################################################

# def clean_label_text(text: str) -> str:
#     """Strip accidental outer math wrappers from pure text fragments while leaving numeric label values untouched."""
#     # If the text is empty or None, return it as is
#     value = (text or "").strip()
#     if not value:
#         return value

#     # If the text is wrapped in a single math block and does not contain any other math blocks inside, strip the outer $...$.
#     if value.startswith("$") and value.endswith("$") and "$" not in value[1:-1]:
#         value = value[1:-1].strip()

#     # If the text does not contain an '=', return it as is
#     if "=" not in value:
#         return value

#     # If the text contains an '=', split it into left and right parts, and check if the right part is a numeric value. 
#     # If it is, return the original text; otherwise, return the cleaned text with the '=' preserved.
#     left, right = value.split("=", 1)
#     left = left.strip()
#     right = right.strip()
#     if re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", right):
#         return value
#     return f"{left} = {right}".strip()


# def translate_and_prefix_label(label: str, translation_dict: dict[str, str] | None = None, bm: bool = False) -> str:
#     """Translate the label and apply SI prefixing without disturbing LaTeX math blocks."""
#     if '=' not in label:
#         translated = label
#         if translation_dict is not None and label in translation_dict:
#             translated = translation_dict[label]
#         return clean_label_text(translated).strip()

#     left, right = label.split('=', 1)
#     variable = left.strip()
#     rest = right.strip()

#     if translation_dict is not None and variable in translation_dict:
#         variable = translation_dict[variable]

#     explicit_math_label = bool(re.search(r"\$.*?\$", variable or ""))

#     if explicit_math_label:
#         unit_match = re.search(r"(?P<unit>\$\[[^\]]*\]\$|\[[^\]]*\]|\$\([^)]*\)\$|\([^)]*\))\s*$", variable.strip())
#         if unit_match:
#             unit_text = unit_match.group("unit").strip()
#             if unit_text.startswith("$") and unit_text.endswith("$") and "$" not in unit_text[1:-1]:
#                 unit_text = unit_text[1:-1].strip()
#             variable = variable[:unit_match.start()].rstrip()
#             if rest and not re.search(r"\[[^\]]*\]|\([^)]*\)", rest):
#                 rest = f"{rest}{unit_text}".strip()
#         result = f"{variable.strip()} = {rest}".strip()
#     else:
#         short, unit = get_formula_unit(variable)
#         if not short:
#             short = variable.strip()

#         if unit and not re.search(r"\[[^\]]*\]|\([^)]*\)", rest):
#             rest = f"{rest} [{unit}]".strip()

#         result = f"{short} = {rest}".strip()

#     result = set_prefix_in_number_unit_string(result, bm=bm).strip()

#     def _is_numeric_value_only(text: str) -> bool:
#         """Return True only for a true numeric value expression, not for mixed text like '..., automatic'."""
#         value = text.strip()
#         if not value:
#             return False
#         if value.startswith("$") and value.endswith("$"):
#             value = value[1:-1].strip()
#         unit_pattern = r"(?:\$\[[^\]]*\]\$|\[[^\]]*\]|\$\([^)]*\)\$|\([^)]*\))?"
#         numeric_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
#         return bool(re.fullmatch(rf"{numeric_pattern}\s*{unit_pattern}", value))

#     if '=' in result:
#         left_expr, right_expr = result.split('=', 1)
#         left_expr = left_expr.strip()
#         right_expr = right_expr.strip()

#         # Only wrap if the whole right-hand side is a single numeric value block.
#         # Mixed text like 'z = -3.5[μm], automatic' must remain plain text.
#         # Do not add another $...$ layer when the value is already a single math block.
#         if right_expr and _is_numeric_value_only(right_expr):
#             if not (right_expr.startswith("$") and right_expr.endswith("$") and "$" not in right_expr[1:-1]):
#                 right_stripped = right_expr
#                 if right_stripped.startswith("$") and right_stripped.endswith("$"):
#                     right_stripped = right_stripped[1:-1].strip()
#                 if not right_stripped.startswith("$") and not right_stripped.endswith("$"):
#                     result = f"{left_expr} = ${right_stripped}$".strip()

#     return result











# =============================================================================
# Regex building blocks (compiled once, reused at every call site)
# =============================================================================

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

        ax.set_title(rf"\textbf{{{main_title}}}"+"\n\n"+rf"{{{subtitle}}}", pad=20)

    ax.set_axisbelow(True)  # Ensure grid is below other plot elements
    ax.grid(True, which='both', linestyle='--', zorder=0)
    
    return fig, ax

##############################################################################
##############################################################################

def dynamic_legend(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    fraction: float = 1.0,
    width_tolerance: float = 1.30, # = 130% of the figure width
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
    if n_entries % (max_columns+1) == 0:
        max_columns = max_columns + 1
    test_ncol_range = [ncol for ncol in test_ncol_range if ncol <= max_columns]

    # get the figure width (= total width of the figure including axes labels, etc.) in points and calculate the allowed maximum width for the legend
    fig_width_pts = fig.get_size_inches()[0] * 72
    allowed_max_width_pts = fig_width_pts * width_tolerance

    # Vorhandene Legende entfernen, damit sie die Messung nicht verfaelscht
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