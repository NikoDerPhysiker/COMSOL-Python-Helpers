#!/usr/bin/env python
"""Regression tests for math-mode wrapping on mixed value + model-name labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "py-helpers"))

import advanced_plotting_functions as apf


cases = [
    ("z = -3.5[μm], automatic", None, "z = $-3.5[μm]$, automatic"),
    ("z = -9.5[μm], areas", None, "z = $-9.5[μm]$, areas"),
    ("z = -3.5[μm], planes", None, "z = $-3.5[μm]$, planes"),
    ("I_spiral_terminal = -3.5", {"I_spiral_terminal": r"$I_{\mathrm{coil}}$ $[\mathrm{A}]$"}, r"$I_{\mathrm{coil}}$ = $-3.5[\mathrm{A}]$"),
    ("I_conductor_terminal = 10e-6", {"I_conductor_terminal": r"$I$ $[\mathrm{A}]$"}, r"$I$ = $10[\mathrm{\upmu}\mathrm{A}]$"),
    ("z = -3.5[μm]", {"z": r"$z$ $[\mathrm{m}]$"}, r"$z$ = $-3.5[μm]$"),
    ("N_grid = 13.0", {"N_grid": r"$N_{\mathrm{grid}}$"}, r"$N_{\mathrm{grid}}$ = $13.0$"),
    ("left_out_lines = 0", None, "left_out_lines = $0$"),
    ("left_out_lines = $0$", None, "left_out_lines = $0$"),
    ("N_grid = $13.0$", {"N_grid": r"$N_{\mathrm{grid}}$"}, r"$N_{\mathrm{grid}}$ = $13.0$"),
    ("z = 2.5", None, "z = $2.5$"),
    ("z = 2.5, automatic", None, "z = $2.5$, automatic"),
]

for raw, translation, expected in cases:
    result = apf.translate_and_prefix_label(raw, translation, bm=False)
    print(f"RAW:    {raw!r}")
    print(f"RESULT: {result!r}")
    print(f"EXPECT: {expected!r}")
    print("MATCH" if result == expected else "FAIL")
    print("-" * 60)
    if result != expected:
        raise SystemExit(1)
