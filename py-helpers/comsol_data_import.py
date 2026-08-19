# Author: Niko Bleidistel
# last change: 2026-08-18

##############################################################################
# import packages
##############################################################################

import re
import io

import pandas as pd
import numpy as np

import importlib
# custom packages
import time_logging as tl          
importlib.reload(tl)



##############################################################################
##############################################################################
# importing functions
##############################################################################
##############################################################################


def read_comsol_export(file_path: str):
    """
    Reads a COMSOL data export file and extracts header information into variables and the numerical data into a pandas DataFrame.
    The function processes the file line by line, distinguishing between header lines (starting with '%') and data lines.
    
    Note: It definitely works for the '.txt'-format. Other Filetypes were not tested.  

    Args:
        file_path (str): The path to the COMSOL data export file.

    Returns:
        (dict, pandas.DataFrame): A tuple containing the header information as key-value pairs and the numerical data as a pandas DataFrame.

    Raises:
        ValueError: If the file_path is not a string.
    """
    if not isinstance(file_path, (str)):
        raise ValueError("file_path must be a string representing the path to the COMSOL data export file.")

    tl.log_message(f"Reading COMSOL export file: {file_path}")
    
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

    tl.log_message(f"COMSOL export file read successfully: {file_path}")
    return header_data, df

##############################################################################
##############################################################################

def round_to_6_sig_digits(series: pd.Series) -> pd.Series:
    """
    Rounds a numeric pandas Series to exactly 6 significant digits.
    """
    valid_series = series.dropna()
    if valid_series.empty:
        return series
    abs_series = valid_series.abs()
    # If all non-NaN values are 0, return a series filled with 0.0
    if (abs_series == 0).all():
        return pd.Series(0.0, index=series.index)
    # Handle zeros carefully to avoid log10 infinity errors
    with np.errstate(divide="ignore"):
        exponent = np.where(abs_series > 0, np.floor(np.log10(abs_series)), 0)
    # 5 minus the exponent gives exactly 6 significant digits
    decimals = 5 - exponent.astype(int)
    # Compute rounded values while strictly preserving the DataFrame indices
    rounded_list = [
        round(val, max(0, dec)) if pd.notna(val) else np.nan
        for val, dec in zip(valid_series, decimals)
    ]
    return pd.Series(rounded_list, index=valid_series.index)

##############################################################################

def find_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies numeric columns in a DataFrame that contain only a single constant
    value within the first 25 rows, ignoring NaN values and checking up to 6
    significant digits. Non-numeric columns are skipped.

    Returns:
        pd.DataFrame: A DataFrame with two columns: "Constants" and "Value",
                      containing the names of constant columns and their
                      corresponding constant values.
    """
    # Look only at the first 25 rows
    df_sample = df.head(25)
    constants_dict = {}
    for col in df_sample.columns:
        series = df_sample[col]
        # Skip non-numeric columns safely (handles StringDtype, object, datetime, etc.)
        if not pd.api.types.is_numeric_dtype(series.dtype):
            continue
        # Extract the series and explicitly drop NaN values
        valid_series = series.dropna()
        # Skip columns that are completely empty or contain only NaNs
        if valid_series.empty:
            continue
        # Round the remaining non-NaN values to 6 significant digits
        rounded_series = round_to_6_sig_digits(valid_series)
        # Check if all remaining valid values are identical after rounding
        if rounded_series.dropna().nunique() == 1:
            # Safely extract the first element as the scalar representation
            constants_dict[col] = rounded_series.dropna().iloc[0]
    # Construct the final resulting DataFrame
    return pd.DataFrame(
        list(constants_dict.items()), columns=["Constants", "Value"]
    )
        

##############################################################################
##############################################################################