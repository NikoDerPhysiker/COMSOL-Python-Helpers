# Author: Niko Bleidistel
# last change: 2026-08-18

##############################################################################
# import packages
##############################################################################

import re
import io

import pandas as pd

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