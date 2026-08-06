# Author: Niko Bleidistel
# last change: 2026-08-04

##############################################################################
# import packages
##############################################################################

import math

import mph                              # api for comsol multiphysics

import numpy as np                     # numpy for numerical operations

import pandas as pd                     # pandas for easy DataAnalysis
from IPython.display import display     # for displaying DataFrames

from os import makedirs                 # for creating directories
from os import devnull                  # for redirecting stdout to null device
import contextlib                       # for redirecting stdout

import pathlib                          # for handling file paths
import csv                              # for handling csv files

import re                               # for regular expressions

import importlib

#custom packages
import time_logging as tl               # for logging time of function calls (custom python file)
importlib.reload(tl)

##############################################################################
##############################################################################
# functions
##############################################################################
##############################################################################

LINELENGTH = 69

def printLine():
    print('-' * LINELENGTH)

def printDoubleLine():
    print('=' * LINELENGTH)

##############################################################################
##############################################################################

def import_models(MODELNAME_LIST: list[str],
                  client: mph.Client | None = None, 
                  printFeedback: bool = True,
                  avoid_reimporting: bool = True,
                  path_to_models: str | None = None,
                  ):
    """
    Imports comsol multiphysics models from the current working directory.
    
    Args:
        MODELNAME_LIST (list of strg): A list of model names (without .mph extension) to be imported.
        client (mph.Client, optional): An optional existing mph.Client instance. If None, a new client will be created.
        printFeedback (bool): If True, prints the names of the loaded models.
        avoid_reimporting (bool): If True, avoids reimporting models that have already been loaded.
        path_to_models (str, optional): The path to the directory containing the model files. If None, the current working directory will be used.

    Returns:
       (mph.Client, list): A tuple containing the mph.Client instance used to load the models and a list of loaded mph.Model instances corresponding to the provided model names.

    Raises:
        KeyError: If MODELNAME_LIST is not provided.
        TypeError: If any of the arguments have incorrect types.
    """

    input_path = pathlib.Path(path_to_models) if path_to_models is not None else None

    # manage model (following: https://mph.readthedocs.io/en/1.3/tutorial.html)
    if client is None:
        client = mph.start()
        tl.log_message('New comsol multiphysics client started.')
    elif not isinstance(client, mph.Client):
        raise TypeError("The argument 'client' needs to be an instance of 'mph.Client' or None.")
    
    # enable check if a model has been previously loaded
    if avoid_reimporting:
        client.caching(True)

    # check if MODELNAME_LIST was given 
    if MODELNAME_LIST is None or not isinstance(MODELNAME_LIST, list) or len(MODELNAME_LIST) == 0:
        raise KeyError('No List of modelnames was given.')
    
    # set path to models if given
    if input_path is not None:
        model_paths = [input_path / (modelname + '.mph') for modelname in MODELNAME_LIST]
    else:
        model_paths = [modelname + '.mph' for modelname in MODELNAME_LIST]
    
    
    # load all Models in MODELNAME_LIST
    model_list=[]
    for path in model_paths:
        model = client.load(path)
        model_list.append(model)
        tl.log_message(f'Model "{model.name()}" loaded from path: {path}')

    # print Feedback if wanted
    if printFeedback:
        names = client.names()
        printDoubleLine()
        print('Following comsol multiphysics models are now loaded:')
        printLine()
        for idx, name in enumerate(names):
            print(f'{idx}:  ' + name)
        printDoubleLine()

    return client, model_list

##############################################################################
##############################################################################

def print_model_parameters(model: mph.Model, doPrint=True):
    """
    Returns (and prints) a list of parameters for a given comsol multiphysics model.

    Args:
        model (mph.Model): An instance of mph.Model for which the parameters should be printed.
        doPrint (bool, optional): If True, prints the parameters to the console. Default is True.

    Returns:
        (list): A list of strings, each containing the description, name, and value of a parameter in the format "description    name = value".

    Raises:
        TypeError: If the provided model is not an instance of mph.Model.
    """

    # check model instance
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    
    parameters = []
    for (name, value) in model.parameters().items(): # type: ignore
        description = model.description(name) # type: ignore

        string = f'{description:20} {name} = {value}'
        parameters.append(string)
        if doPrint:
            print(string)

    return parameters

##############################################################################

def print_model_info(model: mph.Model,
                     printParameters = True,
                     printMaterials = True,
                     printPhysics = True,
                     printStudies = True,
                     doPrint = True,
                     ):
    """
    Prints information about a given comsol multiphysics model, including its parameters, materials, physics, and studies.
    
    Args:
        model (mph.Model): An instance of mph.Model for which the information should be printed.
        printParameters (bool, optional): If True, prints the parameters of the model. Default is True.
        printMaterials (bool, optional): If True, prints the materials used in the model. Default is True.
        printPhysics (bool, optional): If True, prints the physics interfaces used in the model. Default is True.
        printStudies (bool, optional): If True, prints the studies defined in the model. Default is True.
        doPrint (bool, optional): If True, prints the information to the console. Default is True.

    Returns:
        (list): A list of strings containing the printed information about the model.

    Raises:
        TypeError: If the provided model is not an instance of mph.Model.
    """
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    print(model.name())
    output = []
    if printParameters:
        printDoubleLine()
        print('Parameters:')
        printLine()
        output.extend(print_model_parameters(model, doPrint=doPrint))
        printDoubleLine()
        print()

    if printMaterials:
        # create string
        string = ''
        printComma = False
        for material in model.materials():
            if not printComma: # skip comma before first material
                string += material
                printComma=True
            else:
                string += ', ' + material
        
        # print
        printDoubleLine()
        printstring = f'Materials:\t{string}'
        output.extend(printstring)
        print(printstring)
        printDoubleLine()
        print()

    if printPhysics:
        # create string
        string = ''
        printComma = False
        for physic in model.physics():
            if not printComma: # skip comma before first material
                string += physic
                printComma=True
            else:
                string += ', ' + physic
        # print
        printDoubleLine()
        printstring = f'Physics:\t{string}'
        output.extend(printstring)
        print(printstring)
        printDoubleLine()
        print()

    if printStudies:
        # create string
        string = ''
        printComma = False
        for study in model.studies():
            if not printComma: # skip comma before first material
                string += study
                printComma=True
            else:
                string += ', ' + study
        # print
        printDoubleLine()
        printstring = f'Studies:\t{string}'
        output.extend(printstring)
        print(printstring)
        printDoubleLine()
        print()

    return output

##############################################################################
##############################################################################

def save_Parameter_List_to_CSV(
                            model: mph.Model, 
                            csv_path: str, 
                            displayParams = True,
                            ):
    """
    Saves the parameters of a given comsol multiphysics model to a CSV file and optionally displays them as a DataFrame.
    
    Args:
        model (mph.Model): An instance of mph.Model for which the parameters should be saved.
        csv_path (str): The file path where the CSV file will be saved.
        displayParams (bool, optional): If True, displays the parameters as a DataFrame. Default is True.
    
    Raises:
        TypeError: If the provided model is not an instance of mph.Model.
        ValueError: If csv_path is None.
    """
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    
    # get old parameters
    names = []
    values = []
    descriptions = [] 

    for (name, value) in model.parameters().items(): # type: ignore
        description = model.description(name) # type: ignore
        names.append(name)
        values.append(value)
        descriptions.append(description)

    params = {
        'name': names,
        'value': values,
        'description': descriptions
        }
    
    df_params = pd.DataFrame(params)
    df_params.to_csv(csv_path, index=False)
    tl.log_message(f'Parameters of model "{model.name()}" saved to CSV file: {csv_path}')

    if displayParams:
        display(df_params)

##############################################################################    

def find_difference_of_DataFrames(df_old_params: pd.DataFrame, df_new_params: pd.DataFrame):
    """
    Compares two DataFrames containing parameters and identifies added, removed, and changed parameters based on their 'name' column.

    Args:
        df_old_params (pd.DataFrame): The DataFrame containing the old parameters with columns 'name', 'value', and 'description'.
        df_new_params (pd.DataFrame): The DataFrame containing the new parameters with columns 'name', 'value', and 'description'.

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame): A tuple containing four DataFrames: 
            1. DataFrame containing the parameters that were **removed** ('name' present in old but not in new).
            2. DataFrame containing the parameters that were **added** ('name' present in new but not in old).
            3. DataFrame containing the parameters that were **changed** ('name' present in both but with different 'value' or 'description').
            4. DataFrame containing the new values and descriptions of the changed parameters.

    Raises:
        TypeError: If either of the provided arguments is not an instance of pd.DataFrame.
    """
    if not isinstance(df_old_params, pd.DataFrame):
        raise TypeError("The argument 'df_old_params' needs to be an instance of 'pd.DataFrame'.")
    if not isinstance(df_new_params, pd.DataFrame):
        raise TypeError("The argument 'df_new_params' needs to be an instance of 'pd.DataFrame'.")
    
    # Fill NaN values with an empty string to ignore them during comparison
    df_old_params = df_old_params.fillna('')
    df_new_params = df_new_params.fillna('') 

    # 1. Find added and removed parameters by comparing the 'name' columns
    old_names = set(df_old_params['name'])
    new_names = set(df_new_params['name'])

    added_names = new_names - old_names
    removed_names = old_names - new_names

    # Filter the dataframes to show the full rows for added/removed items
    df_added = df_new_params[df_new_params['name'].isin(added_names)]
    df_removed = df_old_params[df_old_params['name'].isin(removed_names)]

    # 2. Find changed parameters (only for names that exist in both DataFrames)
    common_names = old_names.intersection(new_names)

    # Filter both DataFrames to keep only the shared parameter names and sort them
    df_old_common = df_old_params[df_old_params['name'].isin(common_names)].sort_values(by='name').reset_index(drop=True)
    df_new_common = df_new_params[df_new_params['name'].isin(common_names)].sort_values(by='name').reset_index(drop=True)

    # Use compare() to see side-by-side differences for value or description changes
    if not df_old_common.empty:
        # set multiindex with original index and name to keep the original index and also have the name as index for comarison
        # df_old_common.set_index([df_old_common.index, "name"], inplace=True)
        # df_new_common.set_index([df_new_common.index, "name"], inplace=True)
        df_old_common.set_index(["name"], append=True, inplace=True)
        df_new_common.set_index(["name"], append=True, inplace=True)

        # comparison
        df_changed = df_old_common.compare(df_new_common)

        # retset index and get the 'name' column back for better readability
        df_changed = df_changed.reset_index(level="name")
        # display(df_changed) # for debugging

        # rename columns for better clarity
        df_changed.rename(columns={'self': 'old', 'other': 'new'}, level=1, inplace=True)
        
        df_changed_to = df_new_common.loc[df_changed.index].reset_index()
    return df_removed, df_added, df_changed, df_changed_to

##############################################################################

def print_different_Parameters(df_old_params=None, df_new_params=None, df_removed=None, df_added=None, df_changed=None):
    """
    Prints the differences between two sets of parameters. Either by directly providing the DataFrames of removed, added, and changed parameters or by calculating them from the original DataFrames.

    Args:
        df_old_params (pd.DataFrame, optional): The DataFrame containing the old parameters with columns 'name', 'value', and 'description'.
        df_new_params (pd.DataFrame, optional): The DataFrame containing the new parameters with columns 'name', 'value', and 'description'.
        df_removed (pd.DataFrame, optional): A DataFrame containing the parameters that were removed (present in old but not in new).
        df_added (pd.DataFrame, optional): A DataFrame containing the parameters that were added (present in new but not in old).
        df_changed (pd.DataFrame, optional): A DataFrame containing the parameters that were changed (present in both but with different 'value' or 'description').

    Raises:
        TypeError: If the provided arguments are not of the expected types.
        KeyError: If neither the complete DataFrames nor all three kinds of changes are provided.
    """
    def _print_helper(df_removed: pd.DataFrame, df_added: pd.DataFrame, df_changed: pd.DataFrame):
        """
        Helper function to print the removed, added, and changed parameters in a clear format.
        
        Args:
            df_removed (pd.DataFrame): A DataFrame containing the parameters that were removed.
            df_added (pd.DataFrame): A DataFrame containing the parameters that were added.
            df_changed (pd.DataFrame): A DataFrame containing the parameters that were changed.
        """
        printDoubleLine()

        print("Removed Parameters (Name)")
        display(df_removed if not df_removed.empty else "None")
        
        printLine()

        print("Added Parameters (Name)")
        display(df_added if not df_added.empty else "None")

        printLine()

        print("Changed Value and/or Description")
        df_changed_display = df_changed.fillna("-")  # Replace not changed values with "-" for better readability
        display(df_changed_display if not df_changed_display.empty else "None")

        printDoubleLine()

    if df_old_params is not None and df_new_params is not None:
        if not isinstance(df_old_params, pd.DataFrame):
            raise TypeError("The argument 'df_old_params' needs to be an instance of 'pd.DataFrame'.")
        if not isinstance(df_new_params, pd.DataFrame):
            raise TypeError("The argument 'df_new_params' needs to be an instance of 'pd.DataFrame'.")
        
        df_removed, df_added, df_changed, df_changed_to = find_difference_of_DataFrames(df_old_params, df_new_params)
        _print_helper(df_removed, df_added, df_changed)

    elif df_removed is not None and df_added is not None and df_changed is not None:
        if not isinstance(df_removed, pd.DataFrame):
            raise TypeError("The argument 'df_removed' needs to be an instance of 'pd.DataFrame'.")
        if not isinstance(df_added, pd.DataFrame):
            raise TypeError("The argument 'df_added' needs to be an instance of 'pd.DataFrame'.")
        if not isinstance(df_changed, pd.DataFrame):
            raise TypeError("The argument 'df_changed' needs to be an instance of 'pd.DataFrame'.")
        
        _print_helper(df_removed, df_added, df_changed)
    else:
        raise KeyError('Either both complete Dataframes or all three kinds of changes are needed. If for example there are no Parameters added and you want to use three seperate changes, just provide an empty DataFrame for the added parameters.')

##############################################################################
  
def set_DataFrame_as_Parameters(model:mph.Model, df: pd.DataFrame, isRemove=False):
    """
    Sets the (root-)parameters of a given comsol multiphysics model based on the provided DataFrame.
    
    Args:
        model (mph.Model): An instance of mph.Model for which the parameters should be set.
        df (pd.DataFrame): A DataFrame containing the parameters to be set, with columns 'name', 'value', and 'description'.
        isRemove (bool): A flag indicating whether to remove the elements of dataframe in column 'name' from the model.

    Raises:
        TypeError: If the provided model is not an instance of mph.Model or if the provided DataFrame is not an instance of pd.DataFrame.

    """

    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The argument 'df' needs to be an instance of 'pd.DataFrame'.")
    
    for row in df[['name', 'value', 'description']].itertuples(index=False):
        if isRemove: 
            # removing a parameter is not directly supported by the mph.Model API, so we need to access the underlying Java object to remove it.
            try:
                model.java.param().remove(row.name)
            except Exception:
                pass # ignore exceptions since it still works
        else:
            model.parameter(row.name, row.value) # type: ignore
            model.description(row.name, row.description) # type: ignore

    tl.log_message(f'Parameters of model "{model.name()}" updated. Removed: {isRemove}.')

##############################################################################

def set_Parameter_List_from_CSV(model:mph.Model, 
                            csv_path: str, 
                            displayOldParams = False,
                            displayNewParams = False,
                            printFeedback = True,
                            ):
    """
    Sets the parameters of a given comsol multiphysics model based on the provided CSV file and optionally displays the old and new parameters as DataFrames.
    
    Args:
        model (mph.Model): An instance of mph.Model for which the parameters should be set.
        csv_path (str): The file path of the input CSV file containing the parameters, with columns 'name', 'value', and 'description'.
        displayOldParams (bool, optional): If True, displays the old parameters as a DataFrame. Default is False.
        displayNewParams (bool, optional): If True, displays the new parameters as a DataFrame. Default is False.
        printFeedback (bool, optional): If True, prints feedback about the changes made to the parameters. Default is True.

    Raises:
        TypeError: If the provided model is not an instance of mph.Model or if the provided csv_path is not a string.
        ValueError: If csv_path is None.
    """
    # check model instance
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    if not isinstance(csv_path, str):
        raise TypeError("The argument 'csv_path' needs to be a string.")
    
    # get old parameters
    old_names = []
    old_values = []
    old_descriptions = [] 

    for (name, value) in model.parameters().items(): # type: ignore
        description = model.description(name) # type: ignore
        old_names.append(name)
        old_values.append(value)
        old_descriptions.append(description)

    old_params = {
        'name': old_names,
        'value': old_values,
        'description': old_descriptions
        }
    
    df_old_params = pd.DataFrame(old_params)

    if displayOldParams:
        print('Old Parameters:')
        display(df_old_params)

    # get new parameters
    df_new_params = pd.read_csv(csv_path)

    if displayNewParams:
        print('New Parameters:')
        display(df_new_params)

    # compare old and new parameters

    # Sort both DataFrames by the 'name' column and reset the index to align them
    df_old_sorted = df_old_params.sort_values(by='name').reset_index(drop=True)
    df_new_sorted = df_new_params.sort_values(by='name').reset_index(drop=True)
    # Fill NaN values with an empty string to ignore them during comparison
    df_old_sorted = df_old_params.fillna("")
    df_new_sorted = df_new_params.fillna("")

    # Compare the sorted DataFrames
    parameters_match = df_old_sorted.equals(df_new_sorted)

    if parameters_match:
        print('Attention: Old and new parameters match. Ignore if intentionally.')
    else:
        df_removed, df_added, df_changed, df_changed_to = find_difference_of_DataFrames(df_old_params, df_new_params)
        set_DataFrame_as_Parameters(model, df_removed, isRemove=True)
        set_DataFrame_as_Parameters(model, df_added, isRemove=False)
        set_DataFrame_as_Parameters(model, df_changed_to, isRemove=False)

        if printFeedback:
            print_different_Parameters(df_removed=df_removed, df_added=df_added, df_changed=df_changed)

    tl.log_message(f'Parameters of model "{model.name()}" updated from CSV file: {csv_path}.')

##############################################################################
##############################################################################

def Comsol_TXT_to_CSV(txt_file_path: str, csv_file_path: str, printFeedback = True):
    """ 
    Converts a TXT file formatted for import/load into the COMSOL Multiphysics Software into a CSV file containing parameters.
    
    Args:
        txt_file_path (str):    The file path of the input TXT file containing the parameters, with lines formatted as "name value "description"".
        csv_file_path (str):    The file path where the output CSV file will be saved, with columns 'name', 'value', and 'description'.
        printFeedback (bool):   If True, prints a success message after the conversion is completed.

    Raises:
        TypeError:  If either of the provided file paths is not a string.
    """

    if not isinstance(txt_file_path, str):
        raise TypeError("The argument 'txt_file_path' needs to be a string.")
    if not isinstance(csv_file_path, str):
        raise TypeError("The argument 'csv_file_path' needs to be a string.")

    # Regex pattern to capture: name, value+unit, and description inside quotes
    # Example line: U 1[V] "applied voltage"
    # ^ - start of line
    # (\S+) - capture name (non-whitespace characters)
    # \s+ - one or more whitespace characters
    # (\S+) - capture value+unit (non-whitespace characters)
    # \s+ - one or more whitespace characters
    # "([^"]+)" - capture description inside quotes (any characters except quotes)
    pattern = re.compile(r'^(\S+)\s+(\S+)\s+"([^"]+)"')

    rows = []

    # open the text file as read-only and close it automatically after the block
    with open(txt_file_path, "r", encoding="utf-8") as txt_file: 
        
        # read each line
        for line in txt_file:
            line = line.strip() # strip whitespace
            
            #skip empty lines
            if not line:
                continue

            match = pattern.match(line) # apply regex pattern to the line
            if match:
                name, value, description = match.groups()
                rows.append(
                    {
                        "name": name,
                        "value": value,
                        "description": description,
                    }
                )

    # write the extracted data to a CSV file
    with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["name", "value", "description"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    message = f"TXT file '{txt_file_path}' successfully converted to CSV file '{csv_file_path}'."
    tl.log_message(message)
    if printFeedback:
        print(message)

##############################################################################

def CSV_to_Comsol_TXT(csv_file_path: str, txt_file_path: str, printFeedback = True):
    """ 
    Converts a CSV file containing parameters into a TXT file formatted for import/load into the COMSOL Multiphysics Software.
    
    Args:
        csv_file_path (str):    The file path of the input CSV file containing the parameters, with columns 'name', 'value', and 'description'.
        txt_file_path (str):    The file path where the output TXT file will be saved, with lines formatted as "name value "description"".
        printFeedback (bool):   If True, prints a success message after the conversion is completed.

    Raises:
        TypeError:  If either of the provided file paths is not a string.
    """
    if not isinstance(csv_file_path, str):
        raise TypeError("The argument 'csv_file_path' needs to be a string.")
    if not isinstance(txt_file_path, str):
        raise TypeError("The argument 'txt_file_path' needs to be a string.")

    # open the CSV file as read-only and close it automatically after the block
    with open(csv_file_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        # write the data into a text file
        with open(txt_file_path, "w", encoding="utf-8") as txt_file:
            for row in reader:
                # Format each row back to: name value "description"
                line = (
                    f'{row["name"]} {row["value"]} "{row["description"]}"\n'
                )
                txt_file.write(line)
    message = f"CSV file '{csv_file_path}' successfully converted to TXT file '{txt_file_path}'."
    tl.log_message(message)
    if printFeedback:
        print(message)

##############################################################################
############################################################################## 

def studies_features_name_tag_dict(model: mph.Model, printFeedback = True):
    """
    Creates two dictionaries for a given comsol multiphysics model: one mapping study names to their corresponding tags (IDs) and another mapping study names to dictionaries of feature names and their corresponding tags (IDs).
    
    Args:
        model (mph.Model):      An instance of mph.Model for which the study and feature tags should be printed.
        printFeedback (bool):   If True, prints the study and feature names along with their tags (IDs) to the console.

    Returns:
        (dict, dict): A tuple containing two dictionaries:
            1. A dictionary mapping study names (keys) to their corresponding tags (values).
            2. A nested dictionary mapping study names (keys level 1) to dictionaries of feature names (keys level 2) and their corresponding tags (values).

    Raises:
        TypeError: If the provided model is not an instance of mph.Model.
    """
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    
    # create lists and dictionaries to store the study and feature names and tags
    studynames = []                             # list to store the study names
    study_tags = model.java.study().tags()      # list of study tags (IDs)
    feature_names = {}                          # dictionary to store feature names for each study
    feature_tags = {}                           # dictionary to store feature tags (IDs) for each study

    # Loop through each study (tag)
    for study_id in study_tags:
        # get studyname from tag und store it
        studyname = model.java.study(study_id).label()
        studynames.append(studyname)

        
        feature_names[studyname] = []                                           # list to store feature names for the current study
        feature_tags[studyname] = model.java.study(study_id).feature().tags()   # list of feature tags (IDs) for the current study

        # Loop through each feature tag (ID) for the current study
        for feature_id in feature_tags[studyname]:
            # get featurename from tag und store it
            feature_name = model.java.study(study_id).feature(feature_id).label()
            feature_names[studyname].append(feature_name)

    # creat a dictionary for export
    # first level: study names
    # values: study tags (IDs)
    studies_dict = dict(zip(studynames, study_tags))

    # creat a dictionary for export
    # first level: study names
    # second level: feature names
    # values: feature tags (IDs)
    features_dict = {}
    for studyname in studynames:
        features_dict[studyname] = dict(zip(feature_names[studyname], feature_tags[studyname]))

    if printFeedback:
        for studyname in studies_dict.keys():
            print(f"Study: {studyname} (Tag: {studies_dict[studyname]})")
            for feature_name, feature_tag in features_dict[studyname].items():
                print(f"  -> Feature: {feature_name} (Tag: {feature_tag})")
    return studies_dict, features_dict

##############################################################################
##############################################################################

def add_sweep_parameter(model:mph.Model,
                        studyname: str,
                        featurename: str,
                        parameternames: list,
                        parametervalues: list,
                        printFeedback = True,
                        ):
    """
    Adds a sweep parameter to a specified featurename of a specified studyname in a given comsol multiphysics model.
    The function pulls the feature tags for the specified study and feature using the studies_features_name_tag_dict function and then sets the sweep parameter using the mph.Model.java API.
    The parameter values inside the list need to be strings like "range(0, 1, 10)" or "0, 0.5, 1" for example. The parameter names and values need to be provided as lists, even if there is only one parameter to be added.


    Args:
        model (mph.Model):          An instance of mph.Model to which the sweep parameter should be added.
        studyname (str):            The name of the study to which the sweep parameter should be added.
        featurename (str):          The name of the feature within the specified study to which the sweep parameter should be added.
        parameternames (list):      A list of parameter names (str) to be added as sweep parameters.
        parametervalues (list):     A list of parameter values (str) corresponding to the parameter names to be added as sweep parameters.

    Raises:
        TypeError:      If the provided model is not an instance of mph.Model, or if the studyname, featurename, parameternames, or parametervalues are not of the expected types.
        ValueError:     If the specified studyname or featurename is not found in the model.
    """
    # check model instance
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    if not isinstance(studyname, str):
        raise TypeError("The argument 'studyname' needs to be a string.")
    if not isinstance(featurename, str):
        raise TypeError("The argument 'featurename' needs to be a string.")
    if not isinstance(parameternames, list):
        raise TypeError("The argument 'parameternames' needs to be a list of strings.")
    if not isinstance(parametervalues, list):
        raise TypeError("The argument 'parametervalues' needs to be a list of strings.")
    for name in parameternames:
        if not isinstance(name, str):
            raise TypeError("All elements in 'parameternames' need to be strings.")
    for value in parametervalues:
        if not isinstance(value, str):
            raise TypeError("All elements in 'parametervalues' need to be strings.")
    
    # get study IDs and feature IDs
    studies_dict, features_dict = studies_features_name_tag_dict(model, printFeedback=False)
    study_id = studies_dict.get(studyname)
    if study_id is None:
        raise ValueError(f"Study '{studyname}' not found in the model.")
    feature_id = features_dict[studyname].get(featurename)
    if feature_id is None:
        raise ValueError(f"Feature '{featurename}' not found in study '{studyname}'.")
    
    # add sweep parameter to the specified feature of the specified study
    model.java.study(study_id).feature(feature_id).set('pname', parameternames)
    model.java.study(study_id).feature(feature_id).set('plistarr', parametervalues)

    message = f"Sweep parameter(s) '{parameternames}' with value(s) '{parametervalues}' added to feature '{featurename}' of study '{studyname}' of the model '{model.java.label()}'."
    tl.log_message(message)
    if printFeedback:
        print(message)

##############################################################################
##############################################################################

def save_as_copy(model: mph.Model, 
                 client: mph.Client,
                 export_path: str | None = None,
                 smallerFilesize = False):
    """
    Saves the current state of the model as a separate file.
    
    This function avoids permanently modifying the model's active file
    association, ensuring the instance remains linked to its original file.
    
    Args:
        model (mph.Model):                          The mph model instance to be saved.
        client (mph.Client):                        The mph client instance.
        export_path (str | None):    The target file path or file name.
        smallerFilesize (bool):                     Whether to clear the model's data before saving. Defaults to False.

    Raises:
        TypeError:  If the provided model is not an instance of mph.Model, or if the provided client is not an instance of mph.Client, or if the export_path is not a string, pathlib.Path, or None.
    """
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    if not isinstance(client, mph.Client):
        raise TypeError("The argument 'client' needs to be an instance of 'mph.Client'.")
    if export_path is not None and not isinstance(export_path, str):
        raise TypeError("The argument 'export_path' needs to be a string, pathlib.Path, or None.")

    # get the file path of the currently active model from MPh
    original_file_path = str(pathlib.Path(model.file()).resolve())
    
    # determine the full export path based on the provided export_path argument
    if export_path is None:
        full_export_path = original_file_path
    else:
        # force the target path to have the correct '.mph' extension
        target_path_obj = pathlib.Path(export_path)
        if target_path_obj.suffix != '.mph':
            target_path_obj = target_path_obj.with_suffix('.mph')
            
        # get absolute path for the target directory and filename
        full_export_path = str(target_path_obj.resolve())
    
    # If the export path is the same as the original file path, we can save directly without cloning
    if full_export_path == original_file_path:
        if smallerFilesize:
            # Clears computed solution data, meshes, and plot previews
            model.clear()
            # Resets modeling history to clear further internal cached data
            model.reset()
        model.save()
        tl.log_message(f'Model "{model.name()}" saved to its original path: {full_export_path}')
        return
    
    # Get the previous preference state for excluding data in MPH files
    previous_exclude_state = model.java.excludeComputedDataInMph()

    try:
        # set whether to exclude computed, mesh, and plot data during the save operation based on the smallerFilesize flag
        model.java.excludeComputedDataInMph(smallerFilesize)

        # Save the model to the new path without modifying the active RAM state
        model.java.save(full_export_path, True)
        tl.log_message(f'Model "{model.name()}" saved as a copy to: {full_export_path} (smallerFilesize={smallerFilesize})')
        
    finally:
        # Revert the exclusion setting to its original state
        model.java.excludeComputedDataInMph(previous_exclude_state)

        

##############################################################################
##############################################################################

def find_COMSOL_dataset_tag(model: mph.Model, dataset_identifier: str, printFeedback: bool = False) -> str | None:
    """
    Finds the internal COMSOL dataset tag based on the provided dataset identifier.
    
    Args:
        model (mph.Model):                      An instance of mph.Model for which the dataset tag should be found.
        dataset_identifier (str):               The identifier of the dataset to be used in the export node.
        printFeedback (bool, optional):         If True, prints feedback about the found dataset tag. Default is False.

    Returns:
        str | None: The internal COMSOL dataset tag if found, otherwise None.
    """
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    if not isinstance(dataset_identifier, str):
        raise TypeError("The argument 'dataset_identifier' needs to be a string.")

    java_model = model.java
    dataset_registry = java_model.result().dataset()
    dataset_tags = list(dataset_registry.tags())
    actual_dataset_tag = None
    
    # test if the dataset identifier is directly a dataset tag
    if dataset_identifier in dataset_tags:
        actual_dataset_tag = dataset_identifier
    else:
        # search for the dataset tag based on the dataset label containing the provided dataset identifier
        for d_tag in dataset_tags:
            lbl = java_model.result().dataset(d_tag).label()
            if dataset_identifier == lbl or f"({dataset_identifier})" in lbl:
                actual_dataset_tag = d_tag
                if printFeedback:
                    print(f"Dataset: '{dataset_identifier}' has been identified as COMSOL-Tag '{d_tag}'.")
                    tl.log_message(f"Dataset: '{dataset_identifier}' has been identified as internal COMSOL-Tag '{d_tag}'.")
                break

    if actual_dataset_tag is None and printFeedback:
                print(f"Warning: Dataset '{dataset_identifier}' not found.")

    return actual_dataset_tag

def override_export_variables(model: mph.Model, 
                              export_node_name: str,
                              expressions: list,
                              dataset_identifier: str,
                              descriptions: list | None = None,
                              printFeedback: bool = False
                              ):
    """
    Sets or overrides the variables and dataset of the named export node. When the export node does not exist, it will be created. 
    The dataset can be identified by its internal COMSOL tag (e.g. "dset1") or by a string that is part of its label (e.g. "Study 1/Solution 1"). 
    If no dataset identifier is provided, the export node's dataset will not be changed. The expressions and descriptions need to be provided as lists, even if there is only one variable to be added.
    
    Args:
        model (mph.Model):                      An instance of mph.Model for which the export variables should be set.
        export_node_name (str):                 The name of the export node to be created or overridden.
        expressions (list):                     A list of expressions (str) to be set in the export node.
        dataset_identifier (str):               The identifier of the dataset to be used in the export node.
        descriptions (list, optional):          A list of descriptions (str) corresponding to the expressions to be set in the export node. If None, empty descriptions will be used. Default is None.
    """
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    if not isinstance(export_node_name, str):
        raise TypeError("The argument 'export_node_name' needs to be a string.")
    if not isinstance(expressions, list):
        raise TypeError("The argument 'expressions' needs to be a list of strings.")
    if descriptions is not None and not isinstance(descriptions, list):
        raise TypeError("The argument 'descriptions' needs to be a list of strings or None.")
    if dataset_identifier is not None and not isinstance(dataset_identifier, str):
        raise TypeError("The argument 'dataset_identifier' needs to be a string or None.")
    

    java_model = model.java
    export_registry = java_model.result().export()
    
    # find export node with the given name
    target_node = None
    for tag in list(export_registry.tags()):
        node = java_model.result().export(tag)
        if node.label() == export_node_name:
            target_node = node
            break
    
    # if no export node with the given name exists, create a new one
    if target_node is None:
        new_tag = export_registry.uniquetag("data")
        export_registry.create(new_tag, "Data")
        target_node = java_model.result().export(new_tag)
        target_node.label(export_node_name)

    # find the internal dataset tag based on the provided dataset identifier
    actual_dataset_tag = find_COMSOL_dataset_tag(model=model, dataset_identifier=dataset_identifier, printFeedback=printFeedback)

    if actual_dataset_tag is not None:
        target_node.set("data", actual_dataset_tag)
    else:
        if printFeedback:
            print(f"Warning: Dataset '{dataset_identifier}' not found. Using default setting.")

    # (over-)write expressions and descriptions into the export node
    if descriptions is None:
        descriptions = [""] * len(expressions)
    target_node.set("expr", expressions)
    target_node.set("descr", descriptions)

    message = f"Variables successfully set to export '{export_node_name}' of model '{model.name()}'."
    tl.log_message(message)
    if printFeedback:
        print(message)
    
    return str(target_node.label())  # return the label of the export node for reference

##############################################################################
##############################################################################

def create_comsol_cut_line_3d(
        model: mph.Model,
        input_tag: str,
        output_tag: str,
        point1: tuple,
        point2: tuple,
        bounded: bool=True,
        distances: None | list | tuple | np.ndarray | str = None,
        orth_vector: list[int] | None = None,
        label=None,
        enforce_unique_tag: bool = False
        ):
    """
    Creates a Cut Line 3D dataset in the COMSOL model using the native Java API.
    Supports both COMSOL formula strings and Python/NumPy arrays for distances.

    Args:
        model (mph.Model):          The COMSOL model object loaded via Mph
        input_tag (str):            Internal tag of the source dataset (e.g., "sol1")
        output_tag (str):           Internal tag of the output dataset (e.g., "cpl1")
        point1 (tuple):             Coordinates of the first point (x1, y1, z1) for the cut line
        point2 (tuple):             Coordinates of the second point (x2, y2, z2) for the cut line
        bounded (bool):             Whether the cut line is bounded by the two points (True) or extends infinitely (False)
        distances (optional):       String formula, list, tuple, or NumPy array for additional parallel lines
        orth_vector (optional):     List of three integers representing the normal plane span vector for additional parallel lines
        label (optional):           Visible GUI label text for the COMSOL model tree
        enforce_unique_tag (bool):  Whether to enforce a unique internal tag for the output dataset. If True, a new unique tag will be generated if the provided output_tag already exists.
                                    If False, the existing dataset with the same tag will be removed and replaced.

    Returns:
        str: The internal tag of the created Cut Line 3D dataset. 
    """
    # 1. Generate a unique internal tag for the new dataset if the provided output_tag already exists
    existing_tags = [str(t) for t in model.java.result().dataset().tags()]
    if output_tag in existing_tags:
        if enforce_unique_tag:
            counter = 1
            output_tag = f"cpl{counter}"
            while output_tag in existing_tags:
                counter += 1
                output_tag = f"cpl{counter}"
        else:
            model.java.result().dataset().remove(output_tag)
        
    # 2. Create the Cut Line 3D dataset
    cut_line = model.java.result().dataset().create(output_tag, "CutLine3D")
    
    # 3. Set the visible GUI label
    if label:
        cut_line.label(str(label))
    else:
        cut_line.label(f"Cut Line 3D {output_tag}")
        
    # 4. Assign the source dataset
    cut_line.set("data", input_tag)
    
    # 5. Set Line entry method explicitly to "Two points"
    cut_line.set("method", "twopoint")
    
    # 6. Set coordinates for Point 1 and Point 2 via 'genpoints' matrix rows
    # Index format: setIndex("genpoints", value, point_index, coordinate_index)
    for i in range(3):
        cut_line.setIndex("genpoints", str(point1[i]), 0, i)
        cut_line.setIndex("genpoints", str(point2[i]), 1, i)
    
    # 7. Set "Bounded by points" option
    cut_line.set("bounded", bounded)
    
    # 8. Configure additional parallel lines using verified 'genpara...' properties
    if distances is not None:
        cut_line.set("genparaactive", True)
        
        # Convert Python list, tuple, or NumPy array to COMSOL space-separated string
        if hasattr(distances, '__iter__') and not isinstance(distances, (str, bytes)):
            distances_str = " ".join(map(str, distances))
        else:
            distances_str = str(distances)
            
        cut_line.set("genparadist", distances_str)
        
        # Official property name for the normal plane span vector is 'orthvec'
        vec = orth_vector if orth_vector else [0, 0, 1]
        vec_str_array = [str(x) for x in vec]
        cut_line.set("orthvec", vec_str_array)
    else:
        cut_line.set("genparaactive", "off")
        
    return output_tag

##############################################################################

def create_comsol_cut_plane(
        model: mph.Model,
        input_tag: str,
        output_tag: str,
        plane_direction: str,
        coordinate: float | str,
        distances=None,
        label=None,
        enforce_unique_tag: bool = False,
        ):
    """
    Creates a Cut Plane dataset in the COMSOL model using the native Java API.
    Supports both COMSOL formula strings and Python/NumPy arrays for distances.

    Args:
        model (mph.Model):          The COMSOL model object loaded via Mph
        input_tag (str):            Internal tag of the source dataset (e.g., "sol1")
        output_tag (str):           Internal tag of the output dataset (e.g., "cpl1")
        plane_direction (str):      Direction of the cut plane ("xy", "yz" or "xz")
        coordinate (float | str):   Coordinate value for the cut plane (can be a number or a COMSOL formula string)
        distances (optional):       String formula, list, tuple, or NumPy array for additional parallel planes
        label (optional):           Visible GUI label text for the COMSOL model tree
        enforce_unique_tag (bool):  Whether to enforce a unique internal tag for the output dataset. If True, a new unique tag will be generated if the provided output_tag already exists.
                                    If False, the existing dataset with the same tag will be removed and replaced.

    Returns:
        str: The internal tag of the created Cut Plane dataset.
    """
    existing_tags = [str(t) for t in model.java.result().dataset().tags()]
    
    # 1. Generate a unique internal tag for the new dataset if the provided output_tag already exists
    existing_tags = [str(t) for t in model.java.result().dataset().tags()]
    if output_tag in existing_tags:
        if enforce_unique_tag:
            counter = 1
            output_tag = f"cpl{counter}"
            while output_tag in existing_tags:
                counter += 1
                output_tag = f"cpl{counter}"
        else:
            model.java.result().dataset().remove(output_tag)
        
    # 2. Create the Cut Plane dataset using the internal tag
    cut_plane = model.java.result().dataset().create(output_tag, "CutPlane")
    
    # 3. Set the visible GUI label
    if label:
        cut_plane.label(str(label))
    else:
        cut_plane.label(f"Cut Plane ({output_tag})")
    
    # 4. Assign the source dataset
    cut_plane.set("data", input_tag)
    
    # 5. Set the plane type to "Quick"
    cut_plane.set("planetype", "quick")
    
    # 6. Format and set the plane orientation (handles "yz", "xz", or "xy")
    plane_format = plane_direction.lower().replace("-planes", "").replace("zx", "xz")
    cut_plane.set("quickplane", plane_format)
    
    # 7. Set the coordinate depending on the chosen plane direction
    if plane_format == "yz":
        cut_plane.set("quickx", str(coordinate))
    elif plane_format == "xz":
        cut_plane.set("quicky", str(coordinate))
    elif plane_format == "xy":
        cut_plane.set("quickz", str(coordinate))
        
    # 8. Configure additional parallel planes (distances) if provided
    if distances is not None:
        # Fixed: 'genparaactive' is the correct property name for CutPlane (not quickextra)
        cut_plane.set("genparaactive", True)
        
        # Convert Python list, tuple, or NumPy array to COMSOL space-separated string
        if hasattr(distances, '__iter__') and not isinstance(distances, (str, bytes)):
            distances_str = " ".join(map(str, distances))
        else:
            distances_str = str(distances)
            
        # Fixed: 'quickdistance' is correct for CutPlane when planetype is quick
        cut_plane.set("quickdistance", distances_str)
    else:
        cut_plane.set("genparaactive", False)
        
    return output_tag


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

# creat manager to suppress prints based on a boolean flag
@contextlib.contextmanager
def suppress_prints(should_suppress: bool):
    if should_suppress:
        # suppress the prints by redirecting stdout to os.devnull
        with open(devnull, "w") as fnull:
            with contextlib.redirect_stdout(fnull):
                yield
    else:
        # allow prints to go through
        yield


##############################################################################
##############################################################################

def evaluate_parameters_and_terminals(
        model: mph.Model, 
        output_csv_path: pathlib.Path, 
        output_folder: pathlib.Path,
        data_export_folder: pathlib.Path,
        modelname: str
        ):
    """
    Evaluates the parameter expressions in the COMSOL model and saves them to a CSV file.
    Also evaluates the corresponding terminal data for parameters matching specific patterns and saves them to a separate CSV file.

    Args:
        model (mph.Model):                      An instance of mph.Model for which the parameter expressions should be evaluated.
        output_csv_path (pathlib.Path):         The path to the CSV file where the evaluated parameters will be saved.
        output_folder (pathlib.Path):           The folder path where error messages will be saved if evaluation fails.
        data_export_folder (pathlib.Path):      The folder path where the evaluated terminal data will be saved.
        modelname (str):                        The name of the model, used for naming the terminal data CSV file.
    """
    # evaluate the parameter expressions and save them to a CSV file
    df_parameters = pd.read_csv(output_csv_path)
    terminal_data = []
    for param in df_parameters["name"].tolist():
        try:
            evaluated_value = model.evaluate(f"root.{param}")
            df_parameters.loc[df_parameters["name"] == param, "evaluated_value"] = evaluated_value #type: ignore
        except Exception as e:
            tl.log_message(f"Warning: Could not evaluate parameter '{param}': {e}")
            df_parameters.loc[df_parameters["name"] == param, "evaluated_value"] = None

        # check if the parameter name matches the pattern "V01", "V02", ..., "V99" or "V_<suffix>" or "I_<suffix>" 
        Vnum_match = re.match(r"^V(\d{2})$", param) # matches V01, V02, ..., V99
        V_match = re.match(r"^V_(.*)$", param)
        I_match = re.match(r"^I_(.*)$", param) 
        if Vnum_match:
            num_str = Vnum_match.group(1) # extract the two-digit number
            terminal_str = f"G{num_str}"  # construct the corresponding terminal string
        elif V_match:
            terminal_str = V_match.group(1)
        elif I_match:
            terminal_str = I_match.group(1)

        # and evaluate the corresponding "ec.I0_GXX" and "ec.V0_GXX" expressions
        if Vnum_match or V_match or I_match:
            try:
                I_str = f"ec.I0_{terminal_str}"
                I_value = model.evaluate(I_str)

                V_str = f"ec.V0_{terminal_str}"
                V_value = model.evaluate(V_str)

                terminal_data.append({"Terminal": terminal_str, "Voltage (V)": V_value, "Current (A)": I_value})
            except Exception as e:
                error_warning = f"Warning: Could not evaluate terminal data for '{terminal_str}'"
                tl.log_message(error_warning)

                with open(output_folder / 'errormessage.txt', 'a') as f:
                    f.write(f"Error occurred while processing {modelname}: \n{str(e)}\n\n\n")
                print(error_warning)

    # save the evaluated parameters to the CSV file
    df_parameters.to_csv(output_csv_path, index=False)
    
    # save the evaluated terminal data to a separate CSV file
    df_terminals = pd.DataFrame(terminal_data)
    df_terminals.to_csv(data_export_folder / f"{modelname}-terminals.csv", index=False)

##############################################################################

def simulate_model(
        # path settings
        filename: str,
        input_folder: pathlib.Path,
        output_folder: pathlib.Path,

        # simulation settings
        client: mph.client.Client,

        export_params: list,
        export_descriptions: list,

        conductor_export_params: list | None = None,
        conductor_export_descriptions: list | None = None,

        # COMSOL internal interpolation
        Depth_point1: tuple|None = None,
        Depth_point2: tuple|None = None,

        Homogeneity_point1: tuple|None = None,
        Homogeneity_point2: tuple|None = None,
        Homogeneity_distances: None | list | tuple | np.ndarray | str  = None,
        Homogeneity_orth_vector: list[int] | None = None,

        Longitudinal_point1: tuple|None = None,
        Longitudinal_point2: tuple|None = None,
        Longitudinal_distances: None | list | tuple | np.ndarray | str = None,
        Longitudinal_orth_vector: list[int] | None = None,

        xy_plane_coordinate: float | str  = 0.0,

        # boolean flags    
        export_parameters_to_csv: bool = True,
        extend_export_from_params_in_csv: bool = True,
        show_model_info: bool = False,
        solve_model: bool = True,
        save_solved_model: bool = True,
        evaluate_parameter_expressions: bool = True,
        export_all_solution_data: bool = False,
        export_line_solution_data: bool = True,
        export_plane_solution_data: bool = True,
        save_small_model_version: bool = True,
        new_log_file: bool = False,

        # marking iteration
        iteration_number: int | None = None,
        model: mph.Model | None = None,
    ):
    """
    Simulates a COMSOL Multiphysics model by importing it, optionally exporting parameters to CSV, solving the model, exporting solution data, and saving different versions of the model.
    
    Args:
        filename (str):                             The name of the COMSOL model file to be imported.
        input_folder (pathlib.Path):                The folder path where the input model file is located.
        output_folder (pathlib.Path):               The folder path where output files will be saved.

        client (mph.client.Client):                 An instance of the COMSOL client for model operations.

        export_params (list):                       A list of parameters to be exported from the model.
        export_descriptions (list):                 A list of descriptions corresponding to the export parameters. If you dont want to use descriptions, you can provide an empty list.

        conductor_export_params (list | None):       A list of parameters to be exported for the conductor. If None, no conductor export will be performed.
        conductor_export_descriptions (list | None): A list of descriptions corresponding to the conductor export parameters. If None, no descriptions will be used.

        Depth_point1 (tuple|None):                  The first point for the depth cut line. If None, depth cut line export will be skipped.
        Depth_point2 (tuple|None):                  The second point for the depth cut line. If None, depth cut line export will be skipped.

        Homogeneity_point1 (tuple|None):            The first point for the homogeneity cut line. If None, homogeneity cut line export will be skipped.
        Homogeneity_point2 (tuple|None):            The second point for the homogeneity cut line. If None, homogeneity cut line export will be skipped.
        Homogeneity_distances (None | list | tuple | np.ndarray | str):  Distances for additional parallel lines for the homogeneity cut line. If None, no additional lines will be created.
        Homogeneity_orth_vector (list[int]|None):   Orthogonal vector for the additional lines for the homogeneity cut line. If None, defaults to [0, 0, 1].

        Longitudinal_point1 (tuple|None):           The first point for the longitudinal cut line. If None, longitudinal cut line export will be skipped.
        Longitudinal_point2 (tuple|None):           The second point for the longitudinal cut line. If None, longitudinal cut line export will be skipped.
        Longitudinal_distances (None | list | tuple | np.ndarray | str):  Distances for additional parallel lines for the longitudinal cut line. If None, no additional lines will be created.
        Longitudinal_orth_vector (list[int]|None):  Orthogonal vector for the additional lines for the longitudinal cut line. If None, defaults to [0, 0, 1].

        xy_plane_coordinate (float | str ):         The coordinate for the xy cut plane. If None, xy cut plane export will be skipped.

        export_parameters_to_csv (bool):            If True, exports model parameters to a CSV file.
        extend_export_from_params_in_csv (bool):    If True, extends the export parameters and descriptions with those from the exported CSV file. Can be used independet of "export_parameters_to_csv" if the CSV file already exists.
        show_model_info (bool):                     If True, prints model information including parameters, materials, physics, and studies.
        solve_model (bool):                         If True, solves the model. If model is already solved, this can be set to False to save time.
        save_solved_model (bool):                   If True, saves a copy of the solved model.
        evaluate_parameter_expressions (bool):      If True, evaluates parameter expressions and saves them to a CSV file.
        export_all_solution_data (bool):            If True, exports all the solution data from the model.
        export_line_solution_data (bool):           If True, exports solution data along specified cut lines (depth, homogeneity, longitudinal).
        export_plane_solution_data (bool):          If True, exports solution data along a specified cut plane.
        save_small_model_version (bool):            If True, saves a smaller version of the model without solutions but with settings, parameters, and configured exports.
        new_log_file (bool):                        If True, initializes a new log file for this simulation run.

        iteration_number (int | None):     An optional iteration number to mark this simulation run. If provided, the output folder will be adjusted accordingly.
        model (mph.Model | None):          An optional pre-loaded COMSOL model instance. If provided, the function will use this model instead of importing it from the input folder.
        
    Returns:
        None
    """
    # create new log file if requested, otherwise use the existing log file
    last_log_path = tl.LOG_PATH
    last_start_time = tl.START_TIME
    last_last_time = tl.LAST_TIME

    file_iteration = f"{filename}-iteration_{iteration_number}" if iteration_number else filename

    if tl.LOG_PATH is None or new_log_file:
        if tl.LOG_PATH is not None:
            tl.log_message(f"Loging simulation of model '{file_iteration}' in a separate log file as per user request.")
        tl.initialize_time_log(output_folder / f"{file_iteration}-simulation_log.csv", startmessage=f"Created new log file for simulation of model '{file_iteration}'.")

    try:
        if model is None:
            # Import the model from the specified input folder
            client, model_list = import_models(
                    client=client, 
                    MODELNAME_LIST=[filename], 
                    printFeedback=False,
                    avoid_reimporting=True,
                    path_to_models=str(input_folder),
                    )

            # this function only uses one model
            model = model_list[0]
        else:
            keep_model_after_simulation = True  # flag to indicate that the model should not be closed after simulation

        if not isinstance(model, mph.Model):
            raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
        
        modelname = client.names()[0]

        # Time logging
        tl.log_message(f"Working with model {modelname}.")

        if iteration_number is None:
            data_export_folder = output_folder / "Data Export"
        else:
            tl.log_message(f"Marking this simulation as iteration {iteration_number}.")
            data_export_folder = output_folder / "Sweep Export"
            modelname = f"{modelname}-iteration_{iteration_number}"
            
        makedirs(data_export_folder, exist_ok=True)

        # export model parameters to CSV
        output_csv_path = data_export_folder / f"{modelname}-parameters.csv"
        if export_parameters_to_csv:
            save_Parameter_List_to_CSV(model, str(output_csv_path), displayParams=False)


        if extend_export_from_params_in_csv:
            # extend the EXPORT_PARAMS and EXPORT_DESCRIPTION lists with the parameters from the CSV
            df_parameters = pd.read_csv(output_csv_path)
            export_params.extend([f"root.{param}" for param in df_parameters["name"].tolist()])
            export_descriptions.extend([f"{desc}" for desc in df_parameters["description"].tolist()])

        if show_model_info:
            # print parameters, materials, physics, and studies of the model
            _ = print_model_info(model)

            # Display the model tree structure
            mph.tree(model)

        if solve_model:
            # Time logging
            tl.log_message(f"Solving {modelname}...")

            # Solve the model
            model.solve()

            # Time logging        
            tl.log_message(f"Finished solving {modelname}.")
        else:
            tl.log_message(f"Skipping solving {modelname} as per user request.")

        # save the solved model
        if save_solved_model:
            save_as_copy(model, client, str(output_folder / f"{modelname}-solved.mph"), smallerFilesize=False)

        if evaluate_parameter_expressions:
            evaluate_parameters_and_terminals(
                model = model,
                output_csv_path = output_csv_path,
                output_folder = output_folder,
                data_export_folder = data_export_folder,
                modelname = modelname
                )

        if export_all_solution_data:
            # export data from the model
            export_node_label = override_export_variables(model, "PyExport", export_params, "Study 1/Solution 1", export_descriptions)
            model.export(export_node_label, data_export_folder / f"{modelname}-exported_data.txt")

            # Time logging
            tl.log_message(f"Exported all solution data of {modelname}.")

        if export_line_solution_data:
            all_solution = find_COMSOL_dataset_tag(model=model, dataset_identifier= "Study 1/Solution 1", printFeedback = False)

            if all_solution is None:
                all_solution = "sol1"
                tl.log_message(f"Warning: Dataset 'Study 1/Solution 1' not found. Using default dataset tag '{all_solution}' for line cut export.")

            if Depth_point1 is not None and Depth_point2 is not None:
                cut_line_tag = create_comsol_cut_line_3d(
                                    model = model,
                                    input_tag = all_solution,
                                    output_tag="Pycln1",
                                    point1 = Depth_point1,
                                    point2 = Depth_point2,
                                    bounded = True,
                                    distances=None,
                                    orth_vector=None,
                                    label="Cut Line PyDepth"
                                    )
                
                export_node_label = override_export_variables(
                                        model=model, 
                                        export_node_name = "PyDepthExport",
                                        expressions  = export_params,
                                        dataset_identifier = cut_line_tag,
                                        descriptions = export_descriptions
                                        )
                model.export(export_node_label, data_export_folder / f"{modelname}-depth_exported_data.txt")
                
                # Time logging
                tl.log_message(f"Exported depth solution data of {modelname}.")

            if Homogeneity_point1 is not None and Homogeneity_point2 is not None:
                cut_line_tag = create_comsol_cut_line_3d(
                                    model = model,
                                    input_tag = all_solution,
                                    output_tag="Pycln2",
                                    point1 = Homogeneity_point1,
                                    point2 = Homogeneity_point2,
                                    bounded = True,
                                    distances = Homogeneity_distances,
                                    orth_vector = Homogeneity_orth_vector,
                                    label="Cut Line PyHomogeneity"
                                    )
                
                export_node_label = override_export_variables(
                                        model=model, 
                                        export_node_name = "PyHomogeneityExport",
                                        expressions  = export_params,
                                        dataset_identifier = cut_line_tag,
                                        descriptions = export_descriptions
                                        )
                model.export(export_node_label, data_export_folder / f"{modelname}-homogeneity_exported_data.txt")
                
                # Time logging
                tl.log_message(f"Exported homogeneity solution data of {modelname}.")

            if Longitudinal_point1 is not None and Longitudinal_point2 is not None:
                cut_line_tag = create_comsol_cut_line_3d(
                                    model = model,
                                    input_tag = all_solution,
                                    output_tag="Pycln3",
                                    point1 = Longitudinal_point1,
                                    point2 = Longitudinal_point2,
                                    bounded = True,
                                    distances = Longitudinal_distances,
                                    orth_vector = Longitudinal_orth_vector,
                                    label="Cut Line PyLongitudinal"
                                    )
                
                export_node_label = override_export_variables(
                                        model=model, 
                                        export_node_name = "PyLongitudinalExport",
                                        expressions  = export_params,
                                        dataset_identifier = cut_line_tag,
                                        descriptions = export_descriptions
                                        )
                model.export(export_node_label, data_export_folder / f"{modelname}-longitudinal_exported_data.txt")
                
                # Time logging
                tl.log_message(f"Exported longitudinal solution data of {modelname}.")

        if export_plane_solution_data:
                    all_solution = find_COMSOL_dataset_tag(model=model, dataset_identifier= "Study 1/Solution 1", printFeedback = False)
        
                    if all_solution is None:
                        all_solution = "sol1"
                        tl.log_message(f"Warning: Dataset 'Study 1/Solution 1' not found. Using default dataset tag '{all_solution}' for line cut export.")

                    if True:
                        cut_line_tag = create_comsol_cut_plane(
                                            model = model,
                                            input_tag = all_solution,
                                            output_tag="Pycp1",
                                            plane_direction = "xy",
                                            coordinate = xy_plane_coordinate,
                                            distances=None,
                                            label="Cut Plane PyXY")
                        
                        export_node_label = override_export_variables(
                                                model=model, 
                                                export_node_name = "PyPlaneExport",
                                                expressions  = export_params,
                                                dataset_identifier = cut_line_tag,
                                                descriptions = export_descriptions
                                                )
                        model.export(export_node_label, data_export_folder / f"{modelname}-xy_exported_data.txt")
                        
                        # Time logging
                        tl.log_message(f"Exported plane solution data of {modelname}.")

                    if conductor_export_params is not None:
                        cut_line_tag = create_comsol_cut_plane(
                                            model = model,
                                            input_tag = all_solution,
                                            output_tag="Pycp2",
                                            plane_direction = "xy",
                                            coordinate = "0.5*conductor_all_height",
                                            distances=None,
                                            label="Cut Plane PyConductor")
                        
                        export_node_label = override_export_variables(
                                                model=model, 
                                                export_node_name = "PyConductorExport",
                                                expressions  = conductor_export_params,
                                                dataset_identifier = cut_line_tag,
                                                descriptions = conductor_export_descriptions
                                                )
                        model.export(export_node_label, data_export_folder / f"{modelname}-conductor_exported_data.txt")
                        
                        # Time logging
                        tl.log_message(f"Exported plane solution data of {modelname}.")

        # save the solved model
        if save_solved_model:
            save_as_copy(model, client, str(output_folder / f"{modelname}-solved.mph"), smallerFilesize=False)

        if save_small_model_version:
            # save a smaller version of the model (without solutions, but with the actually used settings, parameters and configured exports)
            save_as_copy(model, client, str(output_folder / f"{modelname}-smallfile.mph"), smallerFilesize=True)

        if not keep_model_after_simulation:
            # clear model from client
            client.remove(model)
            tl.log_message(f"Cleared {modelname} from client.")
        else:
            tl.log_message(f"Keeping {modelname} in client since it was provided as an argument.")


        endmessage = f"Simulation of model '{file_iteration}' completed."
        tl.log_message(endmessage)
    except Exception as e:
        endmessage = f"ERROR occurred during simulation of model '{file_iteration}'."
        tl.log_message(endmessage)
        print(endmessage)

        with open(str(output_folder / "errormessage.txt"), "w", encoding="utf-8") as destination:
            destination.write(f"{endmessage}:\n\n{str(e)}\n\n")

        endmessage = f"Error message saved to {output_folder / 'errormessage.txt'}"
        tl.log_message(endmessage)
        print(endmessage)

    if last_log_path is None:
        tl.LOG_PATH = None
    elif new_log_file:
        tl.initialize_time_log(last_log_path, startmessage=endmessage, start_time=last_start_time, last_time=last_last_time)  # restore the previous log file path if it was changed

##############################################################################

if False: # old function, not used anymore, but kept for reference
    def parameter_sweep(model: mph.Model,
                        client: mph.Client,
                        studyname: str, 
                        parameternames: list,
                        parametervalues: list,
                        expressions: list ,
                        export_path: str,
                        dataset_identifier: str,
                        markwithParameters: list[str] | None = None,
                        saveSolutions = True):
        """
        Perform a parameter sweep by iterating through the provided parameter values, setting them in the model, running the specified study, and exporting the results for each iteration. 
        The export variables and dataset can be overridden for each iteration using the override_export_variables function. 
        The exported files will be named based on the provided export_path with an added suffix indicating the iteration number. 
        If saveSolutions is True, a copy of the model with the current solution will also be saved for each iteration using the save_as_copy function. 
        The parameter names and values need to be provided as lists, even if there is only one parameter to be swept.

        Args:
            model (mph.Model):                      An instance of mph.Model for which the parameter sweep should be performed.
            client (mph.Client):                    An instance of mph.Client to be used for saving model copies.
            studyname (str):                        The name of the study to be run for each iteration of the parameter sweep.
            export_path (str, optional):            The base file path for exporting results. The actual exported files will have an added suffix indicating the iteration number. 
                                                    If None, exports will be saved in the current working directory with only suffix naming. Default is None.
            parameternames (list):                  A list of parameter names (str) to be swept.
            parametervalues (list):                 A list of lists, where each inner list contains the values for the corresponding parameter.
            expressions (list):                     A list of variable names to be exported. (Parameternames are always added to expressions by this function). 
            dataset_identifier (str, optional):     The identifier of the dataset to be exported. If None, the default dataset will be used.
            markwithParameters (list, optional):    A list of parameter names (str) to be included in the folder name for each iteration. 
                                                    If None, no parameters will be included in the folder name. Default is None. The Iteration number will always be included in the folder name.
            saveSolutions (bool):                   Whether to save the model solutions for each iteration. Default is True.

        Returns:
            (list): A list of file paths where the model copies with solutions have been saved for each iteration (only if saveSolutions is True, otherwise an empty list).

        Raises:
            TypeError: If the provided model is not an instance of mph.Model, if the client is not an instance of mph.Client, if the studyname or dataset_identifier is not a string, if the export_path is not a string or None, if the parameternames or expressions are not lists of strings, or if the parametervalues is not a list of lists of strings.
        """
        
        # for example:
        # parameternames: list = ["param1", "param2"],
        # parametervalues: list = [[1,2,3], [4,5,6]], 
        # expressions: list = ["var1", "var2"],

        if markwithParameters is not None:
            if not isinstance(markwithParameters, list):
                raise TypeError("The argument 'markwithParameters' needs to be a list of strings or None.")
            for param in markwithParameters:
                if param not in parameternames:
                    raise ValueError(f"The argument 'markwithParameters' needs to be a list of strings that are in 'parameternames'. Provided: {markwithParameters}, Available: {parameternames}")
            
        MODELNAME = model.name()

        sweep_parameters = list(zip(parameternames, parametervalues))
        paths = []

        message = f"Starting parameter sweep on model '{MODELNAME}' for study '{studyname}' with parameters {parameternames}."
        tl.log_message(message)
        print(f"{message}\n")

        first_iteration = True
        for iteration in range(len(parametervalues[0])):
            message = f"Iteration {iteration+1}/{len(parametervalues[0])} with parameter values {[paramvalues[iteration] for _, paramvalues in sweep_parameters]}."
            tl.log_message(message)
            print(message)

            # set the current parameter values for this iteration
            print("Setting parameters...")
            for paramname, paramvalues in sweep_parameters:
                value = paramvalues[iteration]
                model.parameter(paramname, value) # type: ignore
            

            # adjust path for the current iteration, including parameter values in the folder name
            # iteration_param_values = [f"{paramname} {paramvalues[iteration]}" for paramname, paramvalues in sweep_parameters]
            if markwithParameters is not None:
                iteration_param_values = [f"{paramname} {paramvalues[iteration]}" for paramname, paramvalues in sweep_parameters if paramname in markwithParameters]
                param_str = " ("+ ", ".join(iteration_param_values) + ")"
            else: 
                param_str = ""

            folder_path = pathlib.Path(export_path) / f"Iteration {iteration}{param_str}" 
            makedirs(folder_path, exist_ok=True)  # create output folder if it doesn't exist

            # export the current parameter values to a CSV file for reference
            output_csv_path = folder_path / f"{MODELNAME}-iteration_{iteration}-parameters.csv"
            save_Parameter_List_to_CSV(model, str(output_csv_path), displayParams=False)

            if first_iteration:
                first_iteration = False
                df_parameters = pd.read_csv(output_csv_path)
                expressions.extend([f"root.{param}" for param in df_parameters["name"].tolist() if f"root.{param}" not in expressions])


            # solve the study
            message = f"Running study '{studyname}' for iteration {iteration+1}..."
            tl.log_message(message)
            print(message)
            try:
                model.solve(studyname)
            except Exception as e:
                message = f"ERROR occurred while solving study '{studyname}' for iteration {iteration+1}: \n\n{e}\n\n"
                tl.log_message(message)
                print(message)

                with open(str(folder_path / "errormessage.txt"), "w", encoding="utf-8") as destination:
                    destination.write(f"ERROR occurred while solving study '{studyname}' for iteration {iteration+1}:\n\n")
                    destination.write(str(e))

                message = f"Error message saved to {folder_path / 'errormessage.txt'}"
                tl.log_message(message)
                print(message)
                if saveSolutions:
                    print(f"saving failed model to {path}...")
                    save_as_copy(model, client, str(path)+".mph", smallerFilesize = False)

                message = f"Skipping data export for iteration {iteration+1} due to the error."
                tl.log_message(message)
                print(message)
                continue  # Skip to the next iteration if an error occurs

            # export the results with the current parameter values
            message = f"Exporting results for iteration {iteration+1}..."
            tl.log_message(message)
            print(message)
            
            filename = f"{MODELNAME}_iteration_{iteration}"
            path = folder_path / filename
            
            export_node_label = override_export_variables(model, export_node_name="PySweepExport", expressions=expressions , dataset_identifier=dataset_identifier)
            model.export(export_node_label, str(path) + ".txt")

            if saveSolutions:
                message = f"Saving model solution for iteration {iteration+1} to {path}..."
                tl.log_message(message)
                print(message)
                save_as_copy(model, client, str(path)+".mph", smallerFilesize = False)
                paths.append(path)

            message = f"Iteration {iteration+1} completed."
            tl.log_message(message)
            print(f"{message}\n")

        message = f"Parameter sweep for model '{MODELNAME}' completed."
        tl.log_message(message)
        print(message)
        return paths

##############################################################################

def sweep_model(
        # sweep settings
        sweep_parameter: list[str],
        sweep_values: list[list[float | int | str]],

        # path settings
        filename: str,
        input_folder: pathlib.Path,
        output_folder: pathlib.Path,

        # simulation settings
        client: mph.client.Client,

        export_params: list,
        export_descriptions: list,

        conductor_export_params: list | None = None,
        conductor_export_descriptions: list | None = None,

        # COMSOL internal interpolation
        Depth_point1: tuple|None = None,
        Depth_point2: tuple|None = None,

        Homogeneity_point1: tuple|None = None,
        Homogeneity_point2: tuple|None = None,
        Homogeneity_distances: None | list | tuple | np.ndarray | str  = None,
        Homogeneity_orth_vector: list[int] | None = None,

        Longitudinal_point1: tuple|None = None,
        Longitudinal_point2: tuple|None = None,
        Longitudinal_distances: None | list | tuple | np.ndarray | str = None,
        Longitudinal_orth_vector: list[int] | None = None,

        xy_plane_coordinate: float | str  = 0.0,

        # boolean flags    
        export_parameters_to_csv: bool = True,
        evaluate_parameter_expressions: bool = True,
        extend_export_from_params_in_csv: bool = True,
        show_model_info: bool = False,
        solve_model: bool = True,
        save_solved_model: bool = True,
        export_all_solution_data: bool = False,
        export_line_solution_data: bool = True,
        export_plane_solution_data: bool = True,
        save_small_model_version: bool = True,
        new_log_file: bool = False,
    ):
    """
    Sweeps a COMSOL Multiphysics model by iterating through the provided sweep values for the specified sweep parameters.
    The function wraps the simulation process 'simulate_model()' for each combination of sweep values, allowing for automated parameter sweeps. 
    The exported files will be named based on the provided output folder with an added suffix indicating the iteration number.
    
    Args:
        sweep_parameter (list):                     A list of parameter names (str) to be swept.
        sweep_values (list):                        A list of lists, where each inner list contains the values for the corresponding sweep parameter.
                                                    The length of each inner list should be the same, representing the number of iterations for the sweep.

        filename (str):                             The name of the COMSOL model file to be imported.
        input_folder (pathlib.Path):                The folder path where the input model file is located.
        output_folder (pathlib.Path):               The folder path where output files will be saved.

        client (mph.client.Client):                 An instance of the COMSOL client for model operations.

        export_params (list):                       A list of parameters to be exported from the model.
        export_descriptions (list):                 A list of descriptions corresponding to the export parameters. If you dont want to use descriptions, you can provide an empty list.

        conductor_export_params (list | None):       A list of parameters to be exported for the conductor. If None, no conductor export will be performed.
        conductor_export_descriptions (list | None): A list of descriptions corresponding to the conductor export parameters. If None, no descriptions will be used.

        Depth_point1 (tuple|None):                  The first point for the depth cut line. If None, depth cut line export will be skipped.
        Depth_point2 (tuple|None):                  The second point for the depth cut line. If None, depth cut line export will be skipped.

        Homogeneity_point1 (tuple|None):            The first point for the homogeneity cut line. If None, homogeneity cut line export will be skipped.
        Homogeneity_point2 (tuple|None):            The second point for the homogeneity cut line. If None, homogeneity cut line export will be skipped.
        Homogeneity_distances (None | list | tuple | np.ndarray | str):  Distances for additional parallel lines for the homogeneity cut line. If None, no additional lines will be created.
        Homogeneity_orth_vector (list[int]|None):   Orthogonal vector for the additional lines for the homogeneity cut line. If None, defaults to [0, 0, 1].

        Longitudinal_point1 (tuple|None):           The first point for the longitudinal cut line. If None, longitudinal cut line export will be skipped.
        Longitudinal_point2 (tuple|None):           The second point for the longitudinal cut line. If None, longitudinal cut line export will be skipped.
        Longitudinal_distances (None | list | tuple | np.ndarray | str):  Distances for additional parallel lines for the longitudinal cut line. If None, no additional lines will be created.
        Longitudinal_orth_vector (list[int]|None):  Orthogonal vector for the additional lines for the longitudinal cut line. If None, defaults to [0, 0, 1].

        xy_plane_coordinate (float | str ):         The coordinate for the xy cut plane. If None, xy cut plane export will be skipped.

        export_parameters_to_csv (bool):            If True, exports model parameters to a CSV file.
        evaluate_parameter_expressions (bool):      If True, evaluates parameter expressions and saves them to a CSV file.
        extend_export_from_params_in_csv (bool):    If True, extends the export parameters and descriptions with those from the exported CSV file. Can be used independet of "export_parameters_to_csv" if the CSV file already exists.
        show_model_info (bool):                     If True, prints model information including parameters, materials, physics, and studies.
        solve_model (bool):                         If True, solves the model. If model is already solved, this can be set to False to save time.
        save_solved_model (bool):                   If True, saves a copy of the solved model.
        export_all_solution_data (bool):            If True, exports all the solution data from the model.
        export_line_solution_data (bool):           If True, exports solution data along specified cut lines (depth, homogeneity, longitudinal).
        export_plane_solution_data (bool):          If True, exports solution data along a specified cut plane.
        save_small_model_version (bool):            If True, saves a smaller version of the model without solutions but with settings, parameters, and configured exports.
        new_log_file (bool):                        If True, initializes a new log file for this simulation run.

        iteration_number (int | None):     An optional iteration number to mark this simulation run. If provided, the output folder will be adjusted accordingly.

    Returns:
        None
    """
    # create new log file if requested, otherwise use the existing log file
    last_log_path = tl.LOG_PATH
    last_start_time = tl.START_TIME
    last_last_time = tl.LAST_TIME

    if tl.LOG_PATH is None or new_log_file:
        if tl.LOG_PATH is not None:
            tl.log_message(f"Loging sweep of model '{filename}' in a separate log file as per user request.")
        tl.initialize_time_log(output_folder / f"{filename}-sweep_log.csv", startmessage=f"Created new log file for sweep of model '{filename}'.")

    # Import the model from the specified input folder
    client, model_list = import_models(
                        client=client, 
                        MODELNAME_LIST=[filename], 
                        printFeedback=False,
                        avoid_reimporting=True,
                        path_to_models=str(input_folder),
                        )
            
    # this function only uses one model
    model = model_list[0]
    
    sweep_parameters = list(zip(sweep_parameter, sweep_values))
    message = f"Starting parameter sweep on model '{filename}' for parameters {sweep_parameter}."
    tl.log_message(message)


    for iteration in range(len(sweep_values[0])):
        message = f"Iteration {iteration+1}/{len(sweep_values[0])} with parameter values {[sweep_values[iteration] for _, sweep_values in sweep_parameters]}."
        tl.log_message(message)

        # set the current parameters for this iteration
        for paramname, paramvalues in sweep_parameters:
            value = paramvalues[iteration]
            model.parameter(paramname, value) 

        try:
            simulate_model(
                    # path settings
                    filename = filename,
                    input_folder = input_folder,
                    output_folder = output_folder,

                    # simulation settings
                    client = client,

                    export_params = export_params.copy(),
                    export_descriptions = export_descriptions.copy(),

                    conductor_export_params = conductor_export_params.copy() if conductor_export_params else None,
                    conductor_export_descriptions = conductor_export_descriptions.copy() if conductor_export_descriptions else None,

                    # COMSOL internal interpolation
                    Depth_point1 = Depth_point1,
                    Depth_point2 = Depth_point2,

                    Homogeneity_point1 = Homogeneity_point1,
                    Homogeneity_point2 = Homogeneity_point2,
                    Homogeneity_distances = Homogeneity_distances,
                    Homogeneity_orth_vector = Homogeneity_orth_vector,

                    Longitudinal_point1 = Longitudinal_point1,
                    Longitudinal_point2 = Longitudinal_point2,
                    Longitudinal_distances = Longitudinal_distances,
                    Longitudinal_orth_vector = Longitudinal_orth_vector,

                    xy_plane_coordinate = xy_plane_coordinate,

                    # boolean flags    
                    export_parameters_to_csv = export_parameters_to_csv,
                    evaluate_parameter_expressions = evaluate_parameter_expressions,
                    extend_export_from_params_in_csv= extend_export_from_params_in_csv,
                    show_model_info = show_model_info,
                    solve_model = solve_model,
                    save_solved_model = save_solved_model,
                    export_all_solution_data = export_all_solution_data,
                    export_line_solution_data = export_line_solution_data,
                    export_plane_solution_data = export_plane_solution_data,
                    save_small_model_version = save_small_model_version,
                    new_log_file = new_log_file,

                    # marking iteration
                    iteration_number = iteration,
                    model = model,
            )

        except Exception as e:
            with open(output_folder / 'errormessage.txt', 'a') as f:
                errormess = f"Error occurred while sweeping '{filename}.mph' in iteration {iteration}"
                f.write(f"{errormess}: \n{str(e)}\n\n\n")
                tl.log_message(errormess)
        tl.log_message(f"Finished iteration {iteration+1}/{len(sweep_values[0])} of sweep for model '{filename}'.")

    # clear model from client after sweep is completed
    client.remove(model)
    tl.log_message(f"Cleared model '{filename}' from client after sweep.")

    endmessage = f"Sweep of model '{filename}' completed."

    if last_log_path is None:
        tl.LOG_PATH = None
    elif new_log_file:
        tl.initialize_time_log(last_log_path, startmessage=endmessage, start_time=last_start_time, last_time=last_last_time)  # restore the previous log file path if it was changed
    

##############################################################################
##############################################################################

def calculate_grid_voltages(
        z_target: float,
        B_goal: tuple[float, float, float],
        conductor_grid_length: float = 500e-6,
        conductor_grid_width: float = 2e-6,
        conductor_grid_height: float = 1e-6,
        conductivity: float = 6.30e7):
    """
    Calculates the 20 boundary terminal voltages required to generate a specific 
    target magnetic field vector (B_goal) at a central spatial point below or above 
    a 5x5 conductor crossbar grid.

    The model dynamically computes trace segment resistances based on material 
    electrical conductivity and geometric cross-sections. Terminal arm segments 
    extending to boundaries have exactly half the electrical resistance of inner segments.

    Terminal Numbering (1-indexed mapping to 0-19 in array):
    Starts at Pin 1 on the bottom-left edge and increments COUNTER-CLOCKWISE:
    - Pins 1 to 5: Bottom boundary (y = -0.5 * L, running left-to-right from index i=0 to 4)
    - Pins 6 to 10: Right boundary (x = 0.5 * L, running bottom-to-top from index j=0 to 4)
    - Pins 11 to 15: Top boundary (y = 0.5 * L, running right-to-left from index i=4 to 0)
    - Pins 16 to 20: Left boundary (x = -0.5 * L, running top-to-bottom from index j=4 to 0)

    Parameters:
    -----------
    z_target : float
        The vertical z-coordinate of the target center point in meters (must be non-zero).
    B_goal : tuple
        The desired 3D magnetic field target vector (Bx, By, Bz) in Tesla.
    conductor_grid_length : float, optional
        The overall side length of the square substrate footprint in meters. Default is 2 mm.
    conductor_grid_width : float, optional
        The cross-sectional width of internal conductor traces in meters. Default is 10 um.
    conductor_grid_height : float, optional
        The physical height/thickness of deposited metal traces in meters. Default is 3 um.
    conductivity : float, optional
        Material bulk electrical conductivity in S/m. Default is 6.30e7 S/m (bulk silver).

    Returns:
    --------
    V_terminals : numpy.ndarray
        An array of 20 calculated optimal voltage values corresponding to Pin 1 through Pin 20.
    """
    if np.isclose(z_target, 0.0):
        raise ValueError("z_target cannot be 0. Within the plane z=0, B_x and B_y are physically zero.")

    # 1. Spatial Geometry Coordinates Setup
    inner_coords = np.linspace(-(5 - 1) / (2 * 5) * conductor_grid_length, (5 - 1) / (2 * 5) * conductor_grid_length, 5)
    
    # 45 Nodes total: 0..24 are inner crossings, 25..44 are open terminal edges
    coords = np.zeros((45, 3))
    for i in range(5):
        for j in range(5):
            coords[i * 5 + j] = [inner_coords[i], inner_coords[j], 0.0]

    terminal_node_indices = np.zeros(20, dtype=int)
    
    # Pins 1 to 5: Bottom (y = -0.5 * L, x moves left-to-right)
    for idx, i in enumerate(range(5)):
        t_idx = 0 + idx
        terminal_node_indices[t_idx] = 25 + t_idx
        coords[terminal_node_indices[t_idx]] = [inner_coords[i], -0.5 * conductor_grid_length, 0.0]
        
    # Pins 6 to 10: Right (x = 0.5 * L, y moves bottom-to-top)
    for idx, j in enumerate(range(5)):
        t_idx = 5 + idx
        terminal_node_indices[t_idx] = 25 + t_idx
        coords[terminal_node_indices[t_idx]] = [0.5 * conductor_grid_length, inner_coords[j], 0.0]
        
    # Pins 11 to 15: Top (y = 0.5 * L, x moves right-to-left)
    for idx, i in enumerate(reversed(range(5))):
        t_idx = 10 + idx
        terminal_node_indices[t_idx] = 25 + t_idx
        coords[terminal_node_indices[t_idx]] = [inner_coords[i], 0.5 * conductor_grid_length, 0.0]
        
    # Pins 16 to 20: Left (x = -0.5 * L, y moves top-to-bottom)
    for idx, j in enumerate(reversed(range(5))):
        t_idx = 15 + idx
        terminal_node_indices[t_idx] = 25 + t_idx
        coords[terminal_node_indices[t_idx]] = [-0.5 * conductor_grid_length, inner_coords[j], 0.0]

    # 2. Material-Based Resistivity & Admittance Computation
    cross_section = conductor_grid_width * conductor_grid_height
    d_inner = inner_coords[1] - inner_coords[0]  # Pitch distance between intersection nodes
    
    # Resistance calculations via Pouillet's Law
    R_internal = d_inner / (conductivity * cross_section)
    R_arm = 0.5 * R_internal  # Terminal arm resistance constraint enforced here

    # 3. Define Trace Segments and Associate Specific Resistance Values
    segments = []
    
    # Horizontal paths (Left Pin -> Intersections -> Right Pin)
    for j in range(5):
        left_pin_node = terminal_node_indices[19 - j]   
        right_pin_node = terminal_node_indices[5 + j]   
        
        # Left boundary terminal arm
        segments.append((left_pin_node, 0 * 5 + j, R_arm))
        # Inner lattice paths
        for i in range(4):
            segments.append((i * 5 + j, (i + 1) * 5 + j, R_internal))
        # Right boundary terminal arm
        segments.append((4 * 5 + j, right_pin_node, R_arm))

    # Vertical paths (Bottom Pin -> Intersections -> Top Pin)
    for i in range(5):
        bottom_pin_node = terminal_node_indices[0 + i]  
        top_pin_node = terminal_node_indices[14 - i]   
        
        # Bottom boundary terminal arm
        segments.append((bottom_pin_node, i * 5 + 0, R_arm))
        # Inner lattice paths
        for j in range(4):
            segments.append((i * 5 + j, i * 5 + (j + 1), R_internal))
        # Top boundary terminal arm
        segments.append((i * 5 + 4, top_pin_node, R_arm))

    # 4. Assemble Internal Kirchhoff Admittance Matrix (25 x 25)
    A_kirchhoff = np.zeros((25, 25))
    for i in range(5):
        for j in range(5):
            n = i * 5 + j
            
            g_left  = 1.0 / R_internal if i > 0 else 1.0 / R_arm
            g_right = 1.0 / R_internal if i < 4 else 1.0 / R_arm
            g_down  = 1.0 / R_internal if j > 0 else 1.0 / R_arm
            g_up    = 1.0 / R_internal if j < 4 else 1.0 / R_arm
            
            A_kirchhoff[n, n] = g_left + g_right + g_down + g_up
            
            if i > 0: A_kirchhoff[n, (i - 1) * 5 + j] = -1.0 / R_internal
            if i < 4: A_kirchhoff[n, (i + 1) * 5 + j] = -1.0 / R_internal
            if j > 0: A_kirchhoff[n, i * 5 + (j - 1)] = -1.0 / R_internal
            if j < 4: A_kirchhoff[n, i * 5 + (j + 1)] = -1.0 / R_internal

    # 5. FIXED: Superposition Loop to Construct Transmission Coupling Matrix M (3 x 20)
    M = np.zeros((3, 20))
    r_target = np.array([0.0, 0.0, z_target])
    mu_0_over_4pi = 1e-7

    for b_idx in range(20):
        V_boundary_excitation = np.zeros(20)
        V_boundary_excitation[b_idx] = 1.0
        
        b_kirchhoff = np.zeros(25)
        
        # Enforcing a single strict loop configuration using pure index equations
        for i in range(5):
            # Bottom Edge: Pins 1..5 map to inner nodes (i, 0)
            b_kirchhoff[i * 5 + 0] += V_boundary_excitation[0 + i] / R_arm
            
            # Right Edge: Pins 6..10 map to inner nodes (4, i)
            b_kirchhoff[4 * 5 + i] += V_boundary_excitation[5 + i] / R_arm
            
            # Top Edge: Pins 11..15 map to inner nodes (4-i, 4) -> running right-to-left
            b_kirchhoff[(4 - i) * 5 + 4] += V_boundary_excitation[10 + i] / R_arm
            
            # Left Edge: Pins 16..20 map to inner nodes (0, 4-i) -> running top-to-bottom
            b_kirchhoff[0 * 5 + (4 - i)] += V_boundary_excitation[15 + i] / R_arm
            
        V_inner = np.linalg.solve(A_kirchhoff, b_kirchhoff)
        
        V_all = np.zeros(45)
        V_all[0:25] = V_inner
        for t_idx in range(20):
            V_all[terminal_node_indices[t_idx]] = V_boundary_excitation[t_idx]
            
        B_column = np.zeros(3)
        for node_start, node_end, r_seg in segments:
            current = (V_all[node_start] - V_all[node_end]) / r_seg
            r_mid = (coords[node_start] + coords[node_end]) / 2.0
            dl = coords[node_end] - coords[node_start]
            R_vec = r_target - r_mid
            R_mag = np.linalg.norm(R_vec)
            
            dB = mu_0_over_4pi * current * np.cross(dl, R_vec) / (R_mag**3)
            B_column += dB
            
        M[:, b_idx] = B_column

    # 6. Inversion Matrix Solution Using Pseudoinverse
    M_pinv = np.linalg.pinv(M)
    V_terminals = np.dot(M_pinv, B_goal)
    
    return V_terminals

def calculate_xy_vector(alpha: float, magnitude: float, in_degrees: bool = True):
    """
    Calculates a normalized 3D vector in the xy-plane.
    
    :param alpha: The angle relative to the vector (1,0,0)
    :param magnitude: The magnitude of the vector
    :param in_degrees: If True, angle is expected in degrees. 
                       If False, in radians.
    :return: A tuple (x, y, z) representing the normalized vector.
    """
    if in_degrees:
        alpha = math.radians(alpha)
        
    x = math.cos(alpha) * magnitude
    y = math.sin(alpha) * magnitude
    z = 0.0
    
    return (x, y, z)

def get_voltage_sweep_dict(
        angles: list,
        magnitude: float = 10e-6,
        conductor_grid_length: float = 500e-6
        ):
    """
    Generates a dictionary containing the calculated terminal voltages for a sweep of angles in the xy-plane.

    Args:
        angles (list): A list of angles (in degrees) for which to calculate the terminal voltages.
        magnitude (float): The magnitude of the target magnetic field vector. Default is 10e-6 Tesla.
        conductor_grid_length (float): The length of the conductor grid. Default is 500e-6 meters.

    Returns:
        dict: A dictionary where keys are terminal names (e.g., "V01", "V02", ..., "V20") and values are lists of calculated voltages corresponding to each angle in the sweep.
    """
    
    sweep_dict = {}
    num_iterations = len(angles)

    for iteration, alpha in enumerate(angles):
        # 1. Calculate target vector
        b_goal = calculate_xy_vector(alpha, magnitude, in_degrees=True)

        # 2. Simulate grid voltages
        v_terminals = calculate_grid_voltages(
            z_target=1e-3,
            B_goal=b_goal,
            conductor_grid_length=conductor_grid_length,
            conductor_grid_width=2e-6,
            conductor_grid_height=1e-6,
            conductivity=61.6e6,
        )
        
        # 3. Store values in the dictionary
        for terminal, v in enumerate(v_terminals):
            key = f"V{terminal+1:02d}"
            
            # Initialize the list on the first iteration
            if key not in sweep_dict:
                sweep_dict[key] = [None] * num_iterations
            
            # Assign the value to the correct index position

            sweep_dict[key][iteration] = f"{v:.4g}[V]"

    return sweep_dict
