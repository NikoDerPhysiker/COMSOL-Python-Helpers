# Author: Niko Bleidistel
# last change: 2026-06-26

##############################################################################
# import packages
##############################################################################

import mph                              # api for comsol multiphysics

import pandas as pd                     # pandas for easy DataAnalysis
from IPython.display import display     # for displaying DataFrames

from os import makedirs                 # for creating directories
import pathlib                          # for handling file paths
import csv                              # for handling csv files

import re                               # for regular expressions

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
        # print(f'Loading model from path: {path}')
        model = client.load(path)
        model_list.append(model)

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

def save_Parameter_List_to_CSV(model: mph.Model, 
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

    if printFeedback:
        print(f"TXT file '{txt_file_path}' successfully converted to CSV file '{csv_file_path}'.")

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
    if printFeedback:
        print(f"CSV file '{csv_file_path}' successfully converted to TXT file '{txt_file_path}'.")

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

    if printFeedback:
        print(f"Sweep parameter(s) '{parameternames}' with value(s) '{parametervalues}' added to feature '{featurename}' of study '{studyname}' of the model '{model.java.label()}'.")

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
        return
    
    # Get the previous preference state for excluding data in MPH files
    previous_exclude_state = model.java.excludeComputedDataInMph()

    try:
        # set whether to exclude computed, mesh, and plot data during the save operation based on the smallerFilesize flag
        model.java.excludeComputedDataInMph(smallerFilesize)

        # Save the model to the new path without modifying the active RAM state
        model.java.save(full_export_path, True)
        
    finally:
        # Revert the exclusion setting to its original state
        model.java.excludeComputedDataInMph(previous_exclude_state)

        

##############################################################################
##############################################################################

def override_export_variables(model: mph.Model, 
                              export_node_name: str,
                              expressions: list,
                              dataset_identifier: str,
                              descriptions: list | None = None,
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
    
    # find dateset tag based on the provided dataset identifier and set it in the export node
    # get all dataset tags and labels
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
                print(f"Dataset: '{dataset_identifier}' has been identified as COMSOL-Tag '{d_tag}'.")
                break
    
    if actual_dataset_tag is not None:
        target_node.set("data", actual_dataset_tag)
    else:
        print(f"Warning: Dataset '{dataset_identifier}' not found. Using default setting.")

    # (over-)write expressions and descriptions into the export node
    if descriptions is None:
        descriptions = [""] * len(expressions)
    target_node.set("expr", expressions)
    target_node.set("descr", descriptions)
    
    print(f"Variables successfully added to export '{export_node_name}'.")

##############################################################################
##############################################################################

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
    # check model and client instances
    if not isinstance(model, mph.Model):
        raise TypeError("The argument 'model' needs to be an instance of 'mph.Model'.")
    if not isinstance(client, mph.Client):
        raise TypeError("The argument 'client' needs to be an instance of 'mph.Client'.")
    if not isinstance(studyname, str):
        raise TypeError("The argument 'studyname' needs to be a string.")
    if dataset_identifier is not None and not isinstance(dataset_identifier, str):
        raise TypeError("The argument 'dataset_identifier' needs to be a string or None.")
    if  not isinstance(export_path, str):
        raise TypeError("The argument 'export_path' needs to be a string.")
    if not isinstance(parameternames, list):
        raise TypeError("The argument 'parameternames' needs to be a list of strings.")
    if not isinstance(parametervalues, list):
        raise TypeError("The argument 'parametervalues' needs to be a list of lists of strings.")
    if not isinstance(expressions, list):
        raise TypeError("The argument 'expressions' needs to be a list of strings.")
    for name in parameternames:
        if not isinstance(name, str):
            raise TypeError("All elements in 'parameternames' need to be strings.")
    for value_list in parametervalues:
        if not isinstance(value_list, list):
            raise TypeError("All elements in 'parametervalues' need to be lists of strings.")
    for expr in expressions:
        if not isinstance(expr, str):
            raise TypeError("All elements in 'expressions' need to be strings.")
    if dataset_identifier is not None and not isinstance(dataset_identifier, str):
        raise TypeError("The argument 'dataset_identifier' needs to be a string or None.")
    
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
    print(f"Starting parameter sweep for study '{studyname}' with parameters {parameternames}.\n")
    first_iteration = True
    for iteration in range(len(parametervalues[0])):
        print(f"Iteration {iteration+1}/{len(parametervalues[0])} with parameter values {[paramvalues[iteration] for _, paramvalues in sweep_parameters]}.")

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
        output_csv_path = folder_path / f"{MODELNAME}_iteration_{iteration}_parameters.csv"
        save_Parameter_List_to_CSV(model, str(output_csv_path), displayParams=False)

        if first_iteration:
            first_iteration = False
            df_parameters = pd.read_csv(output_csv_path)
            expressions.extend([f"root.{param}" for param in df_parameters["name"].tolist() if f"root.{param}" not in expressions])


        # solve the study
        print(f"Running study '{studyname}'...")
        try:
            model.solve(studyname)
        except Exception as e:
            print(f"ERROR occurred while solving study '{studyname}' for iteration {iteration+1}: \n\n{e}\n\n")

            with open(str(folder_path / "errormessage.txt"), "w", encoding="utf-8") as destination:
                destination.write(f"ERROR occurred while solving study '{studyname}' for iteration {iteration+1}:\n\n")
                destination.write(str(e))
            
            print(f"Error message saved to {folder_path / 'errormessage.txt'}")
            if saveSolutions:
                print(f"saving failed model to {path}...")
                save_as_copy(model, client, str(path)+".mph", smallerFilesize = False)

            print("Not generating data export and continuing with the next one.")
            continue  # Skip to the next iteration if an error occurs

        # export the results with the current parameter values
        print(f"Exporting results to {export_path}...")
        EXPORT_NODE_NAME="PySweepExport"
        
        
        filename = f"{MODELNAME}_iteration_{iteration}"
        path = folder_path / filename
        
        override_export_variables(model, export_node_name=EXPORT_NODE_NAME, expressions=expressions , dataset_identifier=dataset_identifier)
        model.export(EXPORT_NODE_NAME, str(path) + ".txt")

        if saveSolutions:
            print(f"saving model solution to {path}...")
            save_as_copy(model, client, str(path)+".mph", smallerFilesize = False)
            paths.append(path)

        print(f"Iteration {iteration+1} completed.\n")

    print("Parameter sweep completed.")
    return paths
        

##############################################################################
##############################################################################