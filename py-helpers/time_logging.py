# Author: Niko Bleidistel
# last change: 2026-08-02

##############################################################################
# import packages
##############################################################################

from pathlib import Path
import time
import logging
import csv

##############################################################################
# global variables set in initialize_time_log() and used in other functions
##############################################################################
LOG_PATH: Path | None = None
START_TIME: float | None = None
LAST_TIME: float | None = None

##############################################################################
##############################################################################
# functions
##############################################################################
##############################################################################

def initialize_time_log(
        log_path: Path,
        startmessage: str = "Started running Python script",
        start_time: float | None = None,
        last_time: float | None = None
        ):
    """
    Initializes the time logging system.

    The initialization only executes if it has not run during the current
    session (i.e., START_TIME or LAST_TIME is None) OR if a new log_path is
    provided.

    If the log file does not exist or is empty, a CSV header will be created and the start message will be logged. 
    If the log file already exists and is not empty, the function will do nothing.

    Args:
        log_path (Path): The path to the file where time logs will be stored.
        startmessage (str): The message to log at the start of the script.
          Defaults to "Started running Python script".

    Returns:
        tuple: A tuple containing the "start time" and "last logged time".
    """
    global LOG_PATH
    global START_TIME
    global LAST_TIME

    # Guard clause: Skip execution if already initialized AND the path hasn't changed
    if (
        START_TIME is not None
        and LAST_TIME is not None
        and LOG_PATH == log_path
    ):
        return START_TIME, LAST_TIME

    # Update the global path since we are initializing or switching files
    LOG_PATH = log_path

    # Create the CSV header only if the file is new or empty
    if (not LOG_PATH.exists()) or (LOG_PATH.stat().st_size == 0):
        with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                "Timestamp",
                "Log-Level",
                "Message",
                "Since Start (s)",
                "Since Start (H:M:S.ms)",
                "Since Last Log (s)",
                "Since Last Log (H:M:S.ms)",
            ])

    # Configure the standard logging library to output to the specified path
    logging.basicConfig(
        filename=LOG_PATH,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Overwrites any existing root logger configuration
    )

    # Reset and start the timers
    if start_time is None or last_time is None:
        START_TIME = time.perf_counter()
        LAST_TIME = START_TIME
    else:
        START_TIME = start_time
        LAST_TIME = last_time

    # Log the initial entry and update the last logged time tracker
    LAST_TIME = log_time(startmessage, START_TIME, LAST_TIME)

    return START_TIME, LAST_TIME
    
##############################################################################
##############################################################################


def format_hms_ms(seconds: float) -> str:
    """
    Converts seconds into HH:MM:SS.mmm format including milliseconds.
    
    Args:
        seconds (float): The time duration in seconds.
    
    Returns:
        str: The formatted time string in HH:MM:SS.mmm format.
    """

    hms = time.strftime('%H:%M:%S', time.gmtime(seconds))
    milliseconds = int((seconds % 1) * 1000)
    return f"{hms}.{milliseconds:03d}"

##############################################################################
##############################################################################

def log_time(message: str, start_time: float, last_time: float) -> float:
    """
    Logs the elapsed time since the start and since the last log.
    
    Args:
        message (str): The message to log.
        start_time (float): The time when the script started.
        last_time (float): The time when the last log was made.
    
    Returns:
        float: The current time after logging, to be used as the new last_time.
    """
    current_time = time.perf_counter()
    elapsed_since_start = current_time - start_time
    elapsed_since_last = current_time - last_time

    # Convert durations to HH:MM:SS.ms
    hms_start = format_hms_ms(elapsed_since_start)
    hms_last = format_hms_ms(elapsed_since_last)

    # Log all values separated by semicolons
    logging.info(
        f"{message};"
        f"{elapsed_since_start:.4f};{hms_start};"
        f"{elapsed_since_last:.4f};{hms_last}"
    )

    return current_time

##############################################################################
def log_message(message: str):
    """
    Logs a message with the elapsed time since the start and since the last log.
    After initialization, this function should be used for logging instead of calling log_time directly.
    """
    global LAST_TIME # use the global LAST_TIME variable instead of a local one

    if START_TIME is None or LAST_TIME is None:
        raise ValueError("Time logging system not initialized. Call \"initialize_time_log()\" first.")

    LAST_TIME = log_time(message, START_TIME, LAST_TIME)

##############################################################################
##############################################################################
