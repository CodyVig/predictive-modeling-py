"""
R data files (`.Rds`) are binary files that are notoriously difficult to open in
Python. Te solution I was able to find on Windows was a two step process:

1. Add `R_HOME` to the system environment variables. This is the path to the R
installation directory.

2. Run the code in this module.

The `Mouse` class defined in `scripts/mouse.py` uses this module to open the RDS
files for a given session and store the data in a more easily accessible format.

The data structure of the RDS file is as follows:

- `session[0]` - `contrast_left`: contrast of the left stimulus
- `session[1]` - `contrast_right`: contrast of the right stimulus
- `session[2]` - `feedback_type`: type of the feedback, 1 for success and -1 for failure
- `session[3]` - `mouse_name`: The name of the mouse
- `session[4]` - `brain_area`: area of the brain where each neuron lives
- `session[5]` - `date_exp`: The date of the experiment
- `session[6]` - `spks`: numbers of spikes of neurons in the visual cortex in time bins defined in time
- `session[7]` - `time`: centers of the time bins for spks
"""

import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
import rpy2.robjects.pandas2ri


def get_session_file(session: int) -> rpy2.robjects.vectors.ListVector:
    """
    Opens the RDS file for the specified session and returns the data as a
    ListVector object (essentially a list of DataFrames).

    :param session: The session number for which to open the RDS file.
    :return: The data from the RDS file as a ListVector object.
    """

    try:
        session = int(session)
    except ValueError:
        print("Session must be an integer")
        return []

    if session < 1 or session > 19:
        print("Invalid session number. Must be between 1 and 19.")
        return []

    # Load required R package
    utils = rpackages.importr("utils")

    # Select a CRAN mirror
    utils.chooseCRANmirror(ind=1)

    # Define the file path
    file_path = f"data/session{str(session)}.rds"

    # Read the RDS file
    readRDS = ro.r["readRDS"]  # Get the R function readRDS
    data = readRDS(file_path)

    # Convert the data to a Pandas DataFrame if it's a data frame in R
    rpy2.robjects.pandas2ri.activate()

    return rpy2.robjects.pandas2ri.rpy2py(data)
