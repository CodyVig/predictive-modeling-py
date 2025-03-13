"""
This module contains the Mouse class, which is used to represent a single mouse
from a single session from the study conducted by Steinmetz et al. (2019).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scripts.open_rds import get_session_file


class Mouse:
    def __init__(self, session):
        """
        Initializes a Mouse object with the specified session number.

        In the study conducted by Steinmetz et al. (2019), experiments were
        performed on a total of 10 mice over 39 sessions. Each session comprised
        several hundred trials, during which visual stimuli were randomly
        presented to the mouse on two screens positioned on both sides of it.
        The stimuli varied in terms of contrast levels, which took values in
        {0, 0.25, 0.5, 1}, with 0 indicating the absence of a stimulus.
        The mice were required to make decisions based on the visual stimuli,
        using a wheel controlled by their forepaws. A reward or penalty (i.e.,
        feedback) was subsequently administered based on the outcome of their
        decisions.

        This class contains all of the measured values for a single mouse (i.e.,
        a single session) for one of the 18 session we have data for.
        """

        self.session_number = session
        self.session = get_session_file(session)
        self.mouse_name = self._get_name(self.session)
        self.date_exp = self._get_date(self.session)
        self.contrast_left = self._get_contrast_left(self.session)
        self.contrast_right = self._get_contrast_right(self.session)
        self.feedback_type = self._get_feedback_type(self.session)
        self.spikes = self._get_spikes(self.session)
        self.time = self._get_time(self.session)
        del self.session  # Remove the session data to save memory

    def __str__(self):
        return f"{self.mouse_name} on {self.date_exp.strftime('%m/%d/%Y')}"

    def __repr__(self):
        return f"Mouse({self.session_number})"

    # Recall that the data is structured as follows:
    # `session[0]` - `contrast_left`
    # `session[1]` - `contrast_right`
    # `session[2]` - `feedback_type`
    # `session[3]` - `mouse_name`
    # `session[4]` - `brain_area`
    # `session[5]` - `date_exp`
    # `session[6]` - `spks`
    # `session[7]` - `time`

    def _get_name(self, session):
        return session[3][0]

    def _get_date(self, session):
        return datetime.strptime(session[5][0], "%Y-%m-%d").date()

    # The `contrast_left`, `contrast_right`, and `feedback_type` have
    # inconsistent data types for sessions 1 and 18, because test data was
    # pulled from these sessions before the files were distributed to us.
    # These fields, for sessions 2-17, are numpy arrays that contain other numpy
    # arrays. For sessions 1 and 18, they are are numpy arrays that contain
    # lists. Either way, they need to be flattened to a nunmpy array of floats.

    def _get_contrast_left(self, session):
        if isinstance(session[0][0], np.ndarray):
            return np.array([item for sublist in session[0] for item in sublist])
        else:
            return np.array([item for item in session[0]])

    def _get_contrast_right(self, session):
        if isinstance(session[1][0], np.ndarray):
            return np.array([item for sublist in session[0] for item in sublist])
        else:
            return np.array([item for item in session[0]])

    def _get_feedback_type(self, session):
        if isinstance(session[2][0], np.ndarray):
            return np.array([item for sublist in session[0] for item in sublist])
        else:
            return np.array([item for item in session[0]])

    def _get_spikes(self, session):
        """
        This is the first significant deviation from the R data file structure.
        R keeps the brain areas and spikes separate, but it would be much easier
        to combine them into a single DataFrame for each trial. This method does
        that.
        """
        trials = {}
        for idx, trial in enumerate(session[6]):
            spikes = pd.DataFrame(
                trial,
                index=pd.Series(session[4]),
                dtype="category"
            )
            spikes.columns = [f"Time Bin {i+1}" for i in range(spikes.shape[1])]
            trials[f"Trial {idx + 1}"] = spikes
        return trials

    def _get_time(self, session):
        """
        This is another large departure from the R data file structure. I am
        keeping all of the time midpoints in a single DataFrame, rather than
        keeping them separate for each trial. The rows of this dataframe index
        the trials, and the columns index the time bins.
        """
        trials = pd.DataFrame()
        for idx, trial in enumerate(session[7]):
            times = pd.DataFrame(trial).T
            times.columns = [f"Time Bin {i+1}" for i in range(times.shape[1])]
            times.index = [f"Trial {idx+1}"]
            trials = pd.concat([trials, times])
        return trials
