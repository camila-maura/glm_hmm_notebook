# Imports
from one.api import ONE
import os
import numpy as np
import json
import time
from ashwood_preprocessing_utils import remap_choice_vals, create_previous_choice_vector, create_wsls_covariate

from sklearn import preprocessing

# Using ONE's ```load_aggregate``` function, we can retrieve all sessions from a given animal. For this, first we need to instantiate the ```ONE``` object
one = ONE()

# Then we need to choose our subject and run ```load_aggregate```
subject = "CSHL_008"
trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')
print(f"Total # of sessions {len(trials["session"].unique())}")

#! Admonition one.search() returns session IDs (eids) that exist as session records in Alyx, while load_aggregate() downloads a pre computed file with trial data pooled across multiple sessions. If you want to get all sessions from a single animal, it is recommended to use ```load_aggregate```, because some sessions may be located in a dataframe without a session identified in itself (but containing multiple sessions with their own session identifiers). 

# We are only interested in the biased choice trials, as in Ashwood
# Pending: explanation!
# trials_biased = trials #[trials["task_protocol"]=="biasedChoiceWorld"]
# For now I'm not filtering because I need the .5 proba left to rule out sessions

# We can see the information we get by printing the columns
print(trials.columns)

# Let's extract the meaningful data: session, choice, stimulus presented on the left, stimulus presented on the right, reward obtained and probability of reward
trials = trials[["session", "choice", "contrastLeft", "contrastRight", "feedbackType", "probabilityLeft"]]

# We can also see what we have if we select a single session
sessions_ids = trials.session.unique()       # Create a list of ids
print(sessions_ids)

valid_sessions = []
t0 = time.time()

# Session index
chosen_session = 0
session = trials[trials.session == sessions_ids[chosen_session]]   # Choose the first session for display
#print(session.head(5))

#print(session["probabilityLeft"].unique())

# In Ashwood, only the sessions with less than 10 violations were used. Thus, we will now revise the number of violations, defined as a trial where the animal made no choice. i.e choice == 0 during the 50-50 trials. For that, 1) 50-50 trials must be present and 2) there must be less than 10 violations in that subset of trials

# During the first 90 trials of each session, the stimulus appeared randomly on the left or right side of the screen with probability 0.5. Subsequent trials were generated in blocks in which the stimulus appeared on one side with probability 0.8, alternating randomly every 20–100 trials. We analyzed data from animals with at least 3,000 trials of data (across multiple sessions) after they had successfully learned the task (Supplementary Figs. 1 and 2). For each animal, we considered only the data from the first 90 trials of each session, when the stimulus was equally likely to appear on the left or right of the screen.

# keep only relevant columns for filtering
df_trials = trials[["session", "probabilityLeft", "choice"]]

# Check which sessions contain exactly {0.2, 0.5, 0.8}
valid_prob_sessions = (
    df_trials.groupby("session")["probabilityLeft"]
      .agg(lambda x: set(x.unique()) == {0.2, 0.5, 0.8})
)

# Compute violations only on 50-50 trials
violations = (
    df_trials[df_trials["probabilityLeft"] == 0.5]
    .groupby("session")["choice"]
    .apply(lambda x: (x == 0).sum())
)

# Combine conditions
valid_sessions = violations[
    (violations < 10) & (violations.index.isin(valid_prob_sessions[valid_prob_sessions == True].index))
].index.tolist()

print(len(valid_sessions))
print(valid_sessions)

# Now, with the valid sessions, we can compute the design matrix. We are only interested in the 50-50 trials

df_trials = trials[trials["session"].isin(valid_sessions)].copy()

#print(trials)

#df_valid_sessions = trials[trials["session"].isin(valid_sessions)]

#print("valid \n", df_valid_sessions)

#design_matrix = df_trials[df_trials["probabilityLeft"] == 0.5].groupby("session")

#print(design_matrix)
eid = valid_sessions[0]

print(df_trials["choice"].unique())

#### Create design matrix for a session
df_sess = df_trials[
    (df_trials["session"] == eid) &
    (df_trials["probabilityLeft"] == 0.5)
]
print(df_sess["choice"])

choice = df_sess['choice'].to_numpy()
stim_left = df_sess['contrastLeft'].to_numpy()
stim_right = df_sess['contrastRight'].to_numpy()
rewarded = df_sess['feedbackType'].to_numpy()
bias_probs = df_sess['probabilityLeft'].to_numpy()
# Create stim vector
stim_left = np.nan_to_num(stim_left, nan=0)
stim_right = np.nan_to_num(stim_right, nan=0)
# now get 1D stim
signed_contrast = stim_right - stim_left

T = len(signed_contrast)
design_mat = np.zeros((T, 3))
design_mat[:, 0] = signed_contrast

# make choice vector so that correct response for stim>0 is choice =1
    # and is 0 for stim <0 (viol is mapped to -1)

print(choice)
choice = remap_choice_vals(choice)

previous_choice, locs_mapping = create_previous_choice_vector(choice)

# create wsls vector:
wsls = create_wsls_covariate(previous_choice, rewarded, locs_mapping)
# map previous choice to {-1,1}
design_mat[:, 1] = 2 * previous_choice - 1
design_mat[:, 2] = wsls

unnormalized_inpt = design_mat.copy()

y = np.expand_dims(remap_choice_vals(choice), axis=1)
session = [eid for i in range(y.shape[0])]
rewarded = np.expand_dims(rewarded, axis=1)

normalized_inpt = np.copy(unnormalized_inpt)
normalized_inpt[:, 0] = preprocessing.scale(normalized_inpt[:, 0])

print(unnormalized_inpt)