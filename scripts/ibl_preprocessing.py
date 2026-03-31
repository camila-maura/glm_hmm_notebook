# %% [markdown]
# # IBL Preprocessing
# %%
# Imports
from one.api import ONE
import numpy as np
from ashwood_preprocessing_utils import remap_choice_vals, create_previous_choice_vector, create_wsls_covariate
from sklearn import preprocessing

# %% tags=["hide-input"]
def get_data_this_session(eid, df_trials):
    df_sess = df_trials[df_trials["session"] == eid]

    # We can select all the necessary values for the design matrix: choice, contrast of stimuli, reward and bias probs
    stim_left = df_sess['contrastLeft'].to_numpy()
    stim_right = df_sess['contrastRight'].to_numpy()
    rewarded = df_sess['feedbackType'].to_numpy()
    choice = df_sess['choice'].to_numpy()
    
    return choice, stim_left, stim_right, rewarded
        
def create_stim_vector(stim_left, stim_right):
    # Create stim vector
    stim_left = np.nan_to_num(stim_left, nan=0)
    stim_right = np.nan_to_num(stim_right, nan=0)
    # now get 1D stim
    signed_contrast = stim_right - stim_left
    
    return signed_contrast

def create_design_matrix(choice, stim_left, stim_right, rewarded):
    # Stimuli predictor
    signed_contrast = create_stim_vector(stim_left, stim_right)
    T = len(signed_contrast)
    design_mat = np.zeros((T, 3))
    design_mat[:, 0] = signed_contrast    
        
    # make choice vector so that correct response for stim>0 is choice =1
    # and is 0 for stim <0 (viol is mapped to -1)
    choice = remap_choice_vals(choice)

    previous_choice, locs_mapping = create_previous_choice_vector(choice)

    # create wsls vector:
    wsls = create_wsls_covariate(previous_choice, rewarded, locs_mapping)

    # map previous choice to {-1,1}
    design_mat[:, 1] = 2 * previous_choice - 1
    design_mat[:, 2] = wsls
    
    return design_mat


def get_all_unnormalized_data_this_session(eid, df_trials):
    choice, stim_left, stim_right, rewarded = get_data_this_session(eid, df_trials)
    unnormalized_design_matrix = create_design_matrix(choice, stim_left, stim_right, rewarded)
    y = np.expand_dims(remap_choice_vals(choice), axis=1)
    session = [eid for i in range(y.shape[0])]
    rewarded = np.expand_dims(rewarded, axis=1)
    return unnormalized_design_matrix, y, session, rewarded

# %% [markdown]
# Using ONE's ```load_aggregate``` function, we can retrieve all sessions from a given animal. For this, first we need to instantiate the ```ONE``` object
# %%
one = ONE()

# %% [markdown]
# Then we need to choose our subject and run ```load_aggregate```
# %% 
subject = "CSHL_008"
trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')
print(f"Total # of sessions {len(trials["session"].unique())}")
# %% [markdown]
#! Admonition one.search() returns session IDs (eids) that exist as session records in Alyx, while load_aggregate() downloads a pre computed file with trial data pooled across multiple sessions. If you want to get all sessions from a single animal, it is recommended to use ```load_aggregate```, because some sessions may be located in a dataframe without a session identified in itself (but containing multiple sessions with their own session identifiers). 

# %% [markdown]
# We can see the information we get by printing the columns
# %% 
print(trials.columns)

# %% [markdown]
# Modeling choice as result of observables and behavioral state. We need choice, stimuli presented and reward obtained. Additionally, we want to keep the session and 
# Let's extract the meaningful data: session, choice, stimulus presented on the left, stimulus presented on the right, reward obtained and probability of reward
# %%
trials = trials[["session", "choice", "contrastLeft", "contrastRight", "feedbackType", "probabilityLeft", "session_start_time"]]
# %%
print(trials["session_start_time"].unique())

# %% [markdown]
# We can also see what we have if we select a single session
# %%
# Create a list of ids
sessions_ids = trials.session.unique()       

# %% [markdown]
# In Ashwood, only the sessions with less than 10 violations were used. Thus, we will now revise the number of violations, defined as a trial where the animal made no choice. i.e choice == 0 during the 50-50 trials. For that, 1) 50-50 trials must be present and 2) there must be less than 10 violations in that subset of trials

# During the first 90 trials of each session, the stimulus appeared randomly on the left or right side of the screen with probability 0.5. Subsequent trials were generated in blocks in which the stimulus appeared on one side with probability 0.8, alternating randomly every 20–100 trials. We analyzed data from animals with at least 3,000 trials of data (across multiple sessions) after they had successfully learned the task (Supplementary Figs. 1 and 2). For each animal, we considered only the data from the first 90 trials of each session, when the stimulus was equally likely to appear on the left or right of the screen.
# %%
# keep only relevant columns for filtering
df_trials = trials[["session", "probabilityLeft", "choice"]]

# %% [markdown] 
# Why do we keep only these sessions?
# %%
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
# %% [markdown]
# Now we combine the two conditions

# %% 
valid_sessions = violations[
    (violations < 10) & (violations.index.isin(valid_prob_sessions[valid_prob_sessions == True].index))
].index.tolist()

# %% [markdown]
# and make sure they maintain the order of the original dataset (we don't want scrambled trials)

# %%
# Maintain original order
valid_set = set(valid_sessions)
valid_sessions = [
    s for s in trials["session"].drop_duplicates()
    if s in valid_set
]
print(len(valid_sessions))

print(f"current # of sessions {len(df_trials.session.unique())}")

# %% [markdown]
# Now we can select only the valid sessions for subsequent analyses

# %%
df_trials = trials[
    (trials["session"].isin(valid_sessions)) & (df_trials["probabilityLeft"] == 0.5)
]
print(f"current # of sessions {len(df_trials.session.unique())}")

# %% [markdown]
# Now, with the valid sessions, we can compute the design matrix. We are only interested in the 50-50 trials. Let's do it for a single session

# %%
# Select an example session
eid = valid_sessions[0]     
# Filter that session
df_sess = df_trials[df_trials["session"] == eid]

# %% [markdown]
# We can select all the necessary values for the design matrix: choice, contrast of stimuli and reward
# %%
choice = df_sess['choice'].to_numpy()
stim_left = df_sess['contrastLeft'].to_numpy()
stim_right = df_sess['contrastRight'].to_numpy()
rewarded = df_sess['feedbackType'].to_numpy()
#bias_probs = df_sess['probabilityLeft'].to_numpy() #TODO I think this one is unnecesary

# %% [markdown]
# First we use the stimuli contrast from left and right to build a single ```stimuli_contrast``` vector. We are doing this to convert two separate sensory inputs (left and right stimulus strengths) into a single variable that represents both evidence magnitude and decision direction. This reduces redundancy in the design matrix and makes the predictor directly interpretable: positive values encode right evidence, negative values encode left evidence, and zero encodes neutrality or missing information (after NaN handling).
# %%
# Create stim vector
stim_left = np.nan_to_num(stim_left, nan=0)
stim_right = np.nan_to_num(stim_right, nan=0)

# now get 1D stim
signed_contrast = stim_right - stim_left

# %%
# Create matrix to be filled with the predictors
n_trials = len(signed_contrast)
design_mat = np.zeros((n_trials, 3))
# And add ```signed_contrast``` in the first dimension
design_mat[:, 0] = signed_contrast

# %% [markdown]
# We created our design matrix and completed the first column. Now we have to fill it with the other two predictors of choice: previous choice and win stay lose shift. 

# EXPLANATION FOR WHY WE ARE CHOOSING THOSE

# We will start with the previous choice predictor
# make choice vector so that correct response for stim>0 is choice =1
# and is 0 for stim <0 (viol is mapped to -1)
# %% 
choice = remap_choice_vals(choice)

previous_choice, locs_mapping = create_previous_choice_vector(choice)

# map previous choice to {-1,1}
design_mat[:, 1] = 2 * previous_choice - 1

# %% [markdown]
# Now win stay lose shift vector
# %%
# create wsls vector:
wsls = create_wsls_covariate(previous_choice, rewarded, locs_mapping)

design_mat[:, 2] = wsls

#y = np.expand_dims(remap_choice_vals(choice), axis=1)
#session = [eid for i in range(y.shape[0])]
#rewarded = np.expand_dims(rewarded, axis=1)

# %% [markdown]
# We then normalize our stimuli value across  for some reason? This does not yield the exact same value as ashwood because she normalizes across all animals but its quite close anyway.
unnormalized_inpt = design_mat.copy()
normalized_inpt = np.copy(unnormalized_inpt)
normalized_inpt[:, 0] = preprocessing.scale(normalized_inpt[:, 0])

# %% [markdown]
# We will carry out the exact same process but for all trials. Our design matrix each row will be a trial, so effectiveli this means that we will carry out the same process up until the normalization and then normalize.
# %% tags=["hide-input"]
def get_unnormalized_design_mat(valid_sessions, df_trials):
    sess_counter = 0
    for eid in valid_sessions:
        unnormalized_inpt, y, session, rewarded = \
            get_all_unnormalized_data_this_session(
                eid, df_trials)
        if sess_counter == 0:
            animal_unnormalized_inpt = np.copy(unnormalized_inpt)
            animal_y = np.copy(y)
            animal_session = session
            animal_rewarded = np.copy(rewarded)
        else:
            animal_unnormalized_inpt = np.vstack(
                (animal_unnormalized_inpt, unnormalized_inpt))
            animal_y = np.vstack((animal_y, y))
            animal_session = np.concatenate((animal_session, session))
            animal_rewarded = np.vstack((animal_rewarded, rewarded))
        sess_counter += 1
    # Normalize
    animal_normalized_inpt = np.copy(animal_unnormalized_inpt)
    
    animal_normalized_inpt[:, 0] = preprocessing.scale(animal_unnormalized_inpt[:, 0])
    
    return animal_unnormalized_inpt, animal_normalized_inpt, animal_y, animal_session

# %% 
animal_unnormalized_inpt, animal_normalized_inpt, animal_y, animal_session = get_unnormalized_design_mat(valid_sessions, df_trials)

# %% 
# Write out animal's unnormalized data matrix:
np.savez(
    'scripts/IBL/_unnormalized.npz',animal_unnormalized_inpt, 
    animal_normalized_inpt, 
    animal_y, 
    animal_session 
)

