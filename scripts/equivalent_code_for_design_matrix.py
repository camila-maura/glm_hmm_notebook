### This will not run standalone.

# Shift the array elements one position to the right
previous_choice = np.roll(choice[valid_choices_idx], 1)
# Set first choice as first previous_choice
previous_choice[0] = choice[valid_choices_idx][0]
print(previous_choice)

# Now we will compute the final predictor: win-stay lose-shift
# choice change of mapping will also change this probably
# remap previous choice vals to {-1, 1} to match 1 -> rightward evidence and -1 -> leftward evidence
remapped_previous_choice = previous_choice
# Keep only rewards corresponding to valid trials
valid_rewards = rewarded[valid_choices_idx]
# Shift the array elements one position to the right
previous_reward = np.roll(valid_rewards, 1)
# Set first choice as first previous_choice
previous_reward[0] = valid_rewards[0]
# wsls: vector of size T, entries are in {-1, 1}.  1 corresponds to previous choice = right and success OR previous choice = left and failure; -1 corresponds to previous choice = left and success OR previous choice = right and failure
wsls = previous_reward * remapped_previous_choice
print(wsls)

# Now can create our design matrix and fill it with the three vectors: stimuli, previous choice and wsls.
# Remove stimuli of invalid choices
signed_contrast =  signed_contrast[valid_choices_idx]
# Create matrix to be filled with the predictors
n_trials = len(signed_contrast)
# Create empty design matrix of shape (n_trials, n_predictors)
design_mat = np.zeros((n_trials, 3))
# Add signed_contrast in the first predictor in the design matrix
design_mat[:, 0] = signed_contrast
# remap previous choice vals to {-1, 1} to match 1 -> rightward evidence and -1 -> leftward evidence
design_mat[:, 1] = previous_choice #2 * previous_choice - 1
# Add wsls as third predictor in the design matrix
design_mat[:, 2] = wsls

# We then normalize stimuli values across trials.
unnormalized_inpt2 = design_mat.copy()
normalized_inpt = np.copy(unnormalized_inpt2)
normalized_inpt[:, 0] = preprocessing.scale(normalized_inpt[:, 0])


def get_data_this_session(eid, df_trials):
    df_sess = df_trials[df_trials["session"] == eid]
    stim_left = df_sess['contrastLeft'].reset_index(drop=True)
    stim_right = df_sess['contrastRight'].reset_index(drop=True)
    rewarded = df_sess['feedbackType'].reset_index(drop=True)
    choice = df_sess['choice'].reset_index(drop=True)
    return choice, stim_left, stim_right, rewarded

def get_valid_choice(choice, viol_val):
    # takes in remapped choices
    # Get rid of violation trials i.e trials where the mouse didn't make a choice
    # previous choice vector getting rid of violation trials
    valid_choices_idx = np.where(~choice.isin([viol_val]))[0]
    valid_choices = choice[valid_choices_idx]
    return valid_choices, valid_choices_idx
    

def create_previous_choice_vector(valid_choices):
    # takes in already remapped valid choices and .
    
    # Shift the array elements one position to the right
    previous_choice = np.roll(valid_choices, 1)
    # Set first choice as first previous_choice
    previous_choice[0] = valid_choices[0]
    return previous_choice
        
def create_stim_vector(stim_left, stim_right):
    # Create stim vector
    stim_left = np.nan_to_num(stim_left, nan=0)
    stim_right = np.nan_to_num(stim_right, nan=0)
    # now get 1D stim
    #signed_contrast = stim_right - stim_left
    signed_contrast = stim_left - stim_right
    return signed_contrast

def create_wsls_covariate(previous_choice, rewarded, valid_choices_idx):
    '''
    inputs:
    previous_choice: vector of size T, entries are in {0, 1} and 0
    corresponds to left choice, 1 corresponds to right choice
    
    rewarded: vector of size T, entries are in {-1, 1} and -1 corresponds to
    failure, 1 corresponds to success
    valid_choices_idx : idx of valid trials
    
    wsls: vector of size T, entries are in {-1, 1}.  1 corresponds to
    previous choice = right and success OR previous choice = left and
    failure; -1 corresponds to previous choice = left and success OR previous choice = right and failure
    '''
    # remap previous choice vals to {-1, 1}
    remapped_previous_choice = previous_choice #2 * previous_choice - 1
    # Keep only rewards corresponding to valid trials
    valid_rewards = rewarded[valid_choices_idx]

    # Shift the array elements one position to the right
    previous_reward = np.roll(valid_rewards, 1)
    # Set first choice as first previous_choice
    previous_reward[0] = valid_rewards[0]
    # wsls: vector of size T, entries are in {-1, 1}.  1 corresponds to previous choice = right and success OR previous choice = left and failure; -1 corresponds to previous choice = left and success OR previous choice = right and failure
    wsls = previous_reward * remapped_previous_choice
    assert len(np.unique(wsls)) == 2, "wsls should be in {-1, 1}"
    return wsls

def remap_choice_vals(choice):
    # raw choice vector has CW = 1 (correct response for stim on left),
    # CCW = -1 (correct response for stim on right) and viol = 0.  Let's
    # remap so that CW = 0, CCw = 1, and viol = -1
    new_choice = choice.replace({1: 0, -1: 1, 0: -1})
    return new_choice

def create_design_matrix(choice, stim_left, stim_right, rewarded, viol_val):
    # Stimuli predictor before filtering for valid trials
    signed_contrast = create_stim_vector(stim_left, stim_right)

    # make choice vector so that correct response for stim>0 is choice =1
    # and is 0 for stim <0 (viol is mapped to -1)
    #choice = remap_choice_vals(choice)
    valid_choices, valid_choices_idx = get_valid_choice(choice, viol_val)
    previous_choice = create_previous_choice_vector(valid_choices)

    # create wsls vector:
    wsls = create_wsls_covariate(previous_choice, rewarded, valid_choices_idx)
    
    # filter for valid trials
    signed_contrast =  signed_contrast[valid_choices_idx]
    
    n_trials = len(signed_contrast)
    design_mat = np.zeros((n_trials, 3))
    design_mat[:, 0] = signed_contrast    

    # map previous choice to {-1,1}
    design_mat[:, 1] = previous_choice
    design_mat[:, 2] = wsls
    return design_mat

def get_all_unnormalized_data_this_session(eid, df_trials, viol_val):
    choice, stim_left, stim_right, rewarded = get_data_this_session(eid, df_trials)
    unnormalized_design_matrix = create_design_matrix(choice, stim_left, stim_right, rewarded, viol_val)
    #y = np.expand_dims(remap_choice_vals(choice), axis=1)
    valid_choices, _ = get_valid_choice(choice, viol_val)
    # remap choices to 1 and 0 to match bernoulli trials
    valid_choices = valid_choices.replace({1: 1, -1: 0})
    y = np.expand_dims(valid_choices, axis=1)
    session = [eid for i in range(y.shape[0])]
    rewarded = np.expand_dims(rewarded, axis=1)
    return unnormalized_design_matrix, y, session, rewarded

def get_unnormalized_design_mat(valid_sessions, df_trials, viol_val):
    sess_counter = 0
    for eid in valid_sessions:
        unnormalized_inpt, y, session, rewarded = \
            get_all_unnormalized_data_this_session(
                eid, df_trials, viol_val)
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
    
    return animal_unnormalized_inpt, animal_normalized_inpt, animal_y.flatten(), animal_session

unnormalized_inpt2, design_matrix, choices_mouse, session = get_unnormalized_design_mat(valid_sessions, df_trials, viol_val)