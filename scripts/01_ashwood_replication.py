# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: glm_hmm_notebook (3.12.10)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inferring behavioral strategies during decision making using GLM-HMMs
# One can think of decision-making as a stable process: given the same stimulus, an animal could be assumed to respond according to a fixed strategy with some added noise. However, growing evidence suggests that behavior is not stationary. Instead, animals fluctuate between distinct internal states that can persist over many trials. Traditional models, such as the classic lapse model, capture errors as random, independent events, but fail to account for these structured, state-dependent fluctuations in behavior. This raises the question: how can we infer these latent behavioral strategies directly from observed choices?
#
# In this notebook, we address this question using the GLM-HMM framework, which combines a Generalized Linear Model (GLM) with a Hidden Markov Model (HMM) to capture both how decisions change as a function of stimuli and how strategies evolve over time. We will show how to use choice data to recover hidden behavioral states using the NeMoS implementation of a Bernoulli GLM-HMM, replicating the main findings of Ashwood et al. (2022)<span id="cite1a"></span><a href="#ref1a">[1a]</a>.
#
# We have four main goals for this tutorial:
#
# 1. Explain how to download and preprocess real mice data from the [International Brain Laboratory (IBL)](https://www.internationalbrainlab.com)
# 2. Show how to create a design matrix with different behavioral predictors
# 3. Show how to fit choice data using a GLM-HMM
# 4. Show how to interpret GLM-HMM fitting results
# 5. Provide some ideas for follow-up analyses
#
# Importantly, throughout the notebook we will assume you already have a solid theoretical understanding of GLMs and GLM-HMMs. If you need an explanation, please refer to our tutorials on GLMs and GLM-HMMs. Moreover, if you already have a good understanding of GLM-HMMs and are interested in different heuristics you could use to overcome difficulties in the fitting process, please refer to our tutorial for fine-grain details of the fitting algorithm and different initialization methods you could use to ensure the best possible fit and thus description of your data under this model.

# %% [markdown]
# ## GLM-HMMs
# GLM-HMMs, also known as input-out HMM models (Bengio & Frasconi, 1995) <span id="cite2"></span><a href="#ref2">[2]</a>,  are useful to analyze how hidden latent states affect observable behavioral (Ashwood et al., 2022) <span id="cite1b"></span><a href="#ref1b">[1b]</a> and neural (Escola et al., 2011)<span id="cite3a"></span><a href="#ref3a">[3a]</a> dynamics. These models are composed by an HMM, governing the distribution over the latent states, and state-specific GLMs, which specify the activity of the system at each state.
#
# <center><img src="images/graphical_model.png" alt="Graphical model of GLM-HMM" width="800" /></center>

# %% [markdown]
# In all GLM-HMMs, the HMM component is fully defined by three elements: a state transition matrix, an initial probability vector and an emissions probability distribution (Bishop, 2006)<span id="cite4"></span><a href="#ref4">[4]</a>. A HMM with K hidden states has a $K \times K$ transition matrix that specifies the probability of transitioning from any state to any other,
# $$
# p(z_t=j\mid z_{t-1} = i) = A_{ij}
# $$
# where $z_{t-1}$ and $z_t$ indicate the latent state at trials $t-1$ and $t$, respectively. The HMM also has a distribution over the initial states, given by a $K$-element vector $\pi$ whose elements sum to one:
# $$
# p(z_1 = i) = \boldsymbol{\pi}_i
# $$
# Finally, the emissions probability describes the relationship between the state and the observation. In the case of GLM-HMMs, the emissions probability is a GLM (it can be, for example, a Bernoulli GLM); that is, a generalization of linear regression that allows to characterize how an output (behavior, neuronal activity) may vary as a function of an input.
#
# A $K$-state GLM-HMM contains $K$ independent GLMs, each defined by a weight vector specifying how inputs are integrated in that particular state to give rise to activity. These describe the state-dependent mapping from inputs to activity. In this tutorial, we will use a GLM-HMM with a Bernoulli observation model i.e., a Bernoulli GLM.
# $$
# y_t \mid \boldsymbol{x}, \boldsymbol{k} \sim Ber(f(-\boldsymbol{x}_t \cdot \boldsymbol{w}_k)) \\
# $$
# where $\boldsymbol{w}_k \in \mathbb{R}^M$ denotes the GLM weights for latent state $k \in {1,..,K}$. Thus, the probability of success ($y_t = 1$, which can correspond to a given choice in a binary set up, or a spike count for a time bin; in our case, corresponds to a leftward choice) given the input vector $\boldsymbol{x}_t$ is given by:
# $$
# p(y_t=1\mid\boldsymbol{x}_t, z_t = k)  = \frac{1}{1+exp(-\boldsymbol{x}_t \cdot \boldsymbol{w}_k)}
# $$
# considering a Bernoulli GLM with a logistic inverse link function.

# %% [markdown]
# ## 00. Imports

# %%
# Imports that will go away
import nemos as nmo
from nemos.glm_hmm import GLMHMM
nmo.GLMHMM = GLMHMM # this is the only way I got the GLM HMM module to work when using my own installation...I don't really know why but it won't be a problem when we release anyway

# %%
# Imports
import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import seaborn as sns
from one.api import ONE
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy.special import expit
from nemos.glm_hmm.utils import compute_rate_per_state

# %% tags=["hide-input"]
seed = 65  # Random seed for reproducibility
np.random.seed(seed)
jax.config.update("jax_enable_x64", True)

# Parameters for plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)#, context="notebook")

# %% [markdown]
# ## 01. Downloading the data and understanding the task structure: IBL decision-making task
#
# <center><img src="images/IBL_edited.png" alt="Task schematic" width="500" /></center>
#
# <div style="text-align: center;">
#
#   Task illustration. Modified from IBL et al. (2021) 
#    <span id="cite5a"></span><a href="#ref5a">[5a]</a>.
# </div>
#
# We will analyze the IBL decision-making task  (IBL et al., 2021 <span id="cite5b"></span><a href="#ref5b">[5b]</a>), which is a variation of the two-alternative forced-choice perceptual detection task (Burgess et al. (2021) <span id="cite6"></span><a href="#ref6">[6]</a>. During this task, a sinusoidal grating with varying contrast [0\%-100\%] appeared either at the right or left side of the screen. The goal for the mice was to indicate this side turning a little wheel so that this turn would accordingly move the stimuli to the center of the screen (Burgess et al. (2021) <span id="cite6"></span><a href="#ref6">[6]</a>. If the mice chose the side correctly, they would receive a water reward; if not, they would get a noise burst and there would be a 1 second timeout. For the first 90 trials of each session in the task, the stimulus appeared randomly on either side of the screen; after that, the stimulus would appear on one side with fixed probability 0.8 and alternate randomly every 20-100 trials. 
#
# First, let's download the data using [Open Neurophysiology Environment (ONE)](https://docs.internationalbrainlab.org/notebooks_external/one_quickstart.html)

# %%
# Using ONE's ```load_aggregate``` function, we can retrieve all sessions from a given animal. For this, first we need to instantiate the ```ONE``` object
one = ONE(password = 'international')

# Then we need to choose our subject and run ```load_aggregate```
subject = "CSHL_008"
trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')

# We can see the information we get by printing the columns
print(trials.columns)

# %% [markdown]
# :::{admonition} Should I use ```one.search()``` or ```load_aggregate```?
# :class: info
# ```one.search()``` returns session IDs (eids) that exist as session records in Alyx, while load_aggregate() downloads a pre computed file with trial data pooled across multiple sessions. If you want to get all sessions from a single animal, it is recommended to use ```load_aggregate```, because some sessions may be located in a dataframe without a session identified in itself (but containing multiple sessions with their own session identifiers). 
# :::

# %% [markdown]
# We are modeling choice as result of observables and behavioral state. Thus, we need choice, stimuli presented and reward obtained. Additionally, we want to keep the session identifier to know when sessions start and end and for plotting. Furthermore, in this task, the probability of the stimulus being in the left or the right side of the screen changes over time within a session. Thus, we also want the information of the probability of the stimulus appearing in a given position.
#
# | Variable            | Description |
# |---------------------|-------------|
# | choice              | mouse choice: 1 = choice left, -1 = choice right, 0 = violation (no response within the trial period). Since we are going to use a Bernoulli GLM, we will remap the variables to 1 = choice left and 0 = choice right at the end of preprocessing. |
# | contrastLeft        | contrast of stimulus presented on the left |
# | contrastRight       | contrast of stimulus presented on the right |
# | feedbackType        | reward obtained: 1 = success, -1 = failure |
# | probabilityLeft     | probability of stimulus being presented on the left of the screen |
# | session             | id of session |
#
# Let's extract the meaningful data and see how it looks
# %%
trials = trials[["choice", "contrastLeft", "contrastRight", "feedbackType", "probabilityLeft", "session"]]

print(f"choice \nvalues: {trials.choice.unique()}, data type: {trials.choice.dtype}, shape:  \n")
print(f"contrast left \nvalues: {trials.contrastLeft.unique()}, data type: {trials.contrastLeft.dtype} \n")

print(f"contrast right \nvalues: {trials.contrastRight.unique()}, data type: {trials.contrastRight.dtype} \n")

print(f"reward \nvalues: {trials.feedbackType.unique()}, data type: {trials.feedbackType.dtype} \n")

print(f"probability of stimulus on left \nvalues: {trials.probabilityLeft.unique()}, data type: {trials.probabilityLeft.dtype} \n")

print(f"session \n(some) values: {trials.session.unique()[:5]}, data type: {trials.session.dtype}\n")
# %% [markdown]
# Now, we will restrict the analysis to the first 90 trials of each session. In this segment, the stimulus appears on the left and right with equal probability (0.5/0.5). In this regime, choices are driven primarily by sensory evidence rather than learned expectations about stimulus probability. After trial 90, the task switches to a block structure in which the left stimulus occurs with probability 0.8 or 0.2, alternating across blocks within the session.

# %%
# Choose example session
sess_ex = '726b6915-e7de-4b55-a38e-ff4c461211d3'
# Subset session trials
trials_sess = trials[trials.session == sess_ex].reset_index()

# Plot
plt.plot(trials_sess["probabilityLeft"][:300])
plt.axvspan(0, 90, color="skyblue", alpha=0.3, label="first 90 trials")
plt.axvline(90, color="skyblue", linestyle="--")

plt.ylabel("P(stimulus on the left)")
plt.xlabel("Trial number")
plt.show()

# %% [markdown]
# In  Ashwood et al. (2022)<span id="cite1c"></span><a href="#ref1c">[1c]</a>, only the sessions with less than 10 violations were used. To follow this work, we will now revise the number of violations, defined as trials where the animal made no choice. i.e choice == 0 during the 50-50 trials. For this, we will:
#  1) Subset sessions which include 50-50 trials
#  2) Exclude sessions with >10 violation trials

# %%
# Create a list of ids
sessions_ids = trials.session.unique()

# keep only relevant columns for filtering
df_trials = trials[["session", "probabilityLeft", "choice"]]

# Get which sessions contain exactly {0.2, 0.5, 0.8}
valid_prob_sessions = (
    df_trials.groupby("session")["probabilityLeft"]
      .agg(lambda x: set(x.unique()) == {0.2, 0.5, 0.8})
)

# Compute violations only on 50-50 trials
viol_val = 0
violations = (
    df_trials[df_trials["probabilityLeft"] == 0.5]
    .groupby("session")["choice"]
    .apply(lambda x: (x == viol_val).sum())
)

# Apply both restrictions
valid_sessions = violations[
    (violations < 10) & (violations.index.isin(valid_prob_sessions[valid_prob_sessions == True].index))
].index.tolist()

# Make sure they maintain the order of the original dataset (we don't want scrambled trials)
valid_set = set(valid_sessions)
valid_sessions = [
    s for s in trials["session"].drop_duplicates()
    if s in valid_set
]
print(f"# of sessions before restrictions {len(df_trials.session.unique())}")

# Now we can select only the valid sessions for subsequent analyses
df_trials = trials[
    (trials["session"].isin(valid_sessions)) & (df_trials["probabilityLeft"] == 0.5)
]
print(f"# of sessions after restrictions {len(df_trials.session.unique())}")

# %% [markdown]
# ## 02. Building the design matrix
# Now, with the valid sessions, we can compute the design matrix. In our case, we are interested in building a design matrix with three predictors: signed contrast, previous choice and win stay lose shift.
#
# <center><img src="images/design_matrix_table.png" alt="Task schematic" width="1100" /></center>

# %% [markdown]
# The first predictor, signed contrast, encodes sensory evidence in 1D. Within this predictor, magnitude reflects strength of evidence and sign encodes direction. The second prediction, previous choice, is a lagged version of current choice, and it reflects serial dependence on decisions. Finally, the third predictor is win-stay lose-shift. This reflects the interaction between past choice and outcome. If an animal made a decision and it was rewarded in a previous trial, then the predictor indicates to "stay". That is, to repeat that choice. Conversely, if the previous choice was not rewarded, then the predictor indicates to "switch" to the other alternative.
#
# Let's go through the process of building the design matrix with one session.

# %%
# Select an example session
eid = valid_sessions[0]  
df_sess = df_trials[df_trials["session"] == eid]

# We can select all the necessary values for the design matrix: choice, contrast of stimuli and reward
choices = df_sess['choice'].reset_index(drop=True)
stim_left = df_sess['contrastLeft'].reset_index(drop=True)
stim_right = df_sess['contrastRight'].reset_index(drop=True)
rewarded = df_sess['feedbackType'].reset_index(drop=True)

# %% [markdown]
# For the first predictor: signed contrast.

# %%
# Create stim vector
stim_left = np.nan_to_num(stim_left, nan=0)
stim_right = np.nan_to_num(stim_right, nan=0)

# now get 1D stim
signed_contrast = stim_left - stim_right
print(signed_contrast)

# %%
# Get rid of violation trials i.e trials where the mouse didn't make a choice
valid_choices_idx = np.where(~choices.isin([viol_val]))[0]

# %% [markdown]
# With those two elements and using ```nmo.basis```, it is very easy to compute our design matrix for this session.
#
# A basis is a collection of functions that, when combined, can represent more complex relationships. NeMoS has a lot of different basis functions, but here we are interested in using two: ```HistoryConv``` and ```IdentityEval```.
#
# - ```HistoryConv``` includes the history of the samples as predictor. It is intended to be used for including a raw history as predictor. We can use this to create the previous choice predictor.
#
# - ```IdentityEval``` includes the samples themselves as predictors. It is intended to be used for including a task variable directly as a predictor. We can use this for the stimuli predictor. 
#
# It is very easy to declare our basis objects:

# %%
prev_choice_basis = nmo.basis.HistoryConv(1)
stimuli_basis = nmo.basis.IdentityEval()

# %% [markdown]
# However, we are still missing one predictor: win-stay lose shift. This is a combination of previous choice and previous reward. To have interaction between predictors, we can use composite basis. In particular, we can apply two types of operations: addition and multiplication. Addition simply concatenates the basis objects. However, when multiplying two basis objects, one constructs interaction features by taking the row-wise outer product of the feature vectors produced by each basis after evaluation.

# %%
# Create lagged reward predictor
prev_reward_basis = nmo.basis.HistoryConv(1)

# Create multiplicative basis object for interaction between previous choice and reward
wsls_basis = prev_choice_basis*prev_reward_basis

# %% [markdown]
# Now we have all our bases. We can create a composite basis including all of them and then all we need to do now is to apply the basis transformation to the input data. We can do this by using ```compute_features```. This method is designed to be a high-level interface for transforming input data using the basis functions. 
#
# Even though we need just a few lines of code, there is a lot going on. Here's a breakdown of what will happening:
# 1. We will create an additive basis ```basis_object``` with our bases ```stimuli_basis```, ```wsls_basis``` and ```prev_choice_basis```. 
# 2. ```wsls_basis``` is a multiplicative basis that takes two inputs.
# 3. We will compute the features for our ```basis_object``` using ```compute_features```. Since the bases in our composite basis take a total of 4 inputs (```stimuli_basis``` takes 1 input, ```wsls_basis``` takes 2 inputs and ```prev_choice_basis``` takes 1 input), we need to pass 4 features to ```compute_features```.

# %%
# Create a composite basis using our three basis
basis_object = (
    stimuli_basis +                         # will process one input
    wsls_basis +                            # will process two inputs (choice & reward)
    prev_choice_basis                       # will process one input
)

# Compute features
X_unnormalized = basis_object.compute_features(
    signed_contrast[valid_choices_idx],     # input 1 : processed with stimuli_basis
    choices[valid_choices_idx],             # input 2 : wsls input 1: choice
    rewarded[valid_choices_idx],            # input 3 : wsls input 2: reward
    choices[valid_choices_idx]              # input 4 : processed with prev_choice
)        

print(X_unnormalized)     

# %% [markdown]
# And that's it! We have our unnormalized design matrix with signed contrast, win-stay lose-shift and previous choice as predictors.
#
# As as last step, we now need to normalize our signed contrast predictor.

# %%
# Normalize across the signed contrast
X = np.copy(X_unnormalized)
X[:, 0] = preprocessing.scale(X[:, 0])

# %% [markdown]
# :::{admonition} Why do we normalize our stimuli predictor? 
# :class: info
# :class: dropdown
#
# When fitting a GLM-HMM, we are fitting a separate weight for each feature. However, if the features are on different numerical scales for reasons that are not related to the actual influence of each predictor, that renders the weights incomparable. Here we have three predictors:
# - (1) Previous choice and (2) WSLS are always exactly −1 or +1. Their values are discrete and bounded, and they already share the same scale.
# - (3) Stimuli contrast is continuous. While it can reach −1 or +1 (full contrast), this value rarely occurs. 
#
# Because the stimuli contrast values are much smaller in typical magnitude than +/-1, the model compensates by assigning a smaller weight to match the output scale, simply because its values are numerically smaller. In practice, this results inan artifact of scale that is not reflective of the  true influence of the predictor.
#
# By normalizing, we are rescaling the predictor to have mean 0 and standard deviation of 1. Previous choice and WSLS already on a unit scale by construction — their values are symmetric around zero and their spread is naturally 1. This is why we only normalize signed contrast.
# :::

# %% [markdown]
# and see our design matrix.

# %%
fig, axes = plt.subplots(1, 2, figsize=(5, 8), sharey=True)

# ---- define signed contrast bins 
signed_levels = np.array([
    -2.424443102018604,
    -1.0,
    -0.6694842882249763,
    -0.3769911525927051,
    -0.2307445847765695,
    -0.0844980169604339,
     0.06174855085570171,
     0.20799511867183731,
     0.5004882543041086,
     1.0,
     2.255447068097736,
])

all_levels = np.unique(signed_levels)
all_levels.sort()

cmap_cat = LinearSegmentedColormap.from_list(
    "bias_map",
    ["#377eb8", "white", "#4daf4a"]  # left → neutral → right
)

bounds = np.concatenate([
    all_levels - 1e-6,
    [all_levels[-1] + 1]
])

norm = BoundaryNorm(bounds, cmap_cat.N)

# ---- heatmap 1: full design matrix ----
sns.heatmap(
    X[:20,:],
    ax=axes[0],
    square=True,
    cmap=cmap_cat,
    norm=norm,
    cbar=False
)

axes[0].set_xticks([0.5, 1.5, 2.5],
                   ["Sign. contr.", "WSLS", "Prev. choice",], 
                   rotation=90)
axes[0].set_yticks([])
#axes[0].set_xlabel("Predictors")
axes[0].set_ylabel("Trials")
axes[0].set_title("Design matrix")

# ---- heatmap 2: choices ----
sns.heatmap(
    choices[valid_choices_idx].to_numpy().reshape(-1, 1)[:20],
    ax=axes[1],
    square=True,
    cmap=cmap_cat,
    norm=norm,
    cbar=True
)
axes[0].set_yticks([])
axes[1].set_xticks([0.5], 
                   ["Choices"], 
                   rotation=90)

cbar = axes[1].collections[0].colorbar

midpoints = (bounds[:-1] + bounds[1:]) / 2
cbar.ax.minorticks_off()
cbar.set_ticks(midpoints)
cbar.set_ticklabels([f"{v:.4g}" for v in signed_levels])

plt.tight_layout()
plt.show()

# %% [markdown]
# :::{admonition} Why isn't the normalized stimuli vector centered at 0, even if the raw one is?
# :class: info
# :class: dropdown
# The raw stimulus contrast was defined symmetrically (e.g. spanning [−1,+1]). However, in practice, the empirical distribution is rarely perfectly balanced, so its mean is not exactly 0. Centering subtracts this empirical mean, which forces the transformed variable to have mean 0 in the dataset.
#
# Since our data was previously binned, this transformation changes the coordinate system: the original bin locations are shifted in the normalized space, so the “0 bin” may no longer align with a value exactly equal to 0. In other words, centering preserves relative structure but not the original categorical bin positions.
# :::

# %% [markdown]
# Similarly, we can do this process for all the sessions

# %%
# Subset all relevant data
stim_left = df_trials['contrastLeft'].reset_index(drop=True)
stim_right = df_trials['contrastRight'].reset_index(drop=True)
rewarded = df_trials['feedbackType'].reset_index(drop=True)
choices = df_trials['choice'].reset_index(drop=True)
session = df_trials['session'].reset_index(drop=True).to_numpy()

# Select valid choices
valid_choices_idx = np.where(~choices.isin([viol_val]))[0]

# Create stim vector
stim_left = np.nan_to_num(stim_left, nan=0)
stim_right = np.nan_to_num(stim_right, nan=0)

# now get 1D stim
signed_contrast = stim_left - stim_right
print(signed_contrast)

# Create basis objects
prev_choice_basis = nmo.basis.HistoryConv(1)
prev_reward_basis = nmo.basis.HistoryConv(1)
stimuli_basis = nmo.basis.IdentityEval()

wsls = prev_choice_basis*prev_reward_basis

# Create a composite basis using our three basis
basis_object = (
    stimuli_basis +                         # will process one input
    wsls_basis +                            # will process two inputs (choice & reward)
    prev_choice_basis                       # will process one input
)

# Compute features
X_unnormalized = basis_object.compute_features(
    signed_contrast[valid_choices_idx],     # input 1 : processed with stimuli_basis
    choices[valid_choices_idx],             # input 2 : wsls input 1: choice
    rewarded[valid_choices_idx],            # input 3 : wsls input 2: reward
    choices[valid_choices_idx]              # input 4 : processed with prev_choice
)   

# And then normalize across the signed contrast
X = np.copy(X_unnormalized)
X[:, 0] = preprocessing.scale(X[:, 0])

# For fitting a Bernoulli, our variables need to be in 0-1 space. So we will remap them so 1: Left and 0: Right
choices.replace({1: 1, -1: 0}, inplace = True)

# %% [markdown]
# Importantly, do not do 3000 trials at once! Instead, they generally do several sessions of 100-300 trials, and we use all the sessions together to fit our model. For our model to be accurate, we need to tell it when our session boundaries are: we don't want it to compute all sessions as if they were one. 
#
# With NeMoS, we have two ways of indicating the beginning of a new session. When using a design matrix and a choice vector that are Numpy objects, NeMoS treats NaNs either in the input matrix X or in the choice vector y as indicators of new sessions. You can also use a Pynapple Tsd or TsdFrame to demarcate sessions, which is what we will use now.

# %%
# Mark where session changes
new_sess_mouse = np.ones(len(session), dtype=int)
new_sess_mouse[1:] = (session[1:] != session[:-1])

# Get positions of start of sessions
new_sess_pos = np.array(np.where(new_sess_mouse==1))
# Get length of session
sess_length = np.array(np.where(new_sess_mouse==1))[0][1] - 1 # substracting the starting trial


# %% [markdown]
# We will use a ```nap.Tsd``` now, because the choices are a 1-dimensional time series. To initialize it, we need the length of the series ```t``` and the data of the time series ```d``` (our choices vector). In addition to this, we can we can initialize our ```nap.Tsd``` with a ```time_support``` property, and this way store both the choices and the sessions information in the same object.

# %% [markdown]
# admonition When dealing with 2-dimensional time series, you should use the ```nap.TsdFrame``` object. If you want to learn more about Pynapple objects, check out [Pynapple documentation](https://pynapple.org/user_guide/01_introduction_to_pynapple.html).

# %%
choices_tsd = nap.Tsd(t = np.arange(choices.shape[0]), d=choices, time_support = nap.IntervalSet(
    start=new_sess_pos, 
    end = new_sess_pos+ sess_length))

print(f"choices tsd \n {choices_tsd} \n")
print(f"time support\n {choices_tsd.time_support}")

# %% [markdown]
# ## 03. Fitting a GLM-HMM with NeMoS (Maximum Likelihood)
# We will use a Bernoulli GLM to model this mouse's choices. For this, we first need to initialize the ```GLMHMM``` object. The only required parameter is the number of states. Ashwood et al. (2022) <span id="cite1d"></span><a href="#ref1d">[1d]</a> found that most mice used 3 decision-making states when performing the IBL decision-making task  (IBL et al., 2021 <span id="cite5c"></span><a href="#ref5c">[5c]</a>). Following that work, we will initialize our ```GLMHMM``` object with 3 states. 

# %% [markdown]
# :::{admonition} GLM-HMM observation models
# :class: info
# The default observation model for the GLM-HMM is this model is Bernoulli, but Categorical (Multinomial), Poisson, Gamma, Negative Binomial and Gaussian observation models are also available. If you want, you can also set a different observation model of your choice and personalize the inverse link function. However, bare in mind that convexity is non guaranteed for all likelihood functions. For more information, refer to Escola et al (2011)<span id="cite3b"></span><a href="#ref3b">[3b]</a> and also to [our notebook on GLM-HMM theoretical underpinnings - PENDING]()
# :::

# %% [markdown]
# [PENDING - do not edit because it might change]()
# If required, you can further personalize the ```GLMHMM``` object settings. Beyond the number of states, the observation model and the inverse link function, you can also initialization functions for to aid parameter estimation. 
#
# If you don't set up any initialization settings, you would use the NeMoS defaults:
# - ``"glm_params_init"``: ``"random"`` - small random coefficients, mean-rate intercept
# - ``"scale_init"``: ``"constant"`` - scale initialized to 1.0
# - ``"initial_proba_init"``: ``"uniform"`` - equal probability for all states
# - ``"transition_proba_init"``: ``"sticky"`` - high self-transition probability (0.95)
#

# %%
n_states = 3

model = nmo.glm_hmm.GLMHMM(
    n_states,
    regularizer = "Ridge")

print(model)

# %% [markdown]
# :::{admonition} Importance of initial parameters in GLM-HMMs
# :class: warning
# When fitting a GLM-HMMs, the likelihood surface is non-convex, and EM-based fitting can converge to different local optima depending on starting values. As a result, different initializations can lead to qualitatively different parameters. In practice, this makes it necessary to either run multiple random restarts or use informed initializations derived from simpler models (e.g. logistic regression or clustering of behavior).
#
# For a more detailed example of how initialization affects convergence and interpretation, refer to [our notebook on GLM-HMM theoretical underpinnings - PENDING]()
# :::

# %% [markdown]
# Once we created our object, we can fit our model. The fit function takes two mandatory arguments: the design matrix ```X```we created in section 02 and the ```choices_tsd```.

# %%
model.fit(X, choices_tsd)

# %% [markdown]
# Thats all it takes!

# %% [markdown]
# ## 04. Interpreting the fitting results

# %% [markdown]
# ### 04.0 How to see the fitted parameters

# %% tags=["hide-input"]
permutation = jnp.array([1, 2, 0])
model.coef_ = model.coef_[:, permutation]
model.intercept_ = model.intercept_[permutation]
model.transition_prob_ = model.transition_prob_[permutation][:, permutation]

# %% [markdown]
# If we want to see our glm-hmm weights, we can call ```model.coef_```. This will output the coefficients of the glm per state, with shape (n_features, n_states).

# %%
print(f"glm weights shape \n {model.coef_.shape} \n")
print(f"glm weights \n {model.coef_}")

# %% [markdown]
# Similarly, to see the intercept, we can call ```model.intercept_```, which will output the intercept per state. The shape of this object is (n_states)

# %%
print(f"intercept shape \n {model.intercept_.shape} \n")
print(f"intercept \n {model.intercept_}")

# %% [markdown]
# We can also see the estimated transition matrix with ```model.transition_prob``` and the initial probatilities with ```model.initial_prob```, with shapes (n_states, n_states) and (n_states,), respectively.

# %%
print(f"transition matrix shape \n {model.transition_prob_.shape}")
print(f"transition matrix \n {model.transition_prob_}")
print(f"initial probabilities shape \n {model.initial_prob_.shape}")
print(f"initial probabilities \n {model.initial_prob_}")


# %% [markdown]
# Let's see what type of information we can gather.

# %% [markdown]
# ### 04.1 Interpreting the GLM weights
# We can plot the GLM weights obtained for our 3-state model. 

# %% tags=["hide-input"]
def plot_glm_weights(model, n_states = n_states):
    plt.figure(figsize=(6, 5))
    colors = ["#ff7f00",  "#4daf4a","#377eb8"]

    n_features = model.coef_.shape[0]+1 # add 1 for the intercept

    # Change order of weights so output matches Ashwood et al. (2022) 2e plot
    recovered_weights = np.zeros((n_features,n_states)) 
    recovered_weights[0,:] = model.coef_[0,:] # stimulus
    recovered_weights[1,:] = model.intercept_ # bias
    recovered_weights[2,:] = model.coef_[2,:] # prev choice, wsls
    recovered_weights[3,:] = model.coef_[1,:] # prev choice, wsls

    # Labels
    X_labels = ["Stimulus", "Bias", "Prev.choice", "WSLS"]

    for state in range(n_states):
        plt.plot(
            range(n_features),
            recovered_weights[:, state],
            color=colors[state],
            marker="o",
            lw=1.5,
            label="State {}".format(state + 1),
            linestyle="-",
        )
            
    plt.yticks([-2.5, 0, 2.5, 5])
    plt.ylabel("GLM weight")
    plt.xlabel("Covariate")
    plt.xticks([i for i in range(n_features)], X_labels, fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")

    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return None


# %%
plot_glm_weights(model)


# %% [markdown]
# We can see that the coefficients on state 1 have a large weight on the stimulus and low weight on the other predictors. Conversely, in states 2 and 3, the stimulus coefficient is comparatively lower. State 2 has a large positive weight on bias, while State 3 has a large negative weight on bias. Since the sign of our predictors indicates the side of evidence (>0 : left; <0 : right, see the table of variables in section 01) and their magnitude indicates the strength of such evidence, State 2 coefficients suggest a large bias towards leftward choice, while State 3 coefficients suggest a large bias to a rightward choice. All states have similarly low coefficients for prev. choice and wsls, with State 1 showing the smallest of them. 
#
# As a reminder, the task consisted on indicating whether the stimulus was located at the right or the left of the screen using the stimulus contrast information. Thus, the optimal strategy is to maximally use stimulus contrast to guide decision making, and not rely on bias, previous choice or wsls.

# %% [markdown]
# ### 04.2 Interpreting the transition matrix
# We can also see the fitted transition matrix for our three-state model. This describes the transition probabilities among the different states, each corresponding to a different decision-making strategy. Large entries in the diagonal indicate a high probability of remaining in the same state for multiple trials in a row.

# %% tags=["hide-input"]
def plot_transition_matrix(model, n_states= n_states):
    fig = plt.figure(figsize=(8, 3))
    n_decimals = 3
    # Plot matrix colors
    plt.imshow(model.transition_prob_, vmin=-0.8, vmax=1, cmap='bone')

    # Write probabilities
    for i in range(n_states):
        for j in range(n_states):
            text = plt.text(j, i, str(np.around(model.transition_prob_[i, j], decimals=n_decimals))[:n_decimals+2], ha="center", va="center",
                            color="k")
    plt.xlim(-0.5, n_states - 0.5)
    plt.xticks(range(0, n_states), ('1', '2', '3'))
    plt.xlabel("State t")

    plt.yticks(range(0, n_states), ('1', '2', '3'))
    plt.ylim(n_states - 0.5, -0.5)
    plt.ylabel("State t-1",)

    plt.title("Transition matrix")
    plt.subplots_adjust(0, 0, 1, 1)
    plt.show()
    return None


# %%
plot_transition_matrix(model)


# %% [markdown]
# ### 04.3 Interpreting how different GLM weight combinations produce different behaviors: building a psychometric curve using ```compute_rate_per_state```
#
# We can also see the decision-making strategies corresponding to each of the three states by plotting their psychometric curves. These show the probability of a choice as a function of the stimulus and conditioned on previous choice and reward. To compute the psychometric curve matching Ashwood et al., 2022 <span id="cite1e"></span><a href="#ref1e">[1e]</a>, we need to consider that there are four possible scenarios for previous choice and wsls predictors combinations:
# 1) Animal went left on the previous trial and was rewarded
#
# $$
# c_{t-1} = 1, r_{t-1} = 1 \rightarrow  \text{WSLS}_t = 1
# $$
#
# 2) Animal went left on the previous trial and was not rewarded
#
# $$
# c_{t-1} = 1, r_{t-1} = -1 \rightarrow  \text{WSLS}_t = -1
# $$
#
# 3) Animal went right on the previous trial and was not rewarded
#
# $$
# c_{t-1} = -1, r_{t-1} = -1 \rightarrow  \text{WSLS}_t = 1
# $$
#
# 4) Animal went right on the previous trial and was rewarded
#
# $$
# c_{t-1} = -1, r_{t-1} = 1 \rightarrow  \text{WSLS}_t = -1
# $$
#
# As such, we would have four psychometric curves (each corresponding to scenario 1, 2 3 or 4) per state.

# %% [markdown]
# We can compute the probability of leftward choice here by using ```compute_rate_per_state```. This method evaluates the GLM's predicted mean response (rate) for each latent state by computing the linear combination of features and applying the inverse link function. It requires a matrix of predictors X, GLM coefficients and intercepts and an inverse link function. It returns the predicted rate per each state, of shape (n_time_bins, n_states).

# %%
# This will go away when we get a public method for predict_rate_per_state
class PARAMS:
    def __init__(self, coef, intercept):
        self.coef = coef
        self.intercept = intercept
param = PARAMS(model.coef_, model.intercept_)

# %%
# Get max and min stimulus contrast 
min_val_stim = np.min(X[:, 0])
max_val_stim = np.max(X[:, 0])
# and create an array with constant steps between them for plotting
stim_vals = np.arange(max_val_stim, min_val_stim, -0.05)

# Cases 1, 2, 3 and 4
cases = [[1,1], [1,-1], [-1,1], [-1,-1]] 
rates = []  # List of rates

# Instantiate basis objects
prev_choice_const_basis, wsls_const_basis = nmo.basis.IdentityEval(), nmo.basis.IdentityEval()
basis_object = (stimuli_basis + wsls_const_basis + prev_choice_const_basis)

# Iterate through the four cases
for case in cases:
    pc_const, wsls_const = case
    
    # Compute design matrix for this case
    design_matrix_psychometric = basis_object.compute_features(
        stim_vals,   
        np.repeat(pc_const, len(stim_vals)),   
        np.repeat(wsls_const, len(stim_vals))
    )

    # Compute rate per state
    rate_per_state_case = compute_rate_per_state(
        design_matrix_psychometric, 
        param, 
        model.inverse_link_function
    )
    
    # Append to rates list
    rates.append(rate_per_state_case)
    
print(f"first ten trials of case 1 \n  state 1    state 2    state 3 \n{rates[0][:10,:]} \n")
print(f"shape of compute_rate_per_state output \n{rates[0].shape}")


# %% tags=["hide-input"]
def plot_psychometric(n_states = n_states, rates = rates):
    fig = plt.figure(figsize=(8, 4))
    cols = ["#ff7f00", "#4daf4a", "#377eb8"]

    for state in range(n_states):
        plt.subplot(1, 3, state+1)
        # Case 1: went L and was rewarded on previous trial
        plt.plot(stim_vals, 
                1-rates[0][:,state], 
                '--', 
                color=cols[state], 
                alpha=1,
                lw=1,
                label="Case 1")
        # Case 2: went L and was not rewarded on previous trial
        plt.plot(stim_vals,
                1-rates[1][:,state],
                '-',
                color=cols[state],
                alpha=0.5,
                lw=1,
                markersize=3,
                label="Case 2")
        
        # Case 3: went R and was not rewarded on previous trial
        plt.plot(stim_vals,
                1-rates[2][:,state],
                '--',
                color=cols[state],
                alpha=0.5,
                lw=1,
                label="Case 3")
        # Case 4: went R and was rewarded on previous trial
        plt.plot(stim_vals,
                1 - rates[3][:,state],
                '-',
                color=cols[state],
                alpha=1,
                lw=1,
                zorder=5,
                label="Case 4")

        plt.xticks([min(stim_vals), 0, max(stim_vals)],
                    labels=['', '', ''],
                    )

        plt.yticks([0, 0.5, 1], ['', '', ''])
        plt.ylabel('')
        plt.xlabel('')
        if state == 0:
            plt.title("State 1 \n(\"engaged\")", color=cols[state])
            plt.xticks([max(stim_vals), 0, min(stim_vals)],
                        labels=['100', '0', '-100'],
                        )
            plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
            plt.ylabel('p("R")')
            plt.xlabel('stimulus')

        if state == 1:
            plt.title("State 2 \n(\"biased left\")",
                        color=cols[state])
            plt.xticks([max(stim_vals), 0, min(stim_vals)],
                        labels=['', '', ''],
                        )
            plt.yticks([0, 0.5, 1], ['', '', ''])
        if state == 2:
            plt.title("State 3 \n(\"biased right\")",
                        color=cols[state])
            plt.xticks([max(stim_vals), 0, min(stim_vals)],
                        labels=['', '', ''],
                        )
            plt.yticks([0, 0.5, 1], ['', '', ''])
            
        plt.axhline(y=0.5, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.axvline(x=0, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.ylim((-0.01, 1.01))
        # Invert axis because we want stimuli to go
        # From + to - to match Ashwood et al. (2022) 2g plot
        plt.gca().invert_xaxis()
        
    # Add legend table
    im = plt.imread('images/psychometric_curve_legend.png')
    newax = fig.add_axes([-.6, 0.25, 0.6, 0.6], anchor='NE')
    newax.imshow(im)
    newax.axis('off')
    
    plt.show()
    return None


# %%
plot_psychometric(n_states, rates)

# %% [markdown]
# According to state 1 psychometric curve, the larger the magnitude of contrast, the better the performance of the animal. Conversely, states 2 and 3 show bias to the left and to the right respectively, and they are also dependent on previous choice and reward, as indicated by the gap between solid and dashed lines.

# %% [markdown]
# ### 04.4 Using ```smooth_proba``` to see and interpret posterior state probabilities
# To better understand the temporal structure of decision making behavior, we can compute the probability of being in each state at each trial, conditioned on the entire observed sequence. For this, we can use ```smooth_proba```. This method uses the forward-backward algorithm to incorporate information from past and future observations. It answers to the question: "Given all observations, what is the probability that the system was in state $k$ at time $t$?"
#
# ```smooth_proba``` takes two arguments: a design matrix X and the observed neural activity y. The output is either a ```TsdFrame``` or an array of  posterior probabilities, shape ``(n_time_points, n_states)``. Each row sums to 1 and represents the probability distribution over states at that time point.

# %%
# Compute smooth_proba
posteriors = model.smooth_proba(X, choices_tsd)
print(f"First five osteriors \n{posteriors[:5]} \n")

# Each (non nan) row sums to 1
valid = ~np.isnan(posteriors).any(axis=1)
print(
    f"Each row sums to 1: {np.allclose(posteriors[valid].sum(axis=1), 1)}"
)


# %% [markdown]
# And we can plot it!

# %% tags=["hide-input"]
def plot_posteriors(posteriors):
    # Pick three sessions to plot
    sess_to_plot = [
        '0ccee376-2873-47dd-9293-c19e424c1bee',
        '66f20f92-171f-4cc5-aca9-69fc3cb6370f',
        '19f4acbd-aeac-4f83-9f30-85a8aa002820'
    ]

    # Get these sessions' indexes
    sess_ex_1 = np.where(session == sess_to_plot[0])[0]
    sess_ex_2 = np.where(session == sess_to_plot[1])[0]
    sess_ex_3 = np.where(session == sess_to_plot[2])[0]

    sess_examples = [sess_ex_1, sess_ex_2, sess_ex_3]

    colors =["#ff7f00", "#4daf4a", "#377eb8"]
    fig, ax = plt.subplots(1,3,figsize=(20, 4))

    for i, sess_ex in enumerate(sess_examples):
        for state in range(n_states):
            # Plot all trials for a given session and state
            ax[i].plot(
                posteriors[sess_ex][:, state],
                label="State " + str(state + 1), 
                lw=3,
                color=colors[state]
    )
            ax[i].set_title("Example session " + str(i + 1))
            if i == 0:
                ax[i].set_xticks(
                    [
                        sess_ex[0], 
                        sess_ex[0] + 45, 
                        sess_ex[0] + 90
                    ], 
                    [
                        "0", 
                        "45", 
                        "90"
                    ], 
                )
                ax[i].set_ylabel("P(state)")
                ax[i].set_xlabel("Trial #")
                ax[i].set_yticks(
                    [0, 0.5, 1], 
                    ["0", "0.5", "1"], 
                )
            else:
                ax[i].set_xticks(
                    [
                        sess_ex[0], 
                        sess_ex[0] + 45, 
                        sess_ex[0] + 90
                    ], 
                    [
                        " ", 
                        " ", 
                        " "
                    ], 
                )
                ax[i].set_yticks(
                    [0, 0.5, 1], 
                    [" ", " ", " "], 
                )
    return None


# %%
plot_posteriors(posteriors)

# %% [markdown]
# In these sessions, the posterior over latent states can be tracked at each trial, revealing strong confidence in state assignments and extended periods where a single state persists across consecutive trials. This pattern is inconsistent with the short, transient lapses assumed in earlier lapse-based models.

# %% [markdown]
# ### 04.5 Computing fraction of occupancy and accuracy per state using ```decode_state``` or ```smooth_proba```

# %% [markdown]
# We can also be interested in quantify state occupancies (i.e what proportion of the trials a given animal spent in each state) and accuracies per state. For this, we need the inferred sequence of states, and there are (at least) two ways in which we can obtain it: using ```decode_state``` or using ```smooth_proba```.

# %% [markdown]
# #### 4.5.1) Using ```decode_state```
# This method finds the single most likely sequence of hidden states that best explains the observed data. It uses the Viterbi algorithm to compute the state sequence that maximizes the joint probability of states and observations.
#
# This function takes three mandatory parameters, a matrix of predictors X of shape (n_timepoints,n_features), a np.array or nap.Tsd of observations of shap (n_time_points,), and the format of the returned states, either in one-hot encoding format or as an array of shape (n_time_points,) containing the decoded state at each timepoint.

# %%
# get output of viterbi in one-hot encoding
decoded_states = model.decode_state(X,choices_tsd, state_format = "one-hot")
print(f"{decoded_states} \n")

# calculate how many instances of occupancy there is in each of them
print(f"Total instances of each state {np.nansum(decoded_states, axis=0)} \n")

# calculate fraction of occupancy
frac_occupancy_viterbi= np.nansum(decoded_states, axis=0)/len(choices_tsd)
print(f"Fraction of occupancy {frac_occupancy_viterbi} \n")


# %% [markdown]
# Now, we can compute the general accuracy.

# %%
# See where the input is not 0
non_zero_contrast_loc = np.where(signed_contrast!=0)
non_zero_contrast = signed_contrast[non_zero_contrast_loc]

# Get correct answer by looking at sign
correct_ans_task = np.sign(non_zero_contrast)

# Transform into 0-1 to compare with choices
correct_ans_task_remapped = (correct_ans_task+ 1) / 2

# Get accuracy i.e how many choices match / how many choices were made
correct_ans_mouse = np.sum(choices_tsd[non_zero_contrast_loc] == correct_ans_task_remapped) 

total_accuracy = correct_ans_mouse/len(correct_ans_task)

# Create array of accuracies for plotting
accuracies_to_plot_viterbi = np.zeros([4,])

# Add total accuracy
accuracies_to_plot_viterbi[0] = total_accuracy

# %% [markdown]
# And then we can use our output of ```decode_state``` to segment the trials into the estimated states and compute the accuracy within each state.

# %%
for state in range(n_states): 
    # index of trials per state
    idx_this_state = np.where(decoded_states[:,state] == 1)
    
    # Get contrast and choices for this state
    signed_contrast_this_state = signed_contrast[idx_this_state]
    choices_this_state = choices_tsd[idx_this_state]
    
    # See where the input is not 0
    not_zero_contrast_loc_this_state = np.where(signed_contrast_this_state != 0)[0]
    non_zero_contrast_this_state = signed_contrast_this_state[not_zero_contrast_loc_this_state]
    
    # Get correct answer by looking at sign
    correct_ans_this_state = np.sign(non_zero_contrast_this_state)
    
    # Transform into 0-1 to compare with choices
    correct_ans_task_this_state_remapped = (correct_ans_this_state+ 1) / 2

    # Get accuracy i.e how many choices match / how many choices were made
    correct_ans_mouse_this_state = np.sum(choices_this_state[not_zero_contrast_loc_this_state] == correct_ans_task_this_state_remapped) 
    
    accuracy_this_state = correct_ans_mouse_this_state / len(correct_ans_this_state)
    
    # Add state accuracy for plotting
    accuracies_to_plot_viterbi[state+1] = accuracy_this_state

print(accuracies_to_plot_viterbi)


# %% [markdown]
# And we can plot this :)

# %% tags=["hide-input"]
def plot_accuracy_and_occupancy(frac_occupancy, accuracies_to_plot):
    cols = [
        "#ff7f00", "#4daf4a", "#377eb8", '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00'
        ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: state occupancies
    ax = axes[0]
    for z, occ in enumerate(frac_occupancy):
        ax.bar(z, occ, width=0.8, color=cols[z])
        ax.text(z, occ, f"{occ:.2f}", ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['1', '2', '3'])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('state')
    ax.set_ylabel('frac. occupancy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Right: accuracies
    ax = axes[1]
    for z, acc in enumerate(accuracies_to_plot):
        col = 'grey' if z == 0 else cols[z - 1]
        ax.bar(z, acc * 100, width=0.8, color=col)
        ax.text(z, acc * 100 + 1, f"{acc*100:.2f}", ha='center', va='bottom', fontsize=10)

    ax.set_ylim(50, 100)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['All', '1', '2', '3'])
    ax.set_yticks([50, 75, 100])
    ax.set_xlabel('state')
    ax.set_ylabel('accuracy (%)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return None


# %%
plot_accuracy_and_occupancy(frac_occupancy_viterbi, accuracies_to_plot_viterbi)

# %% [markdown]
# According to state occupancy derived with the Viterbi algorithm, this mouse spent the majority of the trials (71%) in the engaged state and a lesser portion of trials in the other two states (29%). We can see that even though this mouse had an overall accuracy of 80.36%, it achieved a higher accuracy of 86.79% in the "engaged" state compared to 66.07% and 62.16% in the "bias left" and "bias right", respectively.

# %% [markdown]
# #### 4.5.2) Using ```smooth_proba```
# Now we can compute the same quantities but using ```smooth_probs```. We used this method in 4.4 to compute the posterior probabilities! In contrast to ```decode_state```, which outputs the globally optimally state sequence, ```smooth_proba``` outputs probabilistic posteriors. With this alternative, we can go by the approach used in Ashwood et al. (2022): we can compute the posterior probability for each state at all times, and subset the trials for which there is high confidence (+90% probability) of being in a given state; then, we can assign each trial to its most likely state and count the fraction of trials assigned to each state. 
#
# The process is very similar to 4.5.1, with the difference in how we slice the trials and assign them to a specific state. We can start with the fraction of occupancy.

# %%
# Get most likely state on a trial by trial basis
states_max_posterior = np.argmax(posteriors, axis=1)
print(f"Most likely state trial by trial \n {states_max_posterior} \n")

# Calculate how many instances of occupancy there is in each of them
occupancy_per_state = np.unique(states_max_posterior, return_counts=True)[1]
print(f"Total instances of each state {occupancy_per_state} \n")

# calculate fraction of occupancy
frac_occupancy_smooth_proba = occupancy_per_state/len(choices_tsd)
print(f"Fraction of occupancy {frac_occupancy_smooth_proba } \n")

# %%
# Segment trials into states estimated on a trial by trial basis
idx_per_state = []
for state in range(n_states): 
    idx_per_state.append(np.where(posteriors[:, state] >= 0.9)[0])


# %% [markdown]
# With this segmentation, we can calculate accuracy in the exact same manner as in 4.5.1. 

# %% tags=["hide-input"]
def get_accuracies_to_plot(idx_per_state, total_accuracy=total_accuracy, n_states=n_states, signed_contrast=signed_contrast, choices_tsd=choices_tsd):
    # Total accuracy remains the same
    accuracies_to_plot = np.zeros([4,])
    # Use previously calculated total_accuracy
    accuracies_to_plot[0] = total_accuracy
    for state in range(n_states): 
        # index of trials per state
        idx_this_state = idx_per_state[state]

        # Get contrast and choices for this state
        signed_contrast_this_state = signed_contrast[idx_this_state]
        choices_this_state = choices_tsd[idx_this_state]
        
        # See where the input is not 0
        not_zero_contrast_loc_this_state = np.where(signed_contrast_this_state != 0)[0]
        non_zero_contrast_this_state = signed_contrast_this_state[not_zero_contrast_loc_this_state]
        
        # Get correct answer by looking at sign
        correct_ans_this_state = np.sign(non_zero_contrast_this_state)
        
        # Transform into 0-1 to compare with choices
        correct_ans_task_this_state_remapped = (correct_ans_this_state+ 1) / 2

        # Get accuracy i.e how many choices match / how many choices were made
        correct_ans_mouse_this_state = np.sum(choices_this_state[not_zero_contrast_loc_this_state] == correct_ans_task_this_state_remapped) 
        
        accuracy_this_state = correct_ans_mouse_this_state / len(correct_ans_this_state)
        
        # Add state accuracy for plotting
        accuracies_to_plot[state+1] = accuracy_this_state
    return accuracies_to_plot


# %%
accuracies_to_plot_smooth_proba = get_accuracies_to_plot(idx_per_state)

plot_accuracy_and_occupancy(frac_occupancy_smooth_proba,accuracies_to_plot_smooth_proba)

# %% [markdown]
# According to state occupancy derived by using the most likely state with the posterior distribution on a trial by trial basis, this mouse spent the majority of the trials (69%) in the engaged state and a lesser portion of trials in the other two states (32%). We can see that even though this mouse had an overall accuracy of 80.36%, it achieved a higher accuracy of 88.80% in the "engaged" state compared to 61.78% and 58.78% in the "bias left" and "bias right", respectively.
#
# Here, we obtained different results than in 4.5.1. This can be explained by the use of different algorithms for segmenting the trials. While Viterbi finds the most likely sequence of states as a whole, the method in 4.5.1 calculates what the most likely state is on a trial by trial basis, and only keeps the state with large confidence (>90%). 

# %% [markdown]
# ## 05. Conclusions and next steps [pending]
#
# follow state descriptions in the paper
# if you dont segment the paper then you would not be able to see some effects (see in what way they do that) -> explain at the very begining as a motivation and in the end as an interpretation -> why they thought to use this method
#
# - show the class or link to the class name and show all code again. Creatte design matrix and fit the model. Emphasize. This is brief and its quick. 
#
# The GLM-HMM captures structure in decision-making that is not explained by stationary models. Across sessions, behavior is better described as a mixture of a small number of latent strategies that persist over multiple trials rather than independent lapses around a single policy. This is visible both in the inferred posterior trajectories and in the Viterbi-decoded state sequences, which show extended dwell times within states and relatively infrequent switching.
#
# Each inferred state corresponds to a distinct decision policy: one primarily stimulus-driven, and others showing strong directional biases. The separation between these regimes is reflected consistently in both GLM weights and psychometric curves, indicating that the model recovers interpretable behavioral strategies rather than arbitrary partitions of the data.
# State occupancy and performance analyses further show that behavioral accuracy is not uniform across latent states. The stimulus-driven state yields higher task-aligned performance, while biased states show reduced accuracy, consistent with reduced sensitivity to sensory evidence. 
#
#
#

# %% [markdown]
# ## Additional resources [pending]
# - [Zoe Ashwood's SSM tutorial on GLM-HMMs](https://github.com/zashwood/ssm/blob/master/notebooks/2b%20Input%20Driven%20Observations%20(GLM-HMM).ipynb)

# %% [markdown]
# ## References
# <a id="ref1a"><a href="#cite1a">[1a]</a> <a id="ref1b"><a href="#cite1b">[1b]</a> <a id="ref1c"><a href="#cite1c">[1c]</a> <a id="ref1d"><a href="#cite1d">[1d]</a> <a id="ref1e"><a href="#cite1e">[1e]</a> Ashwood, Z. C., Roy, N. A., Stone, I. R., Laboratory, I. B., Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. Nature Neuroscience, 25(2), 201–212.
#
# <a id="ref2"><a href="#cite2">[2]</a> Bengio, Y., & Frasconi, P. (1995). An input-output HMM architecture. In G. Tesauro, D. S. Touretzky, & T. K. Leen (Eds.), Advances in neural information processing systems (Vol. 7, pp. 427–434). MIT Press.
#
# <a id="ref3a"><a href="#cite3a">[3a]</a> <a id="ref3b"><a href="#cite3b">[3b]</a> Escola, S., Fontanini, A., Katz, D., & Paninski, L. (2011). Hidden Markov models for the stimulus-response relationships of multistate neural systems. Neural Computation, 23(5), 1071–1132. https://doi.org/10.1162/NECO_a_00118
#
# <a id="ref4"><a href="#cite4">[4]</a> Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
#
# <a id="ref5a"><a href="#cite5a">[5a]</a><a id="ref5b"> <a href="#cite5b">[5b]</a> <a id="ref5c"><a href="#cite5c">[5c]</a> The International Brain Laboratory, Aguillon-Rodriguez, V., Angelaki, D., Bayer, H., Bonacchi, N., Carandini, M., Cazettes, F., Chapuis, G., Churchland, A. K., Dan, Y., Dewitt, E., Faulkner, M., Forrest, H., Haetzel, L., Häusser, M., Hofer, S. B., Hu, F., Khanal, A., Krasniak, C., … Zador, A. M. (2021). Standardized and reproducible measurement of decision-making in mice. eLife, 10, e63711. https://doi.org/10.7554/eLife.63711
#
# <a id="ref6"><a href="#cite6">[6]</a> Burgess, C. P., Lak, A., Steinmetz, N. A., Zatka-Haas, P., Bai Reddy, C., Jacobs, E. A. K., Linden, J. F., Paton, J. J., Ranson, A., Schröder, S., Soares, S., Wells, M. J., Wool, L. E., Harris, K. D., & Carandini, M. (2017). High-Yield Methods for Accurate Two-Alternative Visual Psychophysics in Head-Fixed Mice. Cell Reports, 20(10), 2513–2524. https://doi.org/10.1016/j.celrep.2017.08.047
