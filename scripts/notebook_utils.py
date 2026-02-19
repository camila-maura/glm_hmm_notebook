# Imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo
import pynapple as nap

# Utils
def plot_glm_weights(
    n_features,
    n_states,
    true_projection_weights,
    learned_intercept,
    learned_coef,
    X_labels,
    initialization_setting,
):
    ## Plot
    fig = plt.figure(figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
    cols = ["#ff7f00", "#4daf4a", "#377eb8"]
    recovered_weights = np.zeros_like(true_projection_weights)

    recovered_weights[:1] = learned_intercept
    recovered_weights[1:] = learned_coef

    for k in range(n_states):
        if k == 0:
            plt.plot(
                range(n_features),
                true_projection_weights[:, k],
                marker="o",
                color=cols[k],
                linestyle="-",
                lw=1.5,
                label="Ashwood et al. (2022)",
            )
            plt.plot(
                range(n_features),
                recovered_weights[:, k],
                color=cols[k],
                lw=1.5,
                label="NeMoS GLM-HMM",
                linestyle="--",
            )
        else:
            plt.plot(
                range(n_features),
                true_projection_weights[:, k],
                marker="o",
                color=cols[k],
                linestyle="-",
                lw=1.5,
                label="",
            )

            plt.plot(
                range(n_features),
                recovered_weights[:, k],
                color=cols[k],
                lw=1.5,
                label="",
                linestyle="--",
            )
            
    
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=15)
    plt.xlabel("covariate", fontsize=15)
    plt.xticks([i for i in range(n_features)], X_labels, fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.legend()
    plt.title(f"Weight recovery - {initialization_setting}", fontsize=15)
    plt.tight_layout()
    plt.savefig("figures/fig_07_glm_hmm_behavioral.png")

    plt.show()
    return None


def load_session_fold_lookup(file_path):
    '''By Ashwood
    '''
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table
session_fold_table = load_session_fold_lookup("IBL/CSHL_008_session_fold_lookup.npz")

def load_data(animal_file):
    '''By Ashwood
    '''
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session
