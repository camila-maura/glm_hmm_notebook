
GLM-HMMs, also known as input-out HMM models (Bengio & Frasconi, 1995) <span id="cite2"></span><a href="#ref2">[2]</a>,  are useful to analyze how hidden latent states affect observable behavioral (Ashwood et al., 2022) <span id="cite1a"></span><a href="#ref1a">[1a]</a> and neural (Escola et al., 2011)<span id="cite3a"></span><a href="#ref3a">[3a]</a> dynamics. These models are composed by an HMM, governing the distribution over the latent states, and state-specific GLMs, which specify the activity of the system at each state.

![Graphical model of GLM-HMM](../assets/graphical_model.png)

+++

In all GLM-HMMs, the HMM component is fully defined by three elements: a state transition matrix, an initial probability vector and an emissions probability distribution (Bishop, 2006)<span id="cite4"></span><a href="#ref4">[4]</a>. A HMM with K hidden states has a $K \times K$ transition matrix that specifies the probability of transitioning from any state to any other,

\begin{align}
p(z_t=j\mid z_{t-1} = i) = A_{ij}
\end{align}

where $z_{t-1}$ and $z_t$ indicate the latent state at trials $t-1$ and $t$, respectively. The HMM also has a distribution over the initial states, given by a $K$-element vector $\pi$ whose elements sum to one:

\begin{align}
p(z_1 = i) = \boldsymbol{\pi}_i
\end{align}

Finally, the emissions probability describes the relationship between the state and the observation. In the case of GLM-HMMs, the emissions probability is a GLM (it can be, for example, a Bernoulli GLM); that is, a generalization of linear regression that allows to characterize how an output (behavior, neuronal activity) may vary as a function of an input.

A $K$-state GLM-HMM contains $K$ independent GLMs, each defined by a weight vector specifying how inputs are integrated in that particular state to give rise to activity. These describe the state-dependent mapping from inputs to activity. In this tutorial, we will use a GLM-HMM with a Bernoulli observation model i.e., a Bernoulli GLM.

\begin{align}
y_t \mid \boldsymbol{x}, \boldsymbol{k} \sim Ber(f(-\boldsymbol{x}_t \cdot \boldsymbol{w}_k)) \\
\end{align}

where $\boldsymbol{w}_k \in \mathbb{R}^M$ denotes the GLM weights for latent state $k \in {1,..,K}$. Thus, the probability of success ($y_t = 1$, which can correspond to a given choice in a binary set up, or a spike count for a time bin; in our case, corresponds to a leftward choice) given the input vector $\boldsymbol{x}_t$ is given by:

\begin{align}
p(y_t=1\mid\boldsymbol{x}_t, z_t = k)  = \frac{1}{1+exp(-\boldsymbol{x}_t \cdot \boldsymbol{w}_k)}
\end{align}
considering a Bernoulli GLM with a logistic inverse link function.
