---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
---

# GLM-HMMs: an overview
## What are GLM-HMMs?
GLM-HMMs, also known as input-out HMM models (Bengio & Frasconi, 1995) <span id="cite2"></span><a href="#ref2">[2]</a>,  are useful to analyze how hidden latent states affect observable behavioral (Ashwood et al., 2022) <span id="cite1a"></span><a href="#ref1a">[1a]</a> and neural (Escola et al., 2011)<span id="cite3a"></span><a href="#ref3a">[3a]</a> dynamics. These models are composed by an HMM, governing the distribution over the latent states, and state-specific GLMs, which specify the activity of the system at each state.


```{figure} ../assets/graphical_model.png
:width: 1000%
:alt: Graphical model GLM-HMM
:align: left
Graphical model of a GLM-HMM. The state $z_t$ at time $t$ depends only on the previous state $z_{t-1}$. The observation $y_t$ is independent of all previous observations conditioned on the current hidden state $z_t$. The observation $y_t$ is determined by the parameters of $\mathrm{GLM}_{z_t}$, conditioned on $z_t$. GLMs are models that describe how the output of a system $y_t$ varies as a function of some input $x_t$.
```

+++

In all GLM-HMMs, the HMM component is fully defined by three elements: a state transition matrix, an initial probability vector and an emissions probability distribution (Bishop, 2006)<span id="cite4"></span><a href="#ref4">[4]</a>. A HMM with K hidden states has a $K \times K$ transition matrix $\boldsymbol{A}$ that specifies the probability of transitioning from any state to any other,

\begin{align}
p(z_t=j\mid z_{t-1} = i) = A_{ij}
\end{align}

where $z_{t-1}$ and $z_t$ indicate the latent state at trials $t-1$ and $t$, respectively. The HMM also has a distribution over the initial states, given by a $K$-element vector $\pi$ whose elements sum to one:

\begin{align}
p(z_1 = i) = \pi_i,\quad \sum_{i=1}^{K} \pi_i = 1
\end{align}

Finally, the emissions probability describes the relationship between the state and the observation. In the case of GLM-HMMs, the emissions probability is a GLM; that is, a generalization of linear regression that allows to characterize how an output (behavior, neuronal activity) may vary as a function of an input.  A $K$-state GLM-HMM contains $K$ independent GLMs, each defined by a weight vector specifying how inputs are integrated in that particular state to give rise to activity. These describe the state-dependent mapping from inputs to activity. 


```{admonition} Note
:class: tip
Take a look at the NeMoS GLM documentation if you need a review on GLMs
```

For example, with a Bernoulli GLM and looking at a single timepoint we would have

\begin{align}
y_t \mid \mathbf{x}_t, z_t 
\sim \mathrm{Bernoulli}\left(
\sigma(\mathbf{w}_{z_t} \mathbf{x}_t)
\right)
\end{align}

where $\mathbf{x}_{t}\in \mathbb{R}^M$ corresponds to the input at time $t$ and $\mathbf{w}_{z_t}\in \mathbb{R}^M$ denotes the GLM weights vector corresponding to the latent state $z$ active at time $t$. In the Bernoulli case, the probability of $y_t = 1$ (which can correspond to a given choice in a binary set up, or a spike count for a time bin) given the input vector $\boldsymbol{x}_t$ and the current state $z_t$ is characterized by:

\begin{align}
p(y_t=1\mid\boldsymbol{x}_t, z_t = k)  = \frac{1}{1+exp(-\boldsymbol{x}_t \cdot \boldsymbol{w}_k)}
\end{align}
considering a logistic inverse link function.

```{admonition} Note
:class: tip
- List of observation models
- List of inverse link functions per model
```

+++


## How to fit a GLM-HMM?

When fitting a GLM-HMM, the goal is to learn the set of parameters $\boldsymbol{\theta} \equiv \{A,\boldsymbol{\pi},\{\boldsymbol{w}_k\}_{k=1}^K\}$ :

- the transition matrix $\boldsymbol{A} \in \mathbb{R}^{K \times K}$
- the initial state distribution $\boldsymbol{\pi} \in \mathbb{R}^K$
- the set of weights that influence the activity in each state $\{\boldsymbol{w}_k\}_{k=1}^K$ with $\boldsymbol{w}_k \in \mathbb{R}^M$

that maximize the likelihood of the observed data. We can obtain the likelihood function by marginalizing over the latent variables:
\begin{equation}
p(\mathbf{y}|\mathbf{x}, \boldsymbol{\theta}) = \sum_{Z} p(\boldsymbol{y}, \boldsymbol{z} | \boldsymbol{x},\boldsymbol{\theta})
\end{equation}

 \begin{equation}
     p(\boldsymbol{y}|\boldsymbol{\theta}) = p(z_1|\boldsymbol{\pi}) \left[\prod_{t=2}^Tp(z_t \mid z_{t-1},A)\right]\prod_{m=1}^T p(y_m|z_m, \boldsymbol{w}_m)
 \end{equation}

and we obtain this from the joint distribution over both latent and observed variables:

 \begin{equation}
     p(\boldsymbol{y},\boldsymbol{z} | \theta) = p(z_1|\boldsymbol{\pi}) \left[\prod_{t=2}^Tp(z_t \mid z_{t-1},A)\right]\prod_{m=1}^T p(y_m|z_m, \boldsymbol{w}_m)
 \end{equation}

For this, we want to find the values that maximize the joint probability of the observations (choices or neural activity), the input data, the features and the states given the model parameters $\theta$:

We can obtain $\theta$ using the Expectation Maximization (EM) algorithm (Bishop, 2006) [PENDING]()</a>. It consists of alternating between two steps: "Expectation" and "Maximization" or E and M until convergence. During the E-step, we compute the 'expected complete data log-likelihood'(ECLL), which is a lower bound on the log-likelihood of the data given the parameters, using some initial selection of the parameters $\theta^{\text{old}}$:

\begin{equation}
\label{eq:ECLL_def}
\text{ECLL}(\theta) = \sum_z p(\textbf{z}\mid \boldsymbol{y}, \{\boldsymbol{x}\}^T_{t=1}\,\theta^{\text{old}}) \log p(\textbf{y}, \textbf{z}\mid \theta, \{\boldsymbol{x}\}^T_{t=1}) \\
\end{equation}

Now, we can introduce $\gamma_{t,k)}$ to denote the marginal posterior distribution of $z_t = k$
\begin{align}
\gamma_{t,k} = p(z_t = k \mid \boldsymbol{y}, \{\boldsymbol{x}\}^T_{t=1}, \theta^{\text{old}})
\end{align}

and $\xi_{t,j,k}$ to denote the joint posterior state distribution for two consecutive latents $j$ and $k$
\begin{align}
\xi_{t,j,k} = p(z_t = k, z_{t-1} = j \mid \boldsymbol{y}, \{\boldsymbol{x}\}^T_{t=1}, \theta^{\text{old}}) 
\end{align}

If we substitute the definition of the joint distribution for GLM-HMM,
\begin{equation}
p(\textbf{y}, \textbf{z} \mid \{\boldsymbol{x}\}_{t=1}^T,\theta) = p(z_1)p(y_1 \mid z_1, \boldsymbol{x}_1) \prod_{t=2}^T p(z_t \mid z_{t-1}) p(y_t \mid z_t,\boldsymbol{x}_t) 
\end{equation}

into Eq. \eqref{eq:ECLL_def} and make use of the definitions of $\gamma$ and $\xi$, we get:
```{math}
:label: eq_ecll_separated
\begin{split}
\text{ECLL}(\theta) = \sum_{k=1}^K \gamma_{1,k} \log \boldsymbol{\pi}_k + \sum_{t=1}^T \sum_{j=1}^K \sum_{k=1}^K \xi_{t,j,k}\log A_{jk} \\
+ \sum_{t=1}^T \sum_{k=1}^K \gamma_{t,k} \log p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}_k)
\end{split}
\label{eq_ecll_separated}
```
The single and joint state probabilities, $\gamma_{t,k}$ and $\xi_{t,j,k}$ respectively, are estimated using the forward backward algorithm

### How to fit a GLM-HMM?: Forward Backward Algorithm
During the E-step the single and joint posterior state probabilities for all trials and states are estimated using the forward-backward algorithm at the current setting of the GLM-HMM parameters, $\theta^{\text{old}}$.

The goal of the forward pass is to obtain, for each trial t and each state k, the quantity:

\begin{equation}
\alpha_{t,k} \equiv p(\boldsymbol{y}_{1:t}, z_t \mid \{\boldsymbol{x}_{t'}\}_{t'=1}^t)
\end{equation}

which represents the posterior probability of the acitivy up until trial $t$ and the latent state at trial $t$ being state $k$. The posterior probability associated with trial 1, $\alpha_{1,k}$ can be calculated:
\begin{equation}
\alpha_{1,k} = \boldsymbol{\pi}_k p(\boldsymbol{y}_{1} \mid z_1,\boldsymbol{x}_1, \boldsymbol{w}_k)
\end{equation}

where, in our case,  $p(\boldsymbol{y}_{1} \mid z_1,\boldsymbol{x}_1, \boldsymbol{w}_k)$ is the usual Bernoulli GLM distribution mentioned in an equation above.

For trials $1<t \leq T$, we can obtain the probabilities:

\begin{equation}
\alpha_{t,k} = \sum_{j=1}^K \alpha_{t-1,j}A_{j,k}p(y_t \mid z_t = k, \boldsymbol{x}_t, \boldsymbol{w}_k)
\end{equation}

During the backward pass, the goal is to calculate the posterior probability of the choice data beyond the current trial, $\beta_{t,k}$, for each trial $t$ for all states $k$:
\begin{equation}
\beta_{t,k} \equiv p(\boldsymbol{y}_{[t+1:T]}\mid z_t = k, \{\boldsymbol{x}_{t'}\}^T_{t'=t+1})
\end{equation}
These quantities can be calculated recognizing that
\begin{equation}
\beta_{T,k} = 1
\end{equation}
and, for $t \in \{T-1, ...,1\}$:
\begin{equation}
\beta_{t,j} = \sum_{k=1}^K\beta_{t+1,k}A_{jk}p(y_{t+1}\mid z_{t+1} = k, \boldsymbol{x}_{t+1},\boldsymbol{w}_k
\end{equation}

From the $\alpha_{t,k}$ and $\beta_{t,k}$ quantities obtained using the forward-backward algorithm, we can form the single and joint posterior state probabilities $\gamma_{t,k}$ and $\xi_{t,j,k}$, which we need to maximize the ECLL:
\begin{align}
\gamma_{t,k}
    &= p(z_t = k \mid D, \theta^{\text{old}})\\
    &= \frac{p(y_{[0:t]}, z_t = k \mid \{\boldsymbol{x}\}_{t'=1}^t, \theta^{\text{old}}) p(y_{[t+1:T]}, z_t = k \mid \{\boldsymbol{x}\}_{t'=t}^T, \theta^{\text{old}})}{p(\boldsymbol{y} \mid \{\boldsymbol{x}\}_{t'=1}^T\theta^{\text{old}})} \\
    &= \frac{\alpha_{t,k}\beta_{t,k}}{\sum_{k=1}^K\alpha_{T,k}}
\end{align}
And do similarly for $\xi$:
\begin{equation}
    \xi_{t,j,k} = \frac{\alpha_{t,j}A_{j,k}\beta_{t+1,k}p(y_{t+1}\mid z_{t+1} = k, \boldsymbol{x}_{t+1}, \boldsymbol{w}_k)}{\sum_{k=1}^K\alpha_{T,k}}
\end{equation}
where $p(y_{t+1}\mid z_{t+1} = k, \boldsymbol{x}_{t+1}, \boldsymbol{w}_k)$ is the Bernoulli GLM distribution.

In the *M step*, after having run the forward-backward algorithm, we maximize the ECLL with respect to the GLM-HMM parameters, $\theta$. For the initial distribution $\pi$ and the transition matrix $\boldsymbol{A}$, this results in the closed form updates:
\begin{align}
    \pi_k^\text{new} 
    & = \frac{\gamma_{1,k}}{\sum_{j=1}^K\gamma_{1,j}} \\
     A_{jk}^\text{new} & = \frac{\sum_{t=2}^T \xi_{t,j,k}}{\sum_{t=2}^T\sum_{k=1}^K \xi_{t,j,k} }
\end{align}
The last element we are left with is the likelihood of the GLM weights, which can be updated numerically, with certain caveats. From [](#eq_ecll_separated), it is easy to see that the optimization problem consists of maximizing three terms independently. There are analytical updates for $\pi$ and $A$, but the weights $w$ defining the emission probabilities must be estimated using gradient descent.

## Problems with EM Fitting
This procedure is guaranteed to find a larger likelihood solution at each iteration, but there is no guaranty of a local optimum  in all landscapes (Salakhutdinov, 2003) [PENDING](). Moreover, even with a convex emissions probability distribution, a global optimum is not guaranteed (Salakhutdinov, 2003) [PENDING](). This is not exclusive to GLM-HMMs, but general to the use of EM, which is known for being sensitive to the values of the initialization. Thus, efficient initialization, as well as a clear understanding of the algorithm and its components, are key to ensure that we get to the best local maximum of the likelihood function. 

## Given this framework, what can we be interested in knowing?
## Most likely sequence of states: Viterbi algorithm


## Additional resources

## References
<a id="ref1a"><a href="#cite1a">[1a]</a> <a id="ref1b"><a href="#cite1b">[1b]</a> <a id="ref1c"><a href="#cite1c">[1c]</a> <a id="ref1d"><a href="#cite1d">[1d]</a> <a Ashwood, Z. C., Roy, N. A., Stone, I. R., Laboratory, I. B., Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. Nature Neuroscience, 25(2), 201–212.

<a id="ref2"><a href="#cite2">[2]</a> Bengio, Y., & Frasconi, P. (1995). An input-output HMM architecture. In G. Tesauro, D. S. Touretzky, & T. K. Leen (Eds.), Advances in neural information processing systems (Vol. 7, pp. 427–434). MIT Press.

<a id="ref3a"><a href="#cite3a">[3a]</a> <a id="ref3b"><a href="#cite3b">[3b]</a> Escola, S., Fontanini, A., Katz, D., & Paninski, L. (2011). Hidden Markov models for the stimulus-response relationships of multistate neural systems. Neural Computation, 23(5), 1071–1132. https://doi.org/10.1162/NECO_a_00118

<a id="ref4"><a href="#cite4">[4]</a> Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
