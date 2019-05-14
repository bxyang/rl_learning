# Temporal-Difference Learning

TD learning is a combination of Monte Carlo ideas and dynamic programming ideas.

Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the enviroment's dynamics.

Like DP, TD methods update estimates based in part on other learned estimates,
without waiting for a final outcome.

# Monte Carlo methods
## Monte Carlo Prediction

Each occurrence of state *s* in an episode is called a *visit* to *s* .
在同一个episode中，s可能会被*visited* 多次。

*first-visit MC method*

*every-visit MC method*

## Monte Carlo Estimation of Action Values

if a model is not available => estimate *action* values rather than state *values*

the general problem of *maintaining exploration*, how to solve?

by specifying that the episodes start in a state-action pair, and that every
pair has a nonzero probability of being selected as the start.

this guarantees that all state-action pairs will be visited an infinite number
of times in the limit of an infinite number of episodes.

we call this the assumption of *exploring starts* .


## Monte Carlo Control

Monte Carlo ES, for Monte Carlo with Exploring Starts


## Monte Carlo Control without Exploring Starts

how to avoid the unlikely assumption of exploring starts?
- *on-policy* methods: attempt to evaluate or improve the policy that is used to make decisions
- *off-policy* methods: evaluate or improve a policy different from that used to generate the data .

Monte Carlo ES is an example of an on-policy method.

In on-policy control methods, the policy is generally *soft*,
meaning that the $\pi(a|s) > 0$ for all $s \in S$ and all $a \in A(s)$ .  

$\epsilon$-greedy policies are example of $\epsilon$-soft policies.

## Off-policy Prediction via Importance Sampling

All learning control methods face a dilemma: the seek to learn action values conditional on subsequent optimal behavior, but they need to behave non-optimally in order to explore all actions(to find the optimal actions).

How can they learn about the optimal policy while behaving according to an exploratory policy ?

the on-policy approach in the preceding section is actually a compromise- it learns action values not for the optimal policy, but for  a near-Optimal policy that still explores.

a more straightforward approach is to use two policies,
- one that is learned about and that becomes the optimal policy
- one that is more exploratory and is used to generate behavior is called the behavior policy.

in this case we say that learning is from data "off" the target policy, and the overall process is termed *off-policy leanring* .


Suppose we wish to esimate $v_\pi \ or \ q_\pi$, but all we have are episodes following another policy $b$, where $b \neq \pi$, $\pi$ is the target policy, $b$ is the behavior policy, and both policies are consider fixed and given.

the assumption of *coverage* , every aciton taken under $\pi$ is also taken,
at least occasionally, under b.


*Importance Sampling*, a general technique for estimating expected values
under one distribution given samples from another.


## Incremental Implementation

## Off-policy Monte Carlo Control

## Discounting-aware Importance Sampling

## Per-decision Importance Sampling

# Dynamic Programming

## Policy Evaluation(Prediction)

compute the state-value function $v_\pi$ for an arbitrary policy $\pi$

we also refer to it as the prediction problem.

$v_{k+1}(s) = E_{\pi}[R_{t+1} + \gamma v_k(S_{t+1})|S_t=s]
= \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r + \gamma v_k(s') ]$

the sequence {$v_k$} can be shown in general to converge to $v_\pi$ as
$k \to \infty$

this algo is called *iterative policy evaluation*, $v_{k+1}$ from $v_k$,
this kind of operation is called an expected update.



## Policy Improvement
if $q_\pi(s,\pi'(s)) \ge v_\pi(s)$, then $v_{\pi'}(s) \ge v_\pi(s)$
this is called *policy improvement theorem*.

new greedy policy $\pi'$, given by
$$\begin{split}
\pi'(s) &= \arg \max_\limits{a}q_{\pi}(s, a)\\
\\
&= \arg \max_\limits{a}E[R_{t+1} + \gamma v_{\pi}(S_{t+1})|S_t=s,A_t=a]\\
\\
&= \arg \max_\limits{a}\sum_{s',r}p(s',r|s,a)[r + \gamma v_\pi(s')]
\end{split}$$

this is called *policy improvement*.

## Policy Iteration

$\pi_0 \stackrel{E}{\rightarrow} v_{\pi_0} \stackrel{I}{\rightarrow}\pi_1 \stackrel{E}{\rightarrow} v_{\pi_1} \stackrel{I}{\rightarrow}\pi_2 \stackrel{E}{\rightarrow} v_{\pi_2} \stackrel{I}{\rightarrow} ... \pi_*
 \stackrel{I}{\rightarrow} v_*$

$\stackrel{E}{\rightarrow}$ denotes a policy *evaluation*

$\stackrel{I}{\rightarrow}$ denotes a policy *improvement*

this is called *policy iteration*

## Value Iteration
$$\begin{split}
v_{k+1}(s) &= E_{\pi}[R_{t+1} + \gamma v_k(S_{t+1})|S_t=s] \\
\\
&= \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r + \gamma v_k(s') ]
\end{split}$$

## Asynchronous Dynamic Programming

无锁，类似异步随机梯度下降的思想

## Generalized Policy Iteration

***GPI*** refer to the general idea of letting policy-evaluation and policy-improvement processes interact, independent of the granularity and other details of the two processes.

## Efficiency of Dynamic Programming



# MDP

## Agent-Environment Interface

$$p(s',r|s,a)=Pr\{S_t=s',R_t=r|S_{t-1}=s,A_{t-1}=a\}$$

## Goals and Reward

the use of a reward signal to formalize the idea of a goal is one of the most distinctive features of reinforcement leanning

## Returns and Episodes


### Expected Return
maximize expected return $G_t$

$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R\_{T}$, T is a  final time step


### episodic tasks
the agent-enviroment interaction breaks naturally into subsequences, which we call **_episodes_**
, each episode ends in a special state called the **_terminal state_**

### continuing tasks
**_discounting_**, maximize the expected *discounted return*

$G_{t}=R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}+... = \sum_{k=0}^\infty\gamma^kR_{t+k+1}$

$\gamma$ is called the *discount rate*

### Unified Notation for Episodic and Continuing Tasks

### Policies and Value Functions

the value function of a state s under a policy $\pi$, denoted $v_{\pi}(s)$ is
the expected return when starting in $s$ and following $\pi$ thereafter.

$v_{\pi}(s) = E_{\pi}[G_t|S_t=s]$

$v_{\pi}$: the state-value function for policy $\pi$

$q_{\pi}(s, a) = E_{\pi}[G_t|S_{t} = s, A_{t} = a]$

$q_{\pi}$: the action-value function for policy $\pi$

the value funcions $v_{\pi}$ abd $q_{\pi}$ can be estimated from experience.

*Monte Carlo methods*: estimation methods

value function的基本特点之一是存在某种递归的关系。

$$\begin{split}
v_{\pi}(s) &= E_{\pi}[G_t|S_t=s]\\
\\
&=E_{\pi}[R_{t+1}+ \gamma G_{t+1}|S_t=s]\\
\\
&=\sum_{a}\pi(a|s)\sum_{s'}\sum_{r}p(s',r|s,a)\left[r+\gamma E_{\pi}\left[G_{t+1}|S_{t+1}=s'\right]\right]
\\
&=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r \ + \ \gamma v_{\pi}(s') ]
\end{split}$$


### Optimal Policies and Optimal Value Functions
$v_*(s) = \max_\limits{\pi} \ v_\pi(s)$, *optimal state-value function*

$q_*(s,a)=\max_\limits{\pi}q_{\pi}(s,a)$, *optimal action-value function*

$q_*(s,a)=E[R_{t+1}+\gamma v_*(S_{t+1})|S_t=s, A_t=a]$


### Optimality and Approximation



# Ch1 Introduction

## Elements of RL

### four main subelements
- Policy
- Reward signal
- Value function
- model of the enviroment


Models are used for planning, by which we mean any way of deciding
on a course of action by considering possible future situations before they
are actually experienced.

Methods for solving reinforcement learning problems that use models and planning are called *model-based* methods, as opposed to simpler *model-free*
methods that are explicitly trial-and-error learners-viewed as almost the opposite of planning.
