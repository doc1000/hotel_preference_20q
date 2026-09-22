# Hotel Preference Elicitation as an Information-Gain Problem

This note develops a deliberately small version of a hotel recommendation problem. A user has already narrowed the world to **Hotel A** and **Hotel B**. Both hotels have rich labelled data. The goal is not to build a complete recommender profile. The goal is to ask as few follow-up questions as possible before the user can comfortably make the current decision.

The core framing is closer to **20 Questions / active preference learning** than to ordinary ranking.

The most important design principle is:

> Become confident about the current decision faster than you become confident about the person.

That distinction lets the system be useful without turning the user into an over-fit recommendation profile.

---

## Progressive disclosure: the whole idea in seven layers

You can stop at any layer. Each later layer solves a limitation of the earlier one.

### Layer 1 — Ask about the largest hotel difference

Put both hotels in the same normalized feature space:

```text
                 Hotel A      Hotel B
quiet              0.2          0.9
room_size          0.4          0.8
walkability        0.9          0.5
nightlife          0.9          0.2
service            0.8          0.8
```

Compute:

\[
\Delta = x_A - x_B
\]

Then ask about the feature with the largest standardized absolute difference.

This is a good baseline. It avoids wasting a question on something the hotels share.

It is not yet information-theoretic. A large hotel difference may be irrelevant to the user, or may not tell us much about the final A/B decision.

---

### Layer 2 — Use PCA to find compound hotel contrasts

Real hotel attributes are correlated. Nightlife, restaurants, and walkability may describe one broader concept; quietness, grounds, and room size may describe another side of the same tradeoff.

Fit PCA on a large matrix of labelled hotels:

\[
Z = XW
\]

A component might look like:

```text
+ nightlife
+ restaurants
+ walkability
- quiet
- grounds
```

An LLM or simple templating layer can translate that into a useful concept:

> central / energetic / walkable ↔ quiet / secluded

Project Hotel A and B onto those global axes and identify the components where they are far apart.

This gives more natural compound questions than asking about one feature at a time.

Important limitation: PCA optimizes variance in the **hotel population**, not usefulness for the **current user's choice**.

---

### Layer 3 — Represent uncertainty about the user with preference particles

Let a user's latent preference vector be:

\[
w = [w_{quiet}, w_{room}, w_{walkability}, \ldots]
\]

and define hotel utility as approximately:

\[
U(h) = w^T x_h
\]

For two hotels:

\[
U(A)-U(B) = w^T(x_A-x_B)=w^T\Delta
\]

We do not know \(w\), so represent it as many plausible particles:

```python
prefs.shape == (20_000, n_features)
```

Each row is one plausible current preference function. The proportion of weighted particles for which \(w^T\Delta>0\) gives an estimate of:

\[
P(A \text{ preferred})
\]

The point of particles is not that they are the only way to do Bayesian inference. They are simply very easy to inspect, perturb, filter, and visualize.

---

### Layer 4 — Seed the prior weakly instead of starting totally random

A completely random user prior throws away useful context.

There are at least two weak signals available before the first question.

#### 4a. Hotel-population geometry

The labelled hotel population tells us which features tend to vary together. That covariance can give the preference prior some geometry:

\[
w \sim \mathcal{N}(\mu_0, \Sigma_0)
\]

where \(\Sigma_0\) is partly informed by the hotel feature covariance.

This is only a structural prior. Hotel covariance is not the same thing as human preference covariance. It is a useful place to start, not a truth claim.

#### 4b. The fact that A and B were shortlisted

The two hotels themselves provide weak evidence.

If A and B are both unusually quiet, boutique, expensive, and pet-friendly relative to the population, that suggests those shared characteristics may matter to the current user or current trip.

A simple seed is:

\[
\mu_0 \leftarrow \alpha \cdot \frac{x_A+x_B}{2}
\]

with a deliberately small \(\alpha\).

The principle is more important than the exact formula:

> Shortlisting nudges the prior. It should not define the user.

---

### Layer 5 — Separate global hotel structure from user uncertainty

This is the most useful conceptual split in the design.

There are really three different objects:

```text
1. HOTEL WORLD
   What combinations of attributes exist?

2. USER POSTERIOR
   What preference combinations are still uncertain?

3. CURRENT PAIR
   Where do A and B actually differ?
```

Global hotel PCA can represent the first.

For the second, perform PCA/eigendecomposition on the **current weighted preference particles**. The leading components are not "what the user likes." They are:

> the largest unresolved directions in the current preference posterior.

Suppose the largest posterior uncertainty axis is approximately:

```text
+ quiet
+ room_size
- nightlife
- walkability
```

That means the largest thing we still do not know is whether the user is more on the quiet/spacious side or the active/central side.

But that uncertainty is only useful if A and B differ along it.

For an uncertainty axis \(v\), calculate pair relevance:

\[
|\Delta^T v|
\]

A large value means that resolving this particular preference uncertainty would matter for the current pair.

This yields the useful intersection:

\[
\boxed{
\text{Good question}
=
\text{unresolved preference direction}
\cap
\text{A/B differentiation}
}
\]

---

### Layer 6 — Rank questions by expected information gain

The previous layers generate plausible candidate questions. Information gain chooses among them.

The uncertainty that matters most is not necessarily entropy over the entire preference vector. It is the current binary decision:

\[
P(A), \quad P(B)
\]

Its entropy is:

\[
H(A/B)=-P(A)\log_2 P(A)-P(B)\log_2 P(B)
\]

For a candidate question \(Q\):

\[
IG(Q)
=
H(\text{before})
-
\sum_a P(a)H(\text{after answer }a)
\]

Ask the question with the largest expected reduction in A/B entropy.

This catches an important distinction:

> Difference between hotels is not the same as information about the decision.

A feature can differ dramatically between A and B but still be a poor question if learning the user's preference on that feature barely changes which hotel wins.

The script considers both posterior PCA axes and raw one-feature axes as candidates. This is intentionally conservative: a compound model should not hide a single highly informative feature.

---

### Layer 7 — Learn slowly and deliberately preserve uncertainty

A naive particle update might simply delete every particle inconsistent with the user's answer. That is too aggressive.

It assumes:

- the user is perfectly consistent;
- the question was interpreted perfectly;
- the preference model is correctly specified;
- preferences do not change with context.

All four assumptions are usually false.

The demo keeps four uncertainty controls separate so their roles remain visible.

#### 7a. Soft likelihood

Instead of keeping/deleting particles, assign them likelihoods.

If a particle predicted the observed answer:

\[
P(answer|w)=0.78
\]

otherwise perhaps:

\[
P(answer|w)=0.22
\]

A contradictory answer weakens a hypothesis rather than destroying it.

#### 7b. Tempered likelihood / slow learning rate

Update with:

\[
P(w|D) \propto P(D|w)^\eta P(w)
\]

for \(0<\eta<1\).

With \(\eta=0.5\), evidence accumulates more slowly. This is a simple and explicit anti-overfitting control.

#### 7c. Preference diffusion

Inject a small amount of noise after updates:

\[
w_{t+1}=w_t+\epsilon
\]

This creates an uncertainty floor and represents the idea that preferences are not fixed physical constants.

#### 7d. Permanent population-prior mixture

Retain a small exploratory component:

\[
P_t(w)
=
(1-\epsilon)P_{user}(w)
+
\epsilon P_{population}(w)
\]

This gives the model a formal way to say:

> I have a strong idea of what you usually like, but I am not certain this context is the same.

You probably do not need all four mechanisms on day one. The script includes all four because they solve slightly different failure modes and are easiest to understand when exposed separately.

A practical starting order would be:

1. soft likelihood;
2. likelihood tempering;
3. add prior mixture if profiles become too narrow;
4. add diffusion if you want explicit preference drift over time.

---

# Long-term preference versus trip/context preference

A useful future extension is to separate:

\[
w = w_{stable} + w_{context}
\]

For example:

```text
stable profile
    dislikes giant resorts
    usually values good service
    moderately price sensitive

current trip profile
    work trip
    strongly values walkability
    does not care about room size
    wants low logistical friction
```

The stable vector should learn slowly. The context vector can learn quickly and then be discarded or archived after the trip.

This naturally generalizes to multiple latent recurring profiles:

```text
same person
   ├── work travel
   ├── partner vacation
   ├── solo outdoor trip
   └── family visit
```

The next problem then becomes **context/profile selection**: which latent preference regime should be active now?

One attractive direction is similar to path isolation in transition systems. Instead of assuming one user vector, maintain a small set of recurring preference states and infer which state most plausibly generated the current behavior. Transitions between states can themselves have probabilities. That can eventually look like a mixture model, hidden Markov model, or transition-matrix style state system.

This is intentionally not implemented in the initial script. It is a separate layer rather than something the base decision loop requires.

---

# Why this works well in a chat agent

The chat agent should not hide the preference model. It can use the model to expose **important assumptions** in ordinary language.

A useful interaction pattern is:

```text
Agent:
Hotel B currently looks somewhat more likely to fit.

The main assumptions driving that are that you care more about quiet,
room size, and beach access than nightlife.

The biggest thing I am still unsure about is whether you want a more
central/active experience or a quieter/spacious one.

Would you rather ... ?
```

The user can simply correct the model:

```text
User:
Actually nightlife barely matters, but walkability matters a lot.
```

That statement can become a direct posterior update, potentially stronger than an answer to a generated multiple-choice question.

The agent can therefore use three kinds of evidence:

1. inferred evidence from selections;
2. answers to deliberate high-information questions;
3. explicit natural-language corrections to the current assumptions.

The third is especially valuable because chat creates a transparent feedback channel that conventional recommenders often lack.

---

# Avoiding analysis paralysis

The agent's objective is not to maximize certainty indefinitely.

A practical interaction has a transition from **analysis** to **commitment**.

Early:

> I am close to 50/50. Let me ask one high-value question.

Middle:

> Hotel B is ahead, but one unresolved preference could still flip the result.

Late:

> Hotel B looks like the better fit based on what you've told me. Do you want to go with Hotel B?

That final question is important. Once the decision entropy is sufficiently low, additional preference elicitation has diminishing value and can make the experience worse.

A possible stopping rule is:

```text
stop if:
    max(P(A), P(B)) > confidence threshold
OR
    best remaining question has information gain < minimum value
OR
    user indicates they are ready to choose
```

The conceptual target is:

\[
H(A/B) \rightarrow \text{low}
\]

not:

\[
H(w) \rightarrow 0
\]

The system should become certain enough to support the current choice while remaining humble about the person.

---

# Suggested implementation progression

The repository is easiest to explore in this order:

```text
Version 0
    standardized feature differences

Version 1
    global hotel PCA for compound contrasts

Version 2
    preference particles + weak prior

Version 3
    posterior PCA / unresolved preference axes

Version 4
    expected information gain

Version 5
    soft updates + tempering

Version 6
    diffusion / permanent prior mixture

Later
    stable vs trip-specific vectors
    multiple recurring context profiles
    natural-language assumption correction
    learned question generation
```

If a simpler version produces good decisions, keep it. The later layers are useful only when they solve observed failure modes.

---

# What the demo script intentionally does not solve

This is a conceptual prototype, not a production recommender.

It does not yet address:

- learning calibrated preference priors from real user behavior;
- missing hotel features;
- categorical or free-text feature extraction;
- nonlinear utility interactions;
- contradictory answers over long periods;
- hotel availability, price changes, or booking constraints;
- multiple travelers with competing preferences;
- profile/state discovery across many trips;
- LLM generation/evaluation of natural-language questions;
- causal claims about why a user selected a hotel.

The point of the prototype is to keep the decision machinery visible enough that each of those can be added deliberately rather than absorbed into an opaque recommender.

---

# Core takeaway

The system has three separate kinds of structure:

```text
HOTEL WORLD
What combinations of attributes exist?

USER UNCERTAINTY
What preference directions remain unresolved?

CURRENT DECISION
Which of those unresolved directions actually separate A and B?
```

Candidate questions come from the intersection of those spaces. Expected information gain chooses the question most likely to reduce uncertainty in the current decision. Soft Bayesian-style updates let the system learn without becoming overconfident about the user.

That produces a recommender that can be both decisive and corrigible: it can say what it currently believes, invite correction, ask a small number of high-value questions, and then eventually stop analyzing and ask for the decision.
