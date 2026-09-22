"""
Hotel preference elicitation as a small information-gain / 20 Questions problem.

This is intentionally one sequential script rather than a package. Read it top-to-bottom.

Run:
    python hotel_preference_demo.py
    python hotel_preference_demo.py --interactive

Dependencies:
    numpy
    scikit-learn
"""

import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(7)

# -----------------------------------------------------------------------------
# 0. DATA: a small hotel feature space
# -----------------------------------------------------------------------------
# Every feature is "more of the named thing" and is later standardized.
# User preference weights can be positive or negative.
FEATURES = np.array([
    "quiet",
    "room_size",
    "walkability",
    "nightlife",
    "service",
    "spa",
    "beach_access",
    "value",
])
N_FEATURES = len(FEATURES)

# Build a synthetic hotel population with some real-looking correlation structure.
# In a real system, replace this block with your labelled hotel matrix.
n_hotels = 800
urban = rng.normal(size=n_hotels)
luxury = rng.normal(size=n_hotels)
resort = rng.normal(size=n_hotels)
noise = lambda scale=0.45: rng.normal(scale=scale, size=n_hotels)

hotel_population = np.column_stack([
    -0.65 * urban + 0.20 * resort + noise(),              # quiet
    -0.35 * urban + 0.45 * resort + 0.20 * luxury + noise(),  # room_size
     0.80 * urban - 0.15 * resort + noise(),              # walkability
     0.85 * urban - 0.10 * resort + noise(),              # nightlife
     0.70 * luxury + 0.15 * resort + noise(),             # service
     0.50 * luxury + 0.55 * resort + noise(),             # spa
     0.80 * resort + noise(),                             # beach_access
    -0.45 * luxury + 0.10 * urban + noise(),              # value
])

scaler = StandardScaler()
X = scaler.fit_transform(hotel_population)

# Two hotels the user has already shortlisted. In a real system these would be
# their labelled feature vectors, transformed with the same scaler.
A = np.array([0.25, 0.35, 1.20, 1.35, 0.70, 0.10, -0.30, 0.15])
B = np.array([1.25, 1.10, 0.15, -0.35, 0.85, 0.95, 1.20, -0.05])
delta = A - B


def show_vector(title, v, top_n=None):
    pairs = list(zip(FEATURES, v))
    if top_n is not None:
        pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    print(f"\n{title}")
    for name, value in pairs:
        print(f"  {name:14s} {value:+.3f}")


# -----------------------------------------------------------------------------
# 1. SIMPLEST VERSION: ask about the largest normalized hotel difference
# -----------------------------------------------------------------------------
show_vector("1) Raw A - B standardized feature differences", delta)

j = int(np.argmax(np.abs(delta)))
print(
    f"\nSimplest first question: ask about '{FEATURES[j]}', "
    f"because A and B differ most there ({delta[j]:+.2f} SD)."
)


# -----------------------------------------------------------------------------
# 2. GLOBAL HOTEL PCA: discover interpretable compound axes in hotel space
# -----------------------------------------------------------------------------
hotel_pca = PCA(n_components=5, random_state=0).fit(X)
ZA = hotel_pca.transform(A.reshape(1, -1))[0]
ZB = hotel_pca.transform(B.reshape(1, -1))[0]
pc_delta = ZA - ZB

print("\n2) Global hotel PCA: compound axes that exist in the hotel population")
for k in np.argsort(np.abs(pc_delta))[::-1][:3]:
    loadings = hotel_pca.components_[k]
    top = sorted(zip(FEATURES, loadings), key=lambda x: abs(x[1]), reverse=True)[:4]
    readable = ", ".join(f"{name} {weight:+.2f}" for name, weight in top)
    print(f"  PC{k+1}: pair separation={pc_delta[k]:+.2f} | {readable}")


# -----------------------------------------------------------------------------
# 3. SEED A USER PRIOR: population geometry + weak evidence from choosing A & B
# -----------------------------------------------------------------------------
# Hotel covariance gives us a rough geometry for which features tend to move
# together. This is NOT claimed to be the true preference covariance; it is a
# deliberately weak structural prior.
hotel_corr = np.corrcoef(X, rowvar=False)
prior_cov = 0.65 * np.eye(N_FEATURES) + 0.35 * hotel_corr

# A & B being shortlisted is weak evidence. If both are above/below population
# average on a feature, nudge the prior mean in that direction. Keep it weak.
shared_character = (A + B) / 2.0
SEED_STRENGTH = 0.18
prior_mean = SEED_STRENGTH * shared_character

N_PARTICLES = 20_000
prior_particles = rng.multivariate_normal(prior_mean, prior_cov, size=N_PARTICLES)
particles = prior_particles.copy()
weights = np.full(N_PARTICLES, 1.0 / N_PARTICLES)

show_vector("\n3) Weak prior mean implied by the shortlisted pair", prior_mean)


# -----------------------------------------------------------------------------
# Helper math for the remaining sections
# -----------------------------------------------------------------------------
def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    s = w.sum()
    return np.full_like(w, 1.0 / len(w)) if s <= 0 else w / s


def weighted_mean(points, w):
    return np.average(points, axis=0, weights=w)


def weighted_cov(points, w):
    w = normalize_weights(w)
    mu = weighted_mean(points, w)
    centered = points - mu
    # This denominator is sufficient for our inference/demo purpose.
    return (centered * w[:, None]).T @ centered


def binary_entropy(p):
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def hotel_a_probability(points, w):
    prefers_a = (points @ delta) > 0
    return float(w[prefers_a].sum())


def posterior_axes(points, w, max_axes=5):
    """Principal uncertainty directions of the CURRENT preference posterior."""
    cov = weighted_cov(points, w)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order].T
    return vals[:max_axes], vecs[:max_axes]


def branch_for_axis(points, axis):
    """
    Predict which side of a compound A-vs-B tradeoff each preference particle
    would favor. The sign is aligned to the actual A/B separation on this axis.
    """
    pair_projection = float(delta @ axis)
    if abs(pair_projection) < 1e-10:
        return np.zeros(len(points), dtype=bool)
    return (points @ axis) * pair_projection > 0


def information_gain(points, w, axis):
    """Expected reduction in entropy of the final A-vs-B decision."""
    before = binary_entropy(hotel_a_probability(points, w))
    response = branch_for_axis(points, axis)
    after = 0.0

    for answer in (False, True):
        mask = response == answer
        p_answer = float(w[mask].sum())
        if p_answer <= 1e-12:
            continue
        branch_w = normalize_weights(w[mask])
        p_a = hotel_a_probability(points[mask], branch_w)
        after += p_answer * binary_entropy(p_a)

    return before - after


def axis_question(axis):
    """Turn a compound axis into a rough pair-specific natural-language prompt."""
    # PCA/eigenvector signs are arbitrary. Use |axis * A/B difference| only to
    # decide which features matter to this axis, then use delta itself to say
    # which hotel has more of each feature.
    relevance = np.abs(axis * delta)
    order = np.argsort(relevance)[::-1]

    a_side, b_side = [], []
    for idx in order:
        if relevance[idx] < 0.05:
            continue
        if delta[idx] > 0:
            a_side.append(FEATURES[idx])
        elif delta[idx] < 0:
            b_side.append(FEATURES[idx])
        if len(a_side) >= 3 and len(b_side) >= 3:
            break

    if a_side and b_side:
        a_text = ", ".join(a_side[:3])
        b_text = ", ".join(b_side[:3])
        return f"Would you rather prioritize A's {a_text}, or B's {b_text}?"
    if a_side:
        return f"How important is A's {', '.join(a_side[:3])} advantage relative to B's other strengths?"
    if b_side:
        return f"How important is B's {', '.join(b_side[:3])} advantage relative to A's other strengths?"
    return "Which hotel's overall tradeoffs feel more important to you?"


def candidate_axes(points, w):
    """
    Use posterior PCA axes PLUS raw feature axes. This prevents PCA from hiding
    a single highly-informative feature.
    """
    vals, axes = posterior_axes(points, w, max_axes=5)
    raw_axes = np.eye(N_FEATURES)
    all_axes = np.vstack([axes, raw_axes])
    names = [f"posterior_PC{i+1}" for i in range(len(axes))] + list(FEATURES)
    return vals, all_axes, names


def best_question(points, w, excluded_names=None):
    excluded_names = set() if excluded_names is None else set(excluded_names)
    vals, axes, names = candidate_axes(points, w)
    scored = []
    for name, axis in zip(names, axes):
        if name in excluded_names:
            continue
        separation = abs(float(delta @ axis))
        if separation < 1e-8:
            continue
        ig = information_gain(points, w, axis)
        scored.append((ig, separation, name, axis))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# -----------------------------------------------------------------------------
# 4. USER-POSTERIOR PCA: what do we still not know about this user?
# -----------------------------------------------------------------------------
vals, axes = posterior_axes(particles, weights)
print("\n4) Largest unresolved preference directions")
for i, (variance, axis) in enumerate(zip(vals[:3], axes[:3]), start=1):
    pair_relevance = abs(float(delta @ axis))
    top = sorted(zip(FEATURES, axis), key=lambda x: abs(x[1]), reverse=True)[:4]
    readable = ", ".join(f"{name} {loading:+.2f}" for name, loading in top)
    print(
        f"  Preference PC{i}: uncertainty={variance:.3f}, "
        f"A/B separation={pair_relevance:.3f} | {readable}"
    )


# -----------------------------------------------------------------------------
# 5. INFORMATION GAIN: choose the question that clarifies A vs B most
# -----------------------------------------------------------------------------
ranked = best_question(particles, weights)
print("\n5) Candidate questions ranked by expected information gain")
for ig, separation, name, axis in ranked[:5]:
    print(f"  {name:16s} IG={ig:.3f} bits  pair-separation={separation:.3f}")
    print(f"    {axis_question(axis)}")


# -----------------------------------------------------------------------------
# 6. SOFT UPDATE + TEMPERING + DIFFUSION + PRIOR MIXTURE
# -----------------------------------------------------------------------------
# These are intentionally separate knobs. Start simple; add only what you need.
P_CORRECT = 0.78   # observation/model noise: user may answer inconsistently
ETA = 0.55         # tempered likelihood: <1 means slower learning
DIFFUSION = 0.025  # preference drift / uncertainty floor
PRIOR_MIX = 0.08   # permanent exploratory mass from the original prior


def update_from_answer(points, w, axis, prefers_a_side):
    predicted = branch_for_axis(points, axis)
    observed = bool(prefers_a_side)

    likelihood = np.where(predicted == observed, P_CORRECT, 1.0 - P_CORRECT)
    new_w = normalize_weights(w * (likelihood ** ETA))

    # Diffusion: allow preferences to move instead of collapsing forever.
    if DIFFUSION > 0:
        points = points + rng.normal(0, DIFFUSION, size=points.shape)

    # Prior mixture: always retain some probability mass from broad prior ideas.
    # Implemented as a simple particle refresh to keep the demo inspectable.
    if PRIOR_MIX > 0:
        n_refresh = int(PRIOR_MIX * len(points))
        refresh_idx = rng.choice(len(points), size=n_refresh, replace=False)
        source_idx = rng.choice(len(prior_particles), size=n_refresh, replace=True)
        points[refresh_idx] = prior_particles[source_idx]
        new_w[refresh_idx] = 1.0 / len(points)
        new_w = normalize_weights(new_w)

    return points, new_w


# -----------------------------------------------------------------------------
# 7. AGENT-FACING STATE: recommendation, assumptions, clarification, sale
# -----------------------------------------------------------------------------
def summarize_state(points, w, excluded_names=None):
    p_a = hotel_a_probability(points, w)
    mu = weighted_mean(points, w)
    uncertainty = np.sqrt(np.clip(np.diag(weighted_cov(points, w)), 0, None))

    importance_order = np.argsort(np.abs(mu))[::-1][:3]
    uncertain_order = np.argsort(uncertainty)[::-1][:3]

    likely = [f"{FEATURES[i]} ({mu[i]:+.2f})" for i in importance_order]
    open_questions = [f"{FEATURES[i]} (sd={uncertainty[i]:.2f})" for i in uncertain_order]

    print("\nCurrent agent-facing summary")
    print(f"  P(Hotel A preferred) = {p_a:.3f}")
    print(f"  P(Hotel B preferred) = {1-p_a:.3f}")
    print(f"  Current important assumptions: {', '.join(likely)}")
    print(f"  Still uncertain about: {', '.join(open_questions)}")

    confidence = max(p_a, 1 - p_a)
    recommended = "A" if p_a >= 0.5 else "B"

    if confidence >= 0.82:
        print(f"  Agent: Hotel {recommended} currently looks like the better fit.")
        print(f"  Agent: Do you want to go with Hotel {recommended}?")
    else:
        ranked_now = best_question(points, w, excluded_names=excluded_names)
        if ranked_now:
            print(f"  Agent: Hotel {recommended} is slightly ahead, but one question could help:")
            print(f"         {axis_question(ranked_now[0][3])}")


# -----------------------------------------------------------------------------
# 8. OPTIONAL INTERACTIVE LOOP
# -----------------------------------------------------------------------------
def interactive_loop(points, w, max_questions=5):
    print("\n--- Interactive mode ---")
    print("Answer A, B, or S (skip). This is a stand-in for a chat agent.\n")
    print("Already-asked axes are removed from consideration for the rest of the demo.\n")

    asked_names = set()

    for step in range(max_questions):
        summarize_state(points, w, excluded_names=asked_names)
        p_a = hotel_a_probability(points, w)
        if max(p_a, 1 - p_a) >= 0.90:
            break

        ranked_now = best_question(points, w, excluded_names=asked_names)
        if not ranked_now:
            print("No unused question axes remain.")
            break
        ig, separation, name, axis = ranked_now[0]
        asked_names.add(name)

        print(f"\nQuestion {step+1} [{name}, {ig:.3f} bits]")
        print(axis_question(axis))
        answer = input("Prefer A-side or B-side? [A/B/S]: ").strip().lower()
        if answer == "s":
            continue
        if answer not in {"a", "b"}:
            print("Unrecognized answer; skipping.")
            continue
        points, w = update_from_answer(points, w, axis, prefers_a_side=(answer == "a"))

    summarize_state(points, w, excluded_names=asked_names)
    return points, w

parser = argparse.ArgumentParser()
parser.add_argument("--interactive", action="store_true", help="run a small CLI preference interview")
args = parser.parse_args()

summarize_state(particles, weights)

if args.interactive:
    particles, weights = interactive_loop(particles, weights)
else:
    # Demonstrate one soft update so the default script is fully runnable.
    first_axis = ranked[0][3]
    print("\n6) Demo update: pretend the user chooses the B-side of the first question.")
    particles, weights = update_from_answer(particles, weights, first_axis, prefers_a_side=False)
    summarize_state(particles, weights)

print("\nDone. Read the numbered sections top-to-bottom and stop at any layer you do not need.")
