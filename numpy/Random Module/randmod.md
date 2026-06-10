# NumPy — Random Module

> 💡 **Why this matters:**  
> Random number generation is essential for simulations, data augmentation, train/test splits, weight initialization in neural networks, and statistical sampling. NumPy's `np.random` module is the standard tool for all of these.

---

## Table of Contents
1. [Random Numbers](#1-random-numbers)
2. [rand()](#2-rand)
3. [randn()](#3-randn)
4. [randint()](#4-randint)
5. [choice()](#5-choice)
6. [shuffle() & permutation()](#6-shuffle--permutation)
7. [Seed — Reproducibility](#7-seed--reproducibility)
8. [Probability Distributions](#8-probability-distributions)
9. [Random Generator (Modern API)](#9-random-generator-modern-api)

---

## 1. Random Numbers

NumPy's random module lives at `np.random`. It generates pseudo-random numbers — deterministic sequences that *appear* random, driven by an internal state.

```python
import numpy as np

np.random.random()        # single float in [0.0, 1.0)
np.random.random(5)       # 1D array of 5 floats
np.random.random((3, 4))  # 2D array, shape (3, 4)
```

---

## 2. `rand()`

Uniform distribution — floats in **[0.0, 1.0)**.  
Takes shape as **separate arguments**, not a tuple.

```python
np.random.rand()          # single float
np.random.rand(5)         # 1D → [0.37 0.95 0.73 0.59 0.15]
np.random.rand(2, 3)      # 2D, shape (2, 3)
np.random.rand(2, 3, 4)   # 3D, shape (2, 3, 4)
```

> ⚠️ `rand(3, 4)` vs `random((3, 4))` — same result, different syntax. `rand` takes args directly, `random` takes a tuple.

---

## 3. `randn()`

**Standard Normal distribution** — mean=0, std=1. Values range roughly -3 to +3.

```python
np.random.randn()         # single value
np.random.randn(5)        # 1D
np.random.randn(3, 3)     # 2D
```

### Shift to custom mean and std

```python
mean, std = 50, 10
custom = mean + std * np.random.randn(1000)
# values centered around 50 with spread of 10
```

> 💡 **ML context:** Neural network weights are often initialized using `randn` scaled by a small factor (e.g. `0.01 * np.random.randn(n_in, n_out)`) to break symmetry while keeping values small.

---

## 4. `randint()`

Random **integers** in a given range — stop is **exclusive**.

```python
np.random.randint(10)              # single int: 0 to 9
np.random.randint(1, 10)           # single int: 1 to 9
np.random.randint(1, 100, size=6)         # 1D
np.random.randint(0, 2, size=(4, 4))      # 2D binary matrix (0s and 1s)
```

---

## 5. `choice()`

Randomly sample from a **1D array or range**.

```python
arr = np.array([10, 20, 30, 40, 50])

np.random.choice(arr)                        # single random pick
np.random.choice(arr, size=3)                # 3 picks with replacement
np.random.choice(arr, size=3, replace=False) # 3 picks without replacement
np.random.choice(5)                          # random int from range(5)
```

### Weighted sampling

```python
items = ['A', 'B', 'C', 'D']
probs = [0.5, 0.3, 0.1, 0.1]   # must sum to 1.0

np.random.choice(items, size=10, p=probs)
# 'A' appears ~50% of the time
```

> 💡 **ML context:** Weighted `choice` is used in techniques like bootstrapping, importance sampling, and experience replay in reinforcement learning.

---

## 6. `shuffle()` & `permutation()`

Both randomize order — key difference is **in-place vs new array**.

### `shuffle()` — modifies in-place

```python
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print(arr)    # [3 1 5 2 4]  ← original modified, returns None
```

### `permutation()` — returns a new array

```python
arr = np.array([1, 2, 3, 4, 5])
shuffled = np.random.permutation(arr)

print(arr)       # [1 2 3 4 5]  ← original untouched
print(shuffled)  # [3 1 5 2 4]

# Also works on integer — returns shuffled range
np.random.permutation(6)    # [4 0 2 5 1 3]
```

### shuffle vs permutation

| | `shuffle()` | `permutation()` |
|---|---|---|
| **In-place** | ✅ Yes | ❌ No |
| **Returns** | None | New shuffled array |
| **Original safe** | ❌ No | ✅ Yes |

> 💡 **ML context:** Shuffling the dataset before each training epoch is standard practice to prevent the model from learning order-dependent patterns.

---

## 7. Seed — Reproducibility

Sets the internal state of the random number generator — same seed always produces same sequence.

```python
np.random.seed(42)
print(np.random.rand(3))   # [0.374 0.951 0.732]

np.random.seed(42)
print(np.random.rand(3))   # [0.374 0.951 0.732]  ← identical
```

> 💡 Always set a seed when sharing experiments, writing tests, or debugging — makes results reproducible across runs and machines.

---

## 8. Probability Distributions

Beyond uniform and normal, `np.random` provides many statistical distributions.

### `uniform()` — Uniform over custom range
```python
np.random.uniform(5, 10, size=5)    # floats between 5.0 and 10.0
```

### `normal()` — Normal with custom mean & std
```python
np.random.normal(loc=70, scale=10, size=100)
#                mean=70, std=10
```

### `binomial()` — Number of successes in n trials
```python
np.random.binomial(n=10, p=0.5, size=5)
# e.g. [5 4 6 3 7]  → flipping 10 coins, 5 times
```

### `poisson()` — Events in a fixed interval
```python
np.random.poisson(lam=3, size=5)
# e.g. [2 4 3 1 5]  → avg 3 events per interval
```

### `exponential()` — Time between events
```python
np.random.exponential(scale=2.0, size=5)
# scale = mean wait time
```

### `beta()` — Values between 0 and 1
```python
np.random.beta(a=2, b=5, size=5)
# skewed distribution between 0 and 1
```

### Distribution Quick Reference

| Function | Range | Use case |
|---|---|---|
| `uniform(low, high)` | [low, high) | Equal probability across range |
| `normal(loc, scale)` | (-∞, +∞) | Heights, test scores, noise |
| `binomial(n, p)` | [0, n] | Coin flips, success counts |
| `poisson(lam)` | [0, ∞) | Event counts (emails/hr, clicks) |
| `exponential(scale)` | [0, ∞) | Wait times between events |
| `beta(a, b)` | [0, 1] | Probabilities, proportions |

---

## 9. Random Generator (Modern API)

NumPy 1.17+ introduced `np.random.default_rng()` — the **recommended modern approach**. More reproducible and thread-safe than the legacy `np.random.*` functions.

```python
rng = np.random.default_rng(seed=42)

rng.random(5)               # uniform floats [0, 1)
rng.integers(1, 10, size=5) # random integers (note: integers not randint)
rng.standard_normal(5)      # standard normal
rng.choice([10,20,30,40], size=3, replace=False)
rng.shuffle(arr)            # in-place shuffle
```

> 💡 For new code, prefer `default_rng()`. For legacy code and tutorials, `np.random.*` still works fine. Both are valid — just be consistent within a project.

---

## Summary

```
Random Module
 ├── rand()          → uniform floats [0, 1), args as separate values
 ├── randn()         → standard normal (mean=0, std=1)
 ├── randint()       → random integers, stop exclusive
 ├── choice()        → sample from array, supports weights & replace
 ├── shuffle()       → in-place reorder (returns None)
 ├── permutation()   → reorder into new array (original safe)
 ├── seed()          → fix random state for reproducibility
 ├── Distributions
 │    ├── uniform()      → custom range
 │    ├── normal()       → custom mean & std
 │    ├── binomial()     → success counts
 │    ├── poisson()      → event counts
 │    ├── exponential()  → wait times
 │    └── beta()         → probabilities [0,1]
 └── default_rng()   → modern API (recommended for new code)
```