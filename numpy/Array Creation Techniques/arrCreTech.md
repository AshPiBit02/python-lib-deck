# NumPy — Array Creation Techniques

---

## Table of Contents
1. [zeros(), ones(), empty()](#1-zeros-ones-empty)
2. [arange(), linspace()](#2-arange-linspace)
3. [identity(), eye()](#3-identity-eye)
4. [Random Arrays — np.random](#4-random-arrays--nprandom)

---

## 1. `zeros()`, `ones()`, `empty()`

### `np.zeros()`
Creates an array filled with **0.0** (float64 by default).

```python
import numpy as np

np.zeros(5)             # 1D → [0. 0. 0. 0. 0.]
np.zeros((3, 4))        # 2D → 3 rows, 4 cols, all 0.0
np.zeros((2, 3), dtype=int)  # integer 0s instead of float
```

### `np.ones()`
Creates an array filled with **1.0** (float64 by default).

```python
np.ones(4)              # [1. 1. 1. 1.]
np.ones((2, 3))         # 2D → 2 rows, 3 cols, all 1.0
np.ones((2, 3), dtype=int)   # integer 1s
```

### `np.empty()`
Allocates memory **without initializing** values — faster than `zeros()` but values are garbage (whatever was in memory).

```python
np.empty(5)             # something like [1.e-323, 0., 4.e-323, ...]
np.empty((2, 3))        # 2D uninitialized
```

> ⚠️ **Only use `empty()` when you're going to fill every element immediately after.** Never read from it before writing — values are unpredictable.

### Quick Comparison

| Function | Values | Use when |
|---|---|---|
| `zeros()` | All 0s | Default/safe initialization |
| `ones()` | All 1s | Multiplicative initialization |
| `empty()` | Garbage | You'll overwrite all values immediately |

---

## 2. `arange()`, `linspace()`

Both generate a sequence of numbers — but in different ways.

### `np.arange()`
Works like Python's `range()` — define **start, stop, step**.

```python
np.arange(5)            # [0 1 2 3 4]          (stop only)
np.arange(1, 6)         # [1 2 3 4 5]          (start, stop)
np.arange(1, 10, 2)     # [1 3 5 7 9]          (start, stop, step)
np.arange(0, 1, 0.2)    # [0.  0.2 0.4 0.6 0.8] (float step)
```

> ⚠️ Stop value is **exclusive**. `arange(1, 6)` gives `1` to `5`, not `6`.  
> ⚠️ Avoid float steps with `arange` — floating point rounding can give unexpected element counts. Use `linspace` instead.

### `np.linspace()`
Define **start, stop, number of points** — stop is **inclusive**.

```python
np.linspace(0, 1, 5)       # [0.   0.25 0.5  0.75 1.  ]
np.linspace(0, 10, 3)      # [0.   5.   10. ]
np.linspace(1, 100, 4)     # [1.  34.  67. 100.]

# Exclude the stop value
np.linspace(0, 1, 5, endpoint=False)  # [0.  0.2 0.4 0.6 0.8]
```

### `arange` vs `linspace` — When to use which

| | `arange` | `linspace` |
|---|---|---|
| **Control** | Step size | Number of points |
| **Stop** | Exclusive | Inclusive |
| **Float steps** | Unreliable | Reliable |
| **Use case** | Integer sequences, known step | Evenly spaced points, plotting, sampling |

```python
# Example: same range, different approach
np.arange(0, 1, 0.25)     # [0.   0.25 0.5  0.75]     → stop excluded
np.linspace(0, 1, 5)      # [0.   0.25 0.5  0.75 1.0]  → stop included
```

---

## 3. `identity()`, `eye()`

Both create matrices with **1s on the diagonal and 0s elsewhere** — used in linear algebra.

### `np.identity()`
Always returns a **square** matrix (N × N).

```python
np.identity(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

np.identity(3, dtype=int)
# [[1 0 0]
#  [0 1 0]
#  [0 0 1]]
```

### `np.eye()`
More flexible — supports **non-square** shapes and diagonal **offset**.

```python
np.eye(3)           # same as identity(3) — square, main diagonal
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

np.eye(3, 4)        # non-square: 3 rows, 4 cols
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]]

np.eye(3, k=1)      # diagonal shifted 1 above main
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 0.]]

np.eye(3, k=-1)     # diagonal shifted 1 below main
# [[0. 0. 0.]
#  [1. 0. 0.]
#  [0. 1. 0.]]
```

### `identity` vs `eye`

| | `identity(n)` | `eye(n, m, k)` |
|---|---|---|
| **Shape** | Square only | Square or rectangular |
| **Diagonal offset** | Not supported | Supported via `k` |
| **Use case** | Standard identity matrix | Flexible diagonal placement |

> 💡 If you just need a standard identity matrix, `identity()` is cleaner. Use `eye()` when you need offset diagonals or non-square shapes.

---

## 4. Random Arrays — `np.random`

### `np.random.rand()`
Uniform distribution — floats in **[0.0, 1.0)**.

```python
np.random.rand(4)         # 1D → [0.42 0.71 0.03 0.88]
np.random.rand(2, 3)      # 2D → 2 rows, 3 cols
```

### `np.random.randn()`
**Standard normal distribution** — mean=0, std=1. Values can be negative.

```python
np.random.randn(4)        # [ 0.47 -1.23  0.09  1.84]
np.random.randn(2, 3)     # 2D normally distributed values
```

### `np.random.randint()`
Random **integers** in a given range.

```python
np.random.randint(10)            # single int: 0 to 9
np.random.randint(1, 10)         # single int: 1 to 9
np.random.randint(1, 10, size=5)        # 1D array
np.random.randint(1, 10, size=(2, 4))   # 2D array
```

> ⚠️ Stop is **exclusive** — `randint(1, 10)` gives 1 through 9, not 10.

### `np.random.uniform()`
Uniform distribution over a **custom range** (not just [0,1)).

```python
np.random.uniform(5, 10, size=4)      # floats between 5.0 and 10.0
np.random.uniform(5, 10, size=(2, 3)) # 2D
```

### `np.random.choice()`
Randomly sample from an **existing array**.

```python
arr = np.array([10, 20, 30, 40, 50])

np.random.choice(arr)              # single random pick
np.random.choice(arr, size=3)      # 3 picks with replacement
np.random.choice(arr, size=3, replace=False)  # 3 picks without replacement
```

### `np.random.seed()`
Set a seed for **reproducibility** — same seed = same random values every run.

```python
np.random.seed(42)
np.random.rand(3)   # [0.374 0.951 0.732] — same every time with seed 42
```

> 💡 Always set a seed when sharing code or running experiments you want to reproduce.

### Quick Reference

| Function | Output | Range/Distribution |
|---|---|---|
| `rand(shape)` | floats | Uniform [0.0, 1.0) |
| `randn(shape)` | floats | Normal (mean=0, std=1) |
| `randint(low, high, size)` | integers | Uniform [low, high) |
| `uniform(low, high, size)` | floats | Uniform [low, high) |
| `choice(arr, size)` | elements | Sampled from array |
| `seed(n)` | — | Sets reproducibility |

---

## Summary

```
Array Creation
 ├── zeros() / ones() / empty()   → initialized or uninitialized fixed-value arrays
 ├── arange()                     → sequence by step size (stop exclusive)
 ├── linspace()                   → sequence by point count (stop inclusive)
 ├── identity()                   → square identity matrix
 ├── eye()                        → identity with shape/offset flexibility
 └── np.random
      ├── rand()       → uniform floats [0, 1)
      ├── randn()      → normal distribution
      ├── randint()    → random integers
      ├── uniform()    → uniform floats in custom range
      ├── choice()     → sample from array
      └── seed()       → reproducibility
```