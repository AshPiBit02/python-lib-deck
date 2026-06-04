# NumPy — Mathematical Functions

---

## Table of Contents
1. [Axis Operations](#1-axis-operations)
2. [np.where()](#2-npwhere)
3. [np.clip()](#3-npclip)
4. [np.percentile() & np.quantile()](#4-nppercentile--npquantile)

---

## 1. Axis Operations

`axis=0` → operate **down rows** (result per column)  
`axis=1` → operate **across columns** (result per row)

```
arr = [[1, 2, 3],
       [4, 5, 6]]

axis=0 ↓ (collapse rows)     axis=1 → (collapse cols)
  [5, 7, 9]                    [6, 15]
```

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# No axis — operates on all elements
np.sum(arr)            # 21
np.mean(arr)           # 3.5
np.min(arr)            # 1
np.max(arr)            # 6

# axis=0 — collapse rows, result per column
np.sum(arr, axis=0)    # [5 7 9]
np.mean(arr, axis=0)   # [2.5 3.5 4.5]
np.min(arr, axis=0)    # [1 2 3]
np.max(arr, axis=0)    # [4 5 6]

# axis=1 — collapse columns, result per row
np.sum(arr, axis=1)    # [ 6 15]
np.mean(arr, axis=1)   # [2. 5.]
np.min(arr, axis=1)    # [1 4]
np.max(arr, axis=1)    # [3 6]
```

### `keepdims=True`

By default, the reduced dimension is dropped. `keepdims=True` keeps it as size 1 — useful when you need the result to broadcast back against the original.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

np.sum(arr, axis=1)                  # [6 15]         shape (2,)
np.sum(arr, axis=1, keepdims=True)   # [[6], [15]]    shape (2, 1)

# Practical use — normalize each row
arr / np.sum(arr, axis=1, keepdims=True)
# [[0.167 0.333 0.5  ]
#  [0.267 0.333 0.4  ]]
```

> ⚠️ Without `keepdims=True`, broadcasting back against the original fails due to shape mismatch.

### Axis on 3D Arrays

```python
arr = np.zeros((2, 3, 4))

np.sum(arr, axis=0).shape   # (3, 4) → collapsed first dim
np.sum(arr, axis=1).shape   # (2, 4) → collapsed second dim
np.sum(arr, axis=2).shape   # (2, 3) → collapsed third dim
```

### `np.argmin()` / `np.argmax()` — Index of Min/Max

```python
arr = np.array([[3, 1, 4],
                [1, 5, 9]])

np.argmax(arr)           # 5  → flat index of max element (9)
np.argmax(arr, axis=0)   # [0 1 1] → row index of max per column
np.argmax(arr, axis=1)   # [2 2]   → col index of max per row

np.argmin(arr, axis=1)   # [1 0]   → col index of min per row
```

---

## 2. `np.where()`

Returns elements from one array or another based on a **condition**.

### Syntax
```python
np.where(condition, value_if_true, value_if_false)
```

### Basic Usage

```python
arr = np.array([10, -5, 30, -1, 20])

np.where(arr > 0, arr, 0)        # [10  0 30  0 20]  → negatives → 0
np.where(arr > 0, 1, -1)         # [ 1 -1  1 -1  1]  → sign array
np.where(arr > 0, arr, arr * -1) # [10  5 30  1 20]  → abs value
```

### On 2D Arrays

```python
arr = np.array([[1, -2, 3],
                [-4, 5, -6]])

np.where(arr > 0, arr, 0)
# [[1 0 3]
#  [0 5 0]]
```

### Get Indices Where Condition is True

```python
arr = np.array([10, -5, 30, -1, 20])

np.where(arr > 0)        # (array([0, 2, 4]),)  → indices of positive values
```

> 💡 `np.where(condition)` with no extra args behaves like `np.nonzero()` — returns indices where condition is `True`.

---

## 3. `np.clip()`

**Clamps** all values in an array within a `[min, max]` range. Values outside are clipped to the boundary.

### Syntax
```python
np.clip(arr, min, max)
```

### Usage

```python
arr = np.array([1, 5, 12, -3, 8, 20])

np.clip(arr, 0, 10)      # [ 1  5 10  0  8 10]
#                                 ↑       ↑
#                           12→10       -3→0

np.clip(arr, 5, 15)      # [ 5  5 12  5  8 15]
```

### On 2D Arrays

```python
arr = np.array([[1, 200, 50],
                [-10, 75, 300]])

np.clip(arr, 0, 100)
# [[  1 100  50]
#  [  0  75 100]]
```

### Clip only one side

```python
np.clip(arr, 0, None)    # clip only min (no upper limit)
np.clip(arr, None, 100)  # clip only max (no lower limit)
```

> 💡 Common use case: keeping values in a valid range — pixel values (0–255), probabilities (0–1), or ratings (1–5).

---

## 4. `np.percentile()` & `np.quantile()`

Both measure the **spread** of data. Nearly identical — difference is input scale.

| | `percentile` | `quantile` |
|---|---|---|
| **Input scale** | 0 to 100 | 0.0 to 1.0 |
| **50th percentile** | `q=50` | `q=0.5` |

### `np.percentile()`

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

np.percentile(arr, 50)       # 55.0  → median
np.percentile(arr, 25)       # 32.5  → Q1
np.percentile(arr, 75)       # 77.5  → Q3
np.percentile(arr, [25, 50, 75])  # [32.5 55.  77.5]  → multiple at once
```

### `np.quantile()`

```python
np.quantile(arr, 0.5)        # 55.0  → same as percentile 50
np.quantile(arr, 0.25)       # 32.5
np.quantile(arr, [0.25, 0.5, 0.75])  # [32.5 55.  77.5]
```

### Along an Axis

```python
arr = np.array([[10, 20, 30],
                [40, 50, 60]])

np.percentile(arr, 50, axis=0)   # [25. 35. 45.]  → median per column
np.percentile(arr, 50, axis=1)   # [20. 50.]       → median per row
```

---

## Summary

```
Mathematical Functions
 ├── Axis Operations
 │    ├── axis=0      → collapse rows (result per column)
 │    ├── axis=1      → collapse cols (result per row)
 │    ├── keepdims    → preserve shape for broadcasting
 │    └── argmin/max  → index of min/max element
 ├── np.where()       → conditional element selection
 ├── np.clip()        → clamp values within [min, max]
 └── np.percentile()  → spread (0–100 scale)
     np.quantile()    → spread (0.0–1.0 scale)
```