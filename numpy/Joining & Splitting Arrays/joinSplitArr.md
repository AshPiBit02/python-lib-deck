# NumPy — Joining & Splitting Arrays

---

## Table of Contents
1. [concatenate()](#1-concatenate)
2. [vstack()](#2-vstack)
3. [hstack()](#3-hstack)
4. [split()](#4-split)
5. [vsplit()](#5-vsplit)
6. [hsplit()](#6-hsplit)

---

## 1. `concatenate()`

Joins a sequence of arrays **along an existing axis**.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.concatenate([a, b])          # [1 2 3 4 5 6]  → default axis=0
```

### 2D — along axis=0 (row-wise)

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

np.concatenate([a, b], axis=0)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]    → shape (4, 2)
```

### 2D — along axis=1 (column-wise)

```python
np.concatenate([a, b], axis=1)
# [[1 2 5 6]
#  [3 4 7 8]]   → shape (2, 4)
```

> ⚠️ Arrays must have the **same shape** on all axes except the one being concatenated.

```python
a = np.array([[1, 2], [3, 4]])     # shape (2, 2)
b = np.array([[5, 6, 7]])          # shape (1, 3)

np.concatenate([a, b], axis=0)     # ❌ ValueError — cols don't match
```

---

## 2. `vstack()`

Stacks arrays **vertically** — row on top of row (axis=0).  
Shorthand for `concatenate` along axis=0, but also handles 1D arrays gracefully.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.vstack([a, b])
# [[1 2 3]
#  [4 5 6]]    → shape (2, 3)  ← 1D arrays treated as rows
```

### 2D

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

np.vstack([a, b])
# [[1 2]
#  [3 4]
#  [5 6]]    → shape (3, 2)
```

> ⚠️ Arrays must have the **same number of columns**.

---

## 3. `hstack()`

Stacks arrays **horizontally** — column beside column (axis=1).  
For 1D arrays, it simply concatenates end to end.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.hstack([a, b])       # [1 2 3 4 5 6]  → 1D just concatenates
```

### 2D

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5], [6]])

np.hstack([a, b])
# [[1 2 5]
#  [3 4 6]]    → shape (2, 3)
```

> ⚠️ Arrays must have the **same number of rows**.

### concatenate vs vstack vs hstack

| | `concatenate` | `vstack` | `hstack` |
|---|---|---|---|
| **Axis** | Any (specify with `axis=`) | Always axis=0 | Always axis=1 (axis=0 for 1D) |
| **1D handling** | Concatenates flat | Treats as row → 2D | Concatenates flat |
| **Flexibility** | Most flexible | Row stacking | Column stacking |

---

## 4. `split()`

Splits an array into **multiple sub-arrays** along an axis.

### Split into equal parts

```python
arr = np.array([1, 2, 3, 4, 5, 6])

np.split(arr, 3)        # [array([1,2]), array([3,4]), array([5,6])]
np.split(arr, 2)        # [array([1,2,3]), array([4,5,6])]
```

> ⚠️ If the array can't be split equally, raises a `ValueError`.

### Split at specific indices

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

np.split(arr, [2, 5])
# [array([1, 2]),
#  array([3, 4, 5]),
#  array([6, 7, 8])]
#         ↑     ↑
#     split at index 2 and 5
```

### 2D split

```python
arr = np.arange(12).reshape(4, 3)

np.split(arr, 2, axis=0)   # split into 2 along rows → two (2,3) arrays
np.split(arr, 3, axis=1)   # split into 3 along cols → three (4,1) arrays
```

---

## 5. `vsplit()`

Splits along **axis=0** (row-wise). Shorthand for `split(arr, n, axis=0)`.  
Only works on 2D+ arrays.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10,11,12]])   # shape (4, 3)

np.vsplit(arr, 2)
# [array([[1, 2, 3],
#         [4, 5, 6]]),
#  array([[ 7,  8,  9],
#         [10, 11, 12]])]

np.vsplit(arr, 4)              # each row as its own array
```

### Split at specific rows

```python
np.vsplit(arr, [1, 3])
# row 0        → array([[1, 2, 3]])
# rows 1,2     → array([[4,5,6],[7,8,9]])
# row 3        → array([[10, 11, 12]])
```

---

## 6. `hsplit()`

Splits along **axis=1** (column-wise). Shorthand for `split(arr, n, axis=1)`.

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]])   # shape (2, 4)

np.hsplit(arr, 2)
# [array([[1, 2],        array([[3, 4],
#         [5, 6]]),              [7, 8]])]

np.hsplit(arr, 4)               # each column as its own (2,1) array
```

### Split at specific columns

```python
np.hsplit(arr, [1, 3])
# col 0        → shape (2, 1)
# cols 1,2     → shape (2, 2)
# col 3        → shape (2, 1)
```

### split vs vsplit vs hsplit

| | `split` | `vsplit` | `hsplit` |
|---|---|---|---|
| **Axis** | Any (specify `axis=`) | Always axis=0 | Always axis=1 |
| **Works on 1D** | Yes | No | Yes |
| **Flexibility** | Most flexible | Row splitting | Column splitting |

---

## Summary

```
Joining & Splitting
 ├── Joining
 │    ├── concatenate()  → join along any axis (most flexible)
 │    ├── vstack()       → stack row-wise (axis=0), handles 1D as rows
 │    └── hstack()       → stack column-wise (axis=1), 1D concatenates flat
 └── Splitting
      ├── split()        → split into n parts or at indices, any axis
      ├── vsplit()       → split row-wise (axis=0), 2D+ only
      └── hsplit()       → split column-wise (axis=1)
```