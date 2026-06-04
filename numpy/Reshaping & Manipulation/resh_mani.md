# NumPy — Reshaping and Manipulation

---

## Table of Contents
1. [reshape()](#1-reshape)
2. [flatten()](#2-flatten)
3. [ravel()](#3-ravel)
4. [flatten() vs ravel()](#4-flatten-vs-ravel)
5. [transpose()](#5-transpose)
6. [resize()](#6-resize)

---

## 1. `reshape()`

Changes the **shape** of an array without changing its data. Total number of elements must remain the same.

```python
import numpy as np

arr = np.arange(12)        # [ 0  1  2  3  4  5  6  7  8  9 10 11]

arr.reshape(3, 4)          # 3 rows, 4 cols
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

arr.reshape(2, 6)          # 2 rows, 6 cols
arr.reshape(2, 2, 3)       # 3D → 2 blocks, 2 rows, 3 cols
```

### Using `-1` — Let NumPy Infer a Dimension

```python
arr = np.arange(12)

arr.reshape(3, -1)    # NumPy infers cols → (3, 4)
arr.reshape(-1, 4)    # NumPy infers rows → (3, 4)
arr.reshape(2, 2, -1) # NumPy infers last dim → (2, 2, 3)
```

> ⚠️ Only **one** dimension can be `-1` at a time.

### Invalid Reshape

```python
arr.reshape(5, 3)    # ❌ ValueError: 12 elements can't fill (5, 3) = 15
```

### reshape() returns a View

```python
arr = np.arange(6)
reshaped = arr.reshape(2, 3)
reshaped[0, 0] = 99

print(arr)    # [99  1  2  3  4  5]  ← original modified
```

> Use `.reshape(...).copy()` if you need an independent array.

---

## 2. `flatten()`

Collapses any array into a **1D array**. Always returns a **copy**.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

arr.flatten()          # [1 2 3 4 5 6]
```

### On 3D

```python
arr = np.array([[[1, 2], [3, 4]],
                [[5, 6], [7, 8]]])

arr.flatten()    # [1 2 3 4 5 6 7 8]
```

### Order Parameter

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

arr.flatten(order='C')    # [1 2 3 4 5 6]  → row-major (default)
arr.flatten(order='F')    # [1 4 2 5 3 6]  → column-major (Fortran order)
```

---

## 3. `ravel()`

Also collapses to **1D** — but returns a **view** whenever possible (copy only if needed).

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

arr.ravel()     # [1 2 3 4 5 6]
```

### ravel() is a View

```python
arr = np.array([[1, 2], [3, 4]])
flat = arr.ravel()
flat[0] = 99

print(arr)    # [[99  2]
              #  [ 3  4]]  ← original modified
```

---

## 4. `flatten()` vs `ravel()`

| | `flatten()` | `ravel()` |
|---|---|---|
| **Returns** | Always a copy | View (copy if needed) |
| **Memory** | Higher | Lower |
| **Modifying result** | Safe — original unaffected | Modifies original |
| **Use when** | You need an independent 1D copy | You just need to iterate / read |

---

## 5. `transpose()`

Reverses or permutes the **axes** of an array. For 2D this swaps rows and columns.

### 2D Transpose

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])    # shape (2, 3)

arr.transpose()
# [[1 4]
#  [2 5]
#  [3 6]]                      # shape (3, 2)

arr.T                          # shorthand, same result
```

### 3D — Permuting Axes

```python
arr = np.zeros((2, 3, 4))       # shape (2, 3, 4)

arr.transpose().shape           # (4, 3, 2)  → fully reversed
arr.transpose(0, 2, 1).shape    # (2, 4, 3)  → swap last two axes
arr.transpose(1, 0, 2).shape    # (3, 2, 4)  → swap first two axes
```

> 💡 `arr.transpose(axes)` takes the **new order** of axes as argument.  
> `arr.transpose(0, 2, 1)` means: keep axis 0, then put axis 2, then axis 1.

### transpose() returns a View

```python
arr = np.array([[1, 2], [3, 4]])
t = arr.T
t[0, 0] = 99

print(arr)    # [[99  2]
              #  [ 3  4]]   ← original modified
```

---

## 6. `resize()`

Changes the shape of an array **in-place**. Unlike `reshape()`, it does **not** require the same total elements — it repeats or truncates data to fill the new shape.

### `np.resize()` — returns a new array

```python
arr = np.array([1, 2, 3, 4])

np.resize(arr, (2, 3))
# [[1 2 3]
#  [4 1 2]]   ← repeats data to fill

np.resize(arr, (2, 2))
# [[1 2]
#  [3 4]]     → same element count, no repeat needed
```

### `arr.resize()` — modifies in-place

```python
arr = np.array([1, 2, 3, 4, 5, 6])

arr.resize(2, 3)
print(arr)
# [[1 2 3]
#  [4 5 6]]
```

If new size is **larger**, fills with zeros:

```python
arr = np.array([1, 2, 3])
arr.resize(2, 3)
print(arr)
# [[1 2 3]
#  [0 0 0]]   ← padded with zeros
```

### `reshape()` vs `resize()`

| | `reshape()` | `resize()` |
|---|---|---|
| **Element count** | Must stay the same | Can change |
| **Extra elements** | ❌ Error | Repeats data (`np.resize`) or fills zeros (`arr.resize`) |
| **In-place** | No (returns new/view) | `arr.resize()` modifies in-place |

---

## Summary

```
Reshaping & Manipulation
 ├── reshape()     → change shape (same element count), returns view, use -1 to infer
 ├── flatten()     → collapse to 1D, always a COPY
 ├── ravel()       → collapse to 1D, returns VIEW (prefer for read-only)
 ├── transpose()   → swap/permute axes, returns VIEW (.T shorthand for 2D)
 └── resize()      → change shape freely
      ├── np.resize()   → new array, repeats data if larger
      └── arr.resize()  → in-place, pads zeros if larger
```