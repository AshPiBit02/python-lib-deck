# NumPy — Indexing and Slicing

---

## Table of Contents
1. [Indexing — 1D Arrays](#1-indexing--1d-arrays)
2. [Indexing — 2D Arrays](#2-indexing--2d-arrays)
3. [Slicing Arrays](#3-slicing-arrays)
4. [Boolean Indexing](#4-boolean-indexing)
5. [Fancy Indexing](#5-fancy-indexing)

---

## 1. Indexing — 1D Arrays

Same as Python lists — zero-based, supports negative indexing.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

arr[0]    # 10  → first element
arr[3]    # 40
arr[-1]   # 50  → last element
arr[-2]   # 40  → second from last
```

---

## 2. Indexing — 2D Arrays

Use `[row, col]` syntax — cleaner than Python's nested list `[row][col]`.

```python
arr = np.array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
])

arr[0, 0]    # 1   → row 0, col 0
arr[1, 2]    # 7   → row 1, col 2
arr[2, -1]   # 12  → row 2, last col
arr[-1, -1]  # 12  → last row, last col
```

### Selecting an entire row or column

```python
arr[1]       # [5 6 7 8]     → entire row 1
arr[1, :]    # [5 6 7 8]     → same, explicit
arr[:, 2]    # [3 7 11]      → entire col 2
arr[:, 0]    # [1 5 9]       → entire col 0
```

---

## 3. Slicing Arrays

Syntax: `arr[start:stop:step]` — stop is **exclusive**.

### 1D Slicing

```python
arr = np.array([10, 20, 30, 40, 50])

arr[1:4]     # [20 30 40]   → index 1 to 3
arr[:3]      # [10 20 30]   → start to index 2
arr[2:]      # [30 40 50]   → index 2 to end
arr[::2]     # [10 30 50]   → every 2nd element
arr[::-1]    # [50 40 30 20 10] → reversed
```

### 2D Slicing

```python
arr = np.array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
])

arr[0:2, :]       # rows 0-1, all cols
# [[1 2 3 4]
#  [5 6 7 8]]

arr[:, 1:3]       # all rows, cols 1-2
# [[ 2  3]
#  [ 6  7]
#  [10 11]]

arr[0:2, 1:3]     # rows 0-1, cols 1-2
# [[2 3]
#  [6 7]]
```

### ⚠️ Slices are Views, Not Copies

```python
arr = np.array([10, 20, 30, 40, 50])
sliced = arr[1:4]
sliced[0] = 999

print(arr)     # [ 10 999  30  40  50] ← original is modified!
```

Modifying a slice modifies the original array. Use `.copy()` to avoid this:

```python
sliced = arr[1:4].copy()   # independent copy
```

---

## 4. Boolean Indexing

Filter elements using a **condition** — returns elements where condition is `True`.

### 1D

```python
arr = np.array([10, 25, 3, 47, 18, 6])

arr > 15               # [False  True False  True  True False]
arr[arr > 15]          # [25 47 18]
arr[arr % 2 == 0]      # [10 18 6]  → even numbers only
```

### 2D

```python
arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

arr[arr > 5]           # [6 7 8 9]  → returns 1D of matching elements
```

### Multiple Conditions

Use `&` (and), `|` (or), `~` (not) — **not** Python's `and`/`or`.

```python
arr = np.array([10, 25, 3, 47, 18, 6])

arr[(arr > 10) & (arr < 40)]    # [25 18]
arr[(arr < 5)  | (arr > 40)]    # [3 47]
arr[~(arr > 15)]                # [10  3  6]  → negation
```

> ⚠️ Use `&`, `|`, `~` — not `and`, `or`, `not`. Python's logical operators don't work element-wise on arrays.

### Modify values using Boolean Indexing

```python
arr = np.array([10, 25, 3, 47, 18])
arr[arr < 10] = 0       # set all values below 10 to 0
print(arr)              # [10 25  0 47 18]
```

---

## 5. Fancy Indexing

Select elements using an **array of indices** — returns a copy, not a view.

### 1D

```python
arr = np.array([10, 20, 30, 40, 50])

arr[[0, 2, 4]]      # [10 30 50]  → elements at index 0, 2, 4
arr[[3, 1, 3]]      # [40 20 40]  → repeated index allowed
arr[[-1, -3]]       # [50 30]     → negative indices work too
```

### 2D — Select specific rows

```python
arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

arr[[0, 2]]          # rows 0 and 2
# [[1 2 3]
#  [7 8 9]]

arr[[2, 0, 1]]       # reorder rows
# [[7 8 9]
#  [1 2 3]
#  [4 5 6]]
```

### 2D — Select specific row-column pairs

```python
arr[[0, 1, 2], [0, 1, 2]]   # [1 5 9]  → diagonal elements
arr[[0, 2], [1, 2]]         # [2 9]    → (row0,col1) and (row2,col2)
```

### Fancy Indexing vs Slicing

| | Slicing | Fancy Indexing |
|---|---|---|
| **Syntax** | `arr[1:4]` | `arr[[1, 2, 3]]` |
| **Returns** | View (shares memory) | Copy (independent) |
| **Order** | Sequential only | Any order, repeats allowed |

---

## Summary

```
Indexing & Slicing
 ├── Indexing 1D     → arr[i], arr[-i]
 ├── Indexing 2D     → arr[row, col], arr[row, :], arr[:, col]
 ├── Slicing         → arr[start:stop:step], returns a VIEW
 │    └── .copy()    → use when you don't want to affect original
 ├── Boolean         → arr[arr > 5], supports &, |, ~
 │    └── Can modify → arr[arr < 0] = 0
 └── Fancy           → arr[[0,2,4]], returns a COPY, any order/repeats
```