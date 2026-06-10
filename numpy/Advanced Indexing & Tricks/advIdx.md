# NumPy — Advanced Indexing & Tricks

---

## Table of Contents
1. [Advanced Slicing](#1-advanced-slicing)
2. [argsort()](#2-argsort)
3. [Advanced where()](#3-advanced-where)
4. [Advanced argmax() & argmin()](#4-advanced-argmax--argmin)
5. [nonzero()](#5-nonzero)
6. [take() & put()](#6-take--put)
7. [diag(), triu(), tril()](#7-diag-triu-tril)
8. [np.clip() with Broadcasting](#8-npclip-with-broadcasting)
9. [Useful Tricks](#9-useful-tricks)

---

## 1. Advanced Slicing

### Ellipsis `...`

`...` (Ellipsis) means *"fill in however many `:` are needed here"*. Useful when working with arrays of unknown or high dimensions.

```python
import numpy as np

arr = np.arange(24).reshape(2, 3, 4)   # shape (2, 3, 4)

arr[0, :, :]     # first block, all rows, all cols
arr[0, ...]      # same — ellipsis fills the remaining axes

arr[..., 0]      # all blocks, all rows, first col   → shape (2, 3)
arr[1, ..., 2]   # second block, all rows, third col → shape (3,)
```

> 💡 `...` is especially useful in functions that must handle arrays of any number of dimensions without hardcoding `:, :, :`.

### `np.newaxis` — Add a Dimension

Inserts a new axis of size 1 — used to reshape for broadcasting without calling `reshape()`.

```python
arr = np.array([1, 2, 3, 4, 5])   # shape (5,)

arr[:, np.newaxis]    # shape (5, 1) — column vector
arr[np.newaxis, :]    # shape (1, 5) — row vector

# Practical use — broadcasting two 1D arrays into a 2D outer product
a = np.array([1, 2, 3])        # (3,)
b = np.array([10, 20, 30])     # (3,)

a[:, np.newaxis] + b           # (3,1) + (3,) → (3, 3)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

> 💡 `np.newaxis` is just `None` under the hood — `arr[:, None]` is identical.

### Step slicing on 2D

```python
arr = np.arange(25).reshape(5, 5)

arr[::2, ::2]     # every other row AND col
# [[ 0  2  4]
#  [10 12 14]
#  [20 22 24]]

arr[::-1, :]      # reverse row order
arr[:, ::-1]      # reverse column order
arr[::-1, ::-1]   # full 180° flip
```

---

## 2. `argsort()`

Returns the **indices** that would sort the array — the actual values stay unchanged.

```python
arr = np.array([40, 10, 30, 20, 50])

np.argsort(arr)       # [1 3 2 0 4]  → index 1 (10) is smallest, index 4 (50) is largest

arr[np.argsort(arr)]  # [10 20 30 40 50]  → sorted values via fancy indexing
```

### Descending order

```python
np.argsort(arr)[::-1]          # [4 0 2 3 1]  → largest to smallest indices
arr[np.argsort(arr)[::-1]]     # [50 40 30 20 10]
```

### On 2D — along an axis

```python
arr = np.array([[3, 1, 4],
                [1, 5, 2],
                [9, 6, 7]])

np.argsort(arr, axis=1)    # sort indices within each row
# [[1 0 2]   → row 0: col1(1) < col0(3) < col2(4)
#  [0 2 1]   → row 1: col0(1) < col2(2) < col1(5)
#  [1 2 0]]  → row 2: col1(6) < col2(7) < col0(9)

np.argsort(arr, axis=0)    # sort indices within each column
```

### Practical use — rank items

```python
scores = np.array([88, 95, 72, 61, 90])
names  = np.array(['Alice', 'Bob', 'Carol', 'Dave', 'Eve'])

ranked = names[np.argsort(scores)[::-1]]
print(ranked)   # ['Bob' 'Eve' 'Alice' 'Carol' 'Dave']  → ranked highest to lowest
```

---

## 3. Advanced `where()`

### Chain conditions on different columns

```python
data = np.array([[1, 85, 40000],
                 [2, 72, 55000],
                 [3, 91, 32000],
                 [4, 68, 61000],
                 [5, 88, 47000]])
# cols: id, score, salary

# Replace salary with 0 where score < 75
result = np.where(data[:, 1] < 75, 0, data[:, 2])
print(result)   # [40000     0 32000 0 47000]
```

### Nested `where()` — multiple conditions

```python
scores = np.array([95, 82, 74, 55, 88, 60])

grades = np.where(scores >= 90, 'A',
         np.where(scores >= 80, 'B',
         np.where(scores >= 70, 'C', 'F')))

print(grades)   # ['A' 'B' 'C' 'F' 'B' 'F']
```

### `where()` to get indices only

```python
arr = np.array([10, -5, 30, -1, 20, -8])

indices = np.where(arr < 0)       # (array([1, 3, 5]),)
print(indices[0])                  # [1 3 5]
print(arr[arr < 0])                # [-5 -1 -8]  → values at those indices
```

### 2D — returns row and column indices

```python
arr = np.array([[1, -2, 3],
                [-4, 5, -6],
                [7, -8, 9]])

rows, cols = np.where(arr < 0)
print(rows)    # [0 1 1 2]
print(cols)    # [1 0 2 1]

# Zip them for coordinate pairs
list(zip(rows, cols))   # [(0,1), (1,0), (1,2), (2,1)]
```

---

## 4. Advanced `argmax()` & `argmin()`

### Per-row and per-column

```python
arr = np.array([[3, 7, 1],
                [9, 2, 8],
                [4, 6, 5]])

np.argmax(arr, axis=0)   # [1 0 1]  → row index of max per column
np.argmax(arr, axis=1)   # [1 2 1]  → col index of max per row
np.argmin(arr, axis=1)   # [2 1 0]  → col index of min per row
```

### Combine with `argsort` for top-k

```python
scores = np.array([55, 88, 72, 95, 61, 90, 78])

# Top 3 scores — indices
top3_idx = np.argsort(scores)[-3:]        # [5 3 1] ... bottom-up
top3_idx = np.argsort(scores)[::-1][:3]  # [3 5 1] ... top-down (cleaner)

print(scores[top3_idx])   # [95 90 88]
```

### `np.argwhere()` — returns indices as rows

```python
arr = np.array([[1, 0, 3],
                [0, 5, 0],
                [7, 0, 9]])

np.argwhere(arr == 0)
# [[0 1]   → (row0, col1)
#  [1 0]   → (row1, col0)
#  [1 2]   → (row1, col2)
#  [2 1]]  → (row2, col1)
```

> 💡 `np.argwhere(cond)` is cleaner than `zip(*np.where(cond))` when you want coordinate pairs.

---

## 5. `nonzero()`

Returns indices of all **non-zero elements** — equivalent to `np.where(arr != 0)`.

```python
arr = np.array([0, 3, 0, 0, 7, 0, 2])

np.nonzero(arr)          # (array([1, 4, 6]),)
arr[np.nonzero(arr)]     # [3 7 2]  → non-zero values
```

### 2D

```python
arr = np.array([[0, 1, 0],
                [2, 0, 3],
                [0, 4, 0]])

rows, cols = np.nonzero(arr)
print(rows)    # [0 1 1 2]
print(cols)    # [1 0 2 1]
print(arr[rows, cols])   # [1 2 3 4]  → all non-zero values
```

> 💡 Useful for sparse data — when most values are 0 and you only want to work on the non-zero entries.

---

## 6. `take()` & `put()`

### `np.take()` — fancy indexing with axis control

Equivalent to fancy indexing but works cleanly along a specified axis.

```python
arr = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]

np.take(arr, indices)    # [10 30 50]  → same as arr[[0,2,4]]
```

### `take` along axis on 2D

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

np.take(arr, [0, 2], axis=0)   # rows 0 and 2
# [[1 2 3]
#  [7 8 9]]

np.take(arr, [0, 2], axis=1)   # cols 0 and 2
# [[1 3]
#  [4 6]
#  [7 9]]
```

### `np.put()` — place values at indices in-place

```python
arr = np.array([10, 20, 30, 40, 50])

np.put(arr, [1, 3], [99, 88])
print(arr)   # [10 99 30 88 50]
```

> 💡 `put()` always works on the **flattened** array — useful for placing values at specific flat positions in any shape.

```python
arr = np.zeros((3, 3))
np.put(arr, [0, 4, 8], 1)    # set diagonal via flat indices
print(arr)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

---

## 7. `diag()`, `triu()`, `tril()`

### `np.diag()` — extract diagonal or create diagonal matrix

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

np.diag(arr)        # [1 5 9]  → extract main diagonal
np.diag(arr, k=1)   # [2 6]   → diagonal above main
np.diag(arr, k=-1)  # [4 8]   → diagonal below main

# Create diagonal matrix from 1D array
np.diag([1, 2, 3])
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```

### `np.triu()` — upper triangle

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

np.triu(arr)       # keep upper triangle, zero out below
# [[1 2 3]
#  [0 5 6]
#  [0 0 9]]

np.triu(arr, k=1)  # exclude main diagonal too
# [[0 2 3]
#  [0 0 6]
#  [0 0 0]]
```

### `np.tril()` — lower triangle

```python
np.tril(arr)       # keep lower triangle, zero out above
# [[1 0 0]
#  [4 5 0]
#  [7 8 9]]

np.tril(arr, k=-1) # exclude main diagonal
# [[0 0 0]
#  [4 0 0]
#  [7 8 0]]
```

> 💡 `triu`/`tril` are used heavily in attention masks (transformers), covariance matrix manipulation, and solving triangular linear systems.

---

## 8. `np.clip()` with Broadcasting

Beyond simple scalar bounds, `clip` can use arrays as bounds — different limit per element.

```python
arr = np.array([5, 15, 25, 35, 45])
lower = np.array([0, 10, 20, 30, 40])
upper = np.array([10, 20, 30, 40, 50])

np.clip(arr, lower, upper)   # [5 15 25 35 45]  → all within bounds
```

```python
# Clip each row by different bounds
data  = np.array([[1, 50, 100],
                  [5, 80,  20]])
lower = np.array([[0, 40,  90]])   # shape (1, 3) → broadcasts
upper = np.array([[10, 60, 110]])

np.clip(data, lower, upper)
# [[ 1 50 100]
#  [ 5 60  90]]
```

---

## 9. Useful Tricks

### `np.unique()` — unique values and counts

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])

np.unique(arr)                          # [1 2 3 4 5 6 9]
np.unique(arr, return_counts=True)      # (values, counts)
# (array([1,2,3,4,5,6,9]), array([2,1,2,1,2,1,1]))

np.unique(arr, return_index=True)       # indices of first occurrence
np.unique(arr, return_inverse=True)     # how to reconstruct original from unique
```

### `np.searchsorted()` — binary search on sorted array

```python
arr = np.array([10, 20, 30, 40, 50])

np.searchsorted(arr, 25)    # 2  → insert 25 at index 2 to keep sorted
np.searchsorted(arr, 30)    # 2  → left-side default
np.searchsorted(arr, 30, side='right')  # 3  → right side
```

> 💡 Used internally by many algorithms. Useful for bucketing/binning values into ranges.

### `np.tile()` & `np.repeat()`

```python
arr = np.array([1, 2, 3])

np.repeat(arr, 3)          # [1 1 1 2 2 2 3 3 3]  → repeat each element
np.tile(arr, 3)            # [1 2 3 1 2 3 1 2 3]  → repeat whole array

np.repeat(arr, [1, 2, 3])  # [1 2 2 3 3 3]  → repeat each by different count
```

### `np.pad()` — pad an array

```python
arr = np.array([1, 2, 3])

np.pad(arr, pad_width=2, mode='constant', constant_values=0)
# [0 0 1 2 3 0 0]

# 2D padding
arr2d = np.ones((3, 3))
np.pad(arr2d, pad_width=1, mode='constant', constant_values=0)
# pads with 0s on all sides → shape (5, 5)
```

> 💡 `pad` is used in CNNs (convolutional neural networks) to add zero-padding around images before convolution.

### `np.squeeze()` & `np.expand_dims()`

```python
arr = np.array([[[1, 2, 3]]])   # shape (1, 1, 3)

np.squeeze(arr)                  # [1 2 3]  → shape (3,), removes size-1 dims
np.squeeze(arr, axis=0)          # shape (1, 3) → remove only axis 0

arr2 = np.array([1, 2, 3])       # shape (3,)
np.expand_dims(arr2, axis=0)     # shape (1, 3)
np.expand_dims(arr2, axis=1)     # shape (3, 1)
```

> 💡 `squeeze` and `expand_dims` are used constantly when feeding data into ML frameworks that expect specific shapes like `(batch, features)` or `(batch, channels, H, W)`.

---

## Summary

```
Advanced Indexing & Tricks
 ├── Advanced Slicing
 │    ├── ...  (ellipsis)    → fill remaining axes automatically
 │    └── np.newaxis / None  → insert size-1 axis for broadcasting
 ├── argsort()     → indices that sort, use for ranking & top-k
 ├── where()       → nested conditions, 2D coordinate extraction
 ├── argmax/min    → per-axis, combine with argsort for top-k
 ├── argwhere()    → clean coordinate pairs for non-zero/condition
 ├── nonzero()     → indices of non-zero elements
 ├── take()        → fancy indexing with axis control
 ├── put()         → place values at flat indices in-place
 ├── diag()        → extract/create diagonal
 ├── triu()/tril() → upper/lower triangle masks
 └── Tricks
      ├── unique()        → unique values + counts
      ├── searchsorted()  → binary search / binning
      ├── repeat()/tile() → element vs array repetition
      ├── pad()           → add border values (CNN padding)
      └── squeeze()/expand_dims() → shape manipulation for ML
```