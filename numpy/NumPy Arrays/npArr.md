# NumPy — Arrays

---

## Table of Contents
1. [The `ndarray` Object](#1-the-ndarray-object)
2. [Creating Arrays](#2-creating-arrays)
3. [1D, 2D, and N-Dimensional Arrays](#3-1d-2d-and-n-dimensional-arrays)
4. [Array Attributes](#4-array-attributes)

---

## 1. The `ndarray` Object

**`ndarray`** (N-dimensional array) is NumPy's core data structure.

```python
import numpy as np

arr = np.array([10, 20, 30])
print(type(arr))   # <class 'numpy.ndarray'>
```

| Property | Description |
|---|---|
| **Homogeneous** | All elements must be the same data type |
| **Fixed size** | Size is set at creation; resizing creates a new array |
| **N-dimensional** | Can represent vectors, matrices, or higher-rank tensors |
| **Zero-indexed** | Indexing starts at 0 |

> 💡 **From Pandas:** `df.values` or `df.to_numpy()` returns an `ndarray` — Pandas DataFrames are built on top of it.

---

## 2. Creating Arrays

### From Python Lists / Tuples

```python
arr = np.array([1, 2, 3, 4, 5])        # from list
arr = np.array((1, 2, 3))              # from tuple
arr = np.array([1, 2, 3], dtype=float) # explicit dtype → [1. 2. 3.]
```

> ⚠️ Pass elements as a **list**, not separate arguments.
> ```python
> np.array(1, 2, 3)    # ❌ TypeError
> np.array([1, 2, 3])  # ✅
> ```

### Built-in Creation Functions

```python
np.zeros(5)                        # [0. 0. 0. 0. 0.]
np.ones((2, 3))                    # 2x3 array of 1.0s
np.full((2, 3), 7)                 # 2x3 array filled with 7
np.arange(1, 10, 2)                # [1 3 5 7 9]  (start, stop, step)
np.linspace(0, 1, 5)               # [0. 0.25 0.5 0.75 1.]  (start, stop, n_points)
np.eye(3)                          # 3x3 identity matrix
np.random.randint(1, 10, size=(2, 3))  # random integers
np.random.rand(3, 3)               # random floats in [0, 1)
```

> 💡 `arange` uses a **step size**. `linspace` uses a **number of points** — use it when you need exact endpoint control.

---

## 3. 1D, 2D, and N-Dimensional Arrays

### 1D Array (Vector)

```python
arr = np.array([10, 20, 30, 40, 50])

print(arr)        # [10 20 30 40 50]
print(arr.ndim)   # 1
print(arr.shape)  # (5,)
```

### 2D Array (Matrix)

```python
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(arr)
# [[1 2 3]
#  [4 5 6]]
print(arr.ndim)   # 2
print(arr.shape)  # (2, 3) → 2 rows, 3 columns
```

### 3D Array (Tensor)

```python
arr = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print(arr.ndim)   # 3
print(arr.shape)  # (2, 2, 2) → 2 blocks, each 2x2
```

> 💡 **Real-world analogy:**
> - 1D → a single row of values
> - 2D → a table / grayscale image (height × width)
> - 3D → a color image (height × width × RGB channels)

### N-Dimensional

NumPy supports any number of dimensions. 4D+ arrays are common in deep learning (e.g., batch of images: `(batch, height, width, channels)`).

```python
arr = np.zeros((2, 3, 4, 5))
print(arr.ndim)   # 4
print(arr.shape)  # (2, 3, 4, 5)
```

---

## 4. Array Attributes

### `ndim` — Number of Dimensions
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.ndim)   # 2
```

### `shape` — Size Along Each Dimension
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3) → 2 rows, 3 columns

rows, cols = arr.shape  # unpack if needed
```
> 💡 For 1D arrays, shape is `(n,)` — the trailing comma is intentional, it's a single-element tuple.

### `dtype` — Data Type of Elements
```python
print(np.array([1, 2, 3]).dtype)      # int64
print(np.array([1.0, 2.0]).dtype)     # float64
print(np.array([True, False]).dtype)  # bool
```

NumPy **upcasts** automatically when types conflict:
```python
np.array([1, 2, 3.0]).dtype    # float64  (int promoted to float)
```

Cast dtype after creation using `astype()`:
```python
arr = np.array([1, 2, 3])
arr.astype(np.float32)         # returns new array with float32 dtype
```

### `size` — Total Number of Elements
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.size)   # 6  → 2 × 3
```
Always equals the **product of all values in `shape`**.

---

## Summary

```
ndarray
 ├── Created from: lists, tuples, np.zeros/ones/arange/linspace/random...
 ├── Dimensions:  1D (vector) → 2D (matrix) → 3D+ (tensor)
 └── Key Attributes:
      ├── ndim   → number of dimensions
      ├── shape  → tuple of sizes per dimension
      ├── dtype  → element type (int64, float64, bool...)
      └── size   → total element count (product of shape)
```

---