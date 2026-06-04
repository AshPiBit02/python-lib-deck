# NumPy — Array Operations

---

## Table of Contents
1. [Arithmetic Operations](#1-arithmetic-operations)
2. [Broadcasting](#2-broadcasting)
3. [Element-wise Operations](#3-element-wise-operations)

---

## 1. Arithmetic Operations

NumPy applies arithmetic **element-wise** by default. No loops needed.

### Scalar Operations
A scalar is applied to **every element**.

```python
import numpy as np

arr = np.array([10, 20, 30, 40])

arr + 5     # [15 25 35 45]
arr - 5     # [ 5 15 25 35]
arr * 2     # [20 40 60 80]
arr / 2     # [ 5. 10. 15. 20.]
arr ** 2    # [ 100  400  900 1600]
arr % 3     # [1 2 0 1]
arr // 3    # [3 6 10 13]
```

### Array with Array
Both arrays must have the **same shape** (or be broadcastable — covered next).

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b     # [5 7 9]
a - b     # [-3 -3 -3]
a * b     # [ 4 10 18]
a / b     # [0.25 0.4  0.5 ]
a ** b    # [  1  32 216]
```

### 2D Array Arithmetic

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A + B
# [[ 6  8]
#  [10 12]]

A * B        # element-wise (NOT matrix multiplication)
# [[ 5 12]
#  [21 32]]

A @ B        # matrix multiplication
# [[19 22]
#  [43 50]]
```

> 💡 `*` is element-wise. Use `@` or `np.dot()` for matrix multiplication.

---

## 2. Broadcasting

Broadcasting allows NumPy to perform operations on arrays of **different shapes** by virtually stretching the smaller array to match the larger one — no actual copying of data.

### Rule
Arrays are compatible for broadcasting if, for each dimension, the sizes are either **equal** or one of them is **1**.

### Case 1 — Scalar with Array
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])   # shape (2, 3)

arr + 10
# [[11 12 13]
#  [14 15 16]]
# scalar shape () → broadcast across all elements
```

### Case 2 — 1D Array with 2D Array
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])   # shape (2, 3)

b = np.array([10, 20, 30])    # shape (3,)

arr + b
# [[11 22 33]
#  [14 25 36]]
# b is broadcast across each row
```

```
Visual:
 [[ 1  2  3]       [10  20  30]       [[11  22  33]
  [ 4  5  6]]  +   [10  20  30]  =     [14  25  36]]
               ↑
        b stretched to match rows
```

### Case 3 — Column Vector with Row Vector
```python
col = np.array([[1], [2], [3]])   # shape (3, 1)
row = np.array([10, 20, 30])      # shape (1, 3) → (3,)

col + row
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
# produces a (3, 3) output
```

### Incompatible Shapes — Error
```python
a = np.array([1, 2, 3])     # shape (3,)
b = np.array([1, 2])        # shape (2,)

a + b   # ❌ ValueError: operands could not be broadcast
```

> 💡 Dimensions are compared **right to left**. Each pair must be equal or one must be `1`.

---

## 3. Element-wise Operations

These are NumPy **universal functions (ufuncs)** — optimized C-level functions applied to every element.

### Math Functions

```python
arr = np.array([1, 4, 9, 16, 25])

np.sqrt(arr)      # [1. 2. 3. 4. 5.]
np.square(arr)    # [  1  16  81 256 625]
np.abs(arr)       # works on negative values too
np.log(arr)       # natural log
np.log2(arr)      # log base 2
np.log10(arr)     # log base 10
np.exp(arr)       # e^x for each element
```

### Trigonometric Functions

```python
arr = np.array([0, np.pi/2, np.pi])

np.sin(arr)       # [0.  1.  0.]  (approx)
np.cos(arr)       # [1.  0. -1.]  (approx)
np.tan(arr)       # [0.  large  0.]
np.deg2rad(180)   # π  → convert degrees to radians
np.rad2deg(np.pi) # 180.0 → convert radians to degrees
```

### Rounding

```python
arr = np.array([1.2, 2.567, 3.999, -1.5])

np.round(arr)         # [ 1.  3.  4. -2.]   → nearest even on .5
np.round(arr, 2)      # [ 1.2   2.57  4.   -1.5 ]
np.floor(arr)         # [ 1.  2.  3. -2.]   → round down
np.ceil(arr)          # [ 2.  3.  4. -1.]   → round up
np.trunc(arr)         # [ 1.  2.  3. -1.]   → drop decimal (toward 0)
```

### Comparison (Element-wise) — Returns Boolean Array

```python
a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 1, 4])

np.equal(a, b)           # [False  True False  True]
np.not_equal(a, b)       # [ True False  True False]
np.greater(a, b)         # [False False  True False]
np.less(a, b)            # [ True False False False]
np.greater_equal(a, b)   # [False  True  True  True]
```

> 💡 These are equivalent to `a == b`, `a > b` etc. but are explicit ufuncs — useful when passing functions as arguments.

### Aggregate Operations

These reduce an array to a single value (or along an axis).

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

np.sum(arr)          # 21       → total sum
np.sum(arr, axis=0)  # [5 7 9]  → sum each column
np.sum(arr, axis=1)  # [6 15]   → sum each row

np.min(arr)          # 1
np.max(arr)          # 6
np.mean(arr)         # 3.5
np.median(arr)       # 3.5
np.std(arr)          # standard deviation
np.var(arr)          # variance
np.cumsum(arr)       # [ 1  3  6 10 15 21]  → running total
np.prod(arr)         # 720  → product of all elements
```

> 💡 `axis=0` operates **down rows** (per column). `axis=1` operates **across columns** (per row).

---

## Summary

```
Array Operations
 ├── Arithmetic        → +, -, *, /, **, %, // applied element-wise
 │    └── @ or np.dot  → matrix multiplication (not *)
 ├── Broadcasting      → operate on different shapes without copying
 │    └── Rule         → dims must be equal or one of them is 1 (right to left)
 └── Element-wise (ufuncs)
      ├── Math         → sqrt, square, abs, log, exp
      ├── Trig         → sin, cos, tan, deg2rad, rad2deg
      ├── Rounding     → round, floor, ceil, trunc
      ├── Comparison   → equal, greater, less, ...
      └── Aggregates   → sum, min, max, mean, std, cumsum (support axis)
```