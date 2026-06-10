# NumPy — Performance & Optimization

> 💡 **Why this matters:**  
> Writing correct NumPy code is step one. Writing *fast* NumPy code is step two. This chapter is about the habits that separate beginner NumPy from production-grade data science code — the same principles apply when your arrays go from 1K rows to 10M rows.

---

## Table of Contents
1. [Vectorization](#1-vectorization)
2. [Avoid Python Loops](#2-avoid-python-loops)
3. [Memory Efficiency](#3-memory-efficiency)
4. [Timing with %timeit](#4-timing-with-timeit)

---

## 1. Vectorization

Vectorization means applying an operation to an **entire array at once** instead of element by element. NumPy executes these in pre-compiled C code — no Python overhead per element.

### Concept

```python
import numpy as np

arr = np.arange(1_000_000)

# Not vectorized — Python touches each element
result = [x ** 2 for x in arr]

# Vectorized — C handles all elements in one go
result = arr ** 2
```

### Vectorized vs loop — same task, different approach

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Loop approach
result = np.empty(len(a))
for i in range(len(a)):
    result[i] = a[i] + b[i]

# Vectorized
result = a + b
```

### Vectorizing conditions — replace if/else loops

```python
arr = np.array([-3, 5, -1, 8, -2, 4])

# Loop approach
result = []
for x in arr:
    result.append(x if x > 0 else 0)

# Vectorized with np.where
result = np.where(arr > 0, arr, 0)
```

### Vectorizing custom logic — `np.vectorize()`

When you have a Python function that can't be rewritten as array operations directly:

```python
def grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    else: return 'F'

scores = np.array([95, 82, 74, 61, 88])

# Without vectorize — would need a loop
v_grade = np.vectorize(grade)
v_grade(scores)    # ['A' 'B' 'C' 'F' 'B']
```

> ⚠️ `np.vectorize()` is a convenience wrapper — it still loops internally. It's cleaner than writing the loop yourself but **not faster**. Use it only when true vectorization isn't possible.

---

## 2. Avoid Python Loops

Python loops over NumPy arrays are the single biggest performance mistake. Each iteration carries full Python overhead — object lookup, type checking, memory allocation.

### Rule: if you're looping over array elements, there's almost always a NumPy way.

#### Sum — don't loop
```python
arr = np.arange(1_000_000)

# ❌ Slow
total = 0
for x in arr:
    total += x

# ✅ Fast
total = np.sum(arr)
```

#### Conditional count — don't loop
```python
arr = np.random.randint(0, 100, size=1_000_000)

# ❌ Slow
count = sum(1 for x in arr if x > 50)

# ✅ Fast
count = np.sum(arr > 50)
```

#### Row-wise operations — don't loop over rows
```python
data = np.random.rand(10_000, 5)

# ❌ Slow — looping over rows
row_means = []
for row in data:
    row_means.append(np.mean(row))

# ✅ Fast — axis operation
row_means = np.mean(data, axis=1)
```

#### Cumulative operations — don't loop
```python
arr = np.array([1, 2, 3, 4, 5])

# ❌ Slow
running = []
total = 0
for x in arr:
    total += x
    running.append(total)

# ✅ Fast
running = np.cumsum(arr)
```

### When a loop is unavoidable

Sometimes true sequential dependency exists (each step depends on the previous). In that case:
- Use **Numba** (`@jit`) to JIT-compile the loop
- Use **Cython** for compiled extensions
- Restructure logic to remove the dependency if possible

---

## 3. Memory Efficiency

NumPy arrays are already far more memory-efficient than Python lists, but you can go further with intentional dtype choices and avoiding unnecessary copies.

### Choose the right dtype

```python
arr_default = np.arange(1_000_000)           # int64 — 8 bytes each
arr_small   = np.arange(1_000_000, dtype=np.int32)  # int32 — 4 bytes each

print(arr_default.nbytes)    # 8_000_000 bytes  (8 MB)
print(arr_small.nbytes)      # 4_000_000 bytes  (4 MB)  ← 50% less
```

```python
# float64 vs float32
a = np.random.rand(1_000_000)                        # float64 — 8 MB
b = np.random.rand(1_000_000).astype(np.float32)     # float32 — 4 MB
```

> 💡 In deep learning, `float32` is the standard — GPUs are optimized for it and it uses half the memory of `float64` with negligible precision loss for most tasks.

### Common dtype sizes

| dtype | Bytes | Max value |
|---|---|---|
| `int8` | 1 | 127 |
| `int16` | 2 | 32,767 |
| `int32` | 4 | ~2.1 billion |
| `int64` | 8 | ~9.2 quintillion |
| `float32` | 4 | ~3.4 × 10³⁸ |
| `float64` | 8 | ~1.8 × 10³⁰⁸ |
| `bool` | 1 | True/False |

### Avoid unnecessary copies

```python
arr = np.arange(1_000_000)

# ❌ Creates a copy — doubles memory usage
processed = arr * 2

# ✅ In-place operation — no extra allocation
arr *= 2
```

```python
# ❌ astype always creates a copy
arr_f = arr.astype(np.float32)

# ✅ copy=False avoids copy if already correct dtype
arr_f = arr.astype(np.float32, copy=False)
```

### Use views instead of copies where safe

```python
arr = np.arange(12).reshape(3, 4)

# View — no memory cost
col = arr[:, 0]       # shares memory with arr

# Copy — full memory cost
col = arr[:, 0].copy()  # use only when you need independence
```

### Check memory usage

```python
arr = np.random.rand(1000, 1000)

print(arr.dtype)       # float64
print(arr.nbytes)      # 8_000_000 bytes = ~8 MB
print(arr.itemsize)    # 8 bytes per element
```

---

## 4. Timing with `%timeit`

`%timeit` is an IPython/Jupyter magic command that runs a statement many times and reports the average execution time. Essential for comparing approaches.

### Basic usage

```python
arr = np.arange(1_000_000)

%timeit sum(arr)           # Python built-in sum → ~80ms
%timeit np.sum(arr)        # NumPy sum           → ~1ms
```

### Comparing loop vs vectorized

```python
arr = np.random.rand(100_000)

%timeit [x**2 for x in arr]    # list comprehension
%timeit arr**2                  # vectorized
```

### Multi-line with `%%timeit`

```python
%%timeit
result = []
for x in arr:
    result.append(x * 2)
```

### `%timeit` options

```python
%timeit -n 100 -r 5 np.sum(arr)
#          ↑       ↑
#     100 runs  5 repeats — reports best of 5 averages
```

### In plain Python scripts — use `time` module

```python
import time

start = time.time()
result = arr ** 2
end = time.time()

print(f"Elapsed: {(end - start) * 1000:.3f} ms")
```

### Typical speedups you'll see

| Operation | Python loop | NumPy | Speedup |
|---|---|---|---|
| Sum 1M elements | ~80ms | ~1ms | ~80x |
| Square 1M elements | ~200ms | ~3ms | ~65x |
| Boolean filter 1M | ~150ms | ~2ms | ~75x |

> 💡 Speedups vary by machine, array size, and operation. Always measure on your own data — don't assume.

---

## Summary

```
Performance & Optimization
 ├── Vectorization
 │    ├── Operate on whole arrays, not elements
 │    ├── np.where()       → vectorized if/else
 │    └── np.vectorize()   → wraps Python fn (cleaner, not faster)
 ├── Avoid Python Loops
 │    ├── Use np.sum, np.mean, np.cumsum over manual loops
 │    ├── Use axis= for row/col operations
 │    └── True sequential deps → consider Numba/Cython
 ├── Memory Efficiency
 │    ├── dtype choice     → int32/float32 saves 50% vs 64-bit
 │    ├── In-place ops     → arr *= 2 instead of arr = arr * 2
 │    ├── Views over copies → slice don't copy unless needed
 │    └── astype(copy=False) → skip copy if dtype already matches
 └── Timing
      ├── %timeit          → Jupyter/IPython, auto-repeats
      ├── %%timeit         → multi-line timing
      └── time.time()      → plain Python scripts
```