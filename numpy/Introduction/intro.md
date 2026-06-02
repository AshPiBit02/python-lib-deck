# NumPy — Introduction

> **Series:** Python Data Science | **File:** 01 — Introduction  
> **Prerequisites:** Python basics, Pandas (done ✅)

---

## Table of Contents
1. [What is NumPy?](#1-what-is-numpy)
2. [Why NumPy is Faster than Python Lists](#2-why-numpy-is-faster-than-python-lists)
3. [Installing NumPy](#3-installing-numpy)
4. [Importing NumPy](#4-importing-numpy)

---

## 1. What is NumPy?

**NumPy** (Numerical Python) is a core Python library for **numerical and scientific computing**.  
It provides a powerful **N-dimensional array object (`ndarray`)** along with a large collection of mathematical functions to operate on arrays efficiently.

### Key capabilities:
- Multi-dimensional arrays and matrices (`ndarray`)
- Mathematical operations: linear algebra, statistics, Fourier transforms
- Broadcasting — performing operations on arrays of different shapes
- Integration with libraries like **Pandas**, **Matplotlib**, **Scikit-learn**, **TensorFlow**

> 💡 **Coming from Pandas?**  
> Pandas is *built on top of NumPy*. The underlying data in a DataFrame is a NumPy array. Understanding NumPy gives you better control over Pandas internals.

---

## 2. Why NumPy is Faster than Python Lists?

This is one of the most important concepts to understand early.

### Python List vs NumPy Array

| Factor | Python List | NumPy Array (`ndarray`) |
|---|---|---|
| **Type** | Can hold mixed types | Fixed type (homogeneous) |
| **Memory** | Each element is a Python object (high overhead) | Raw C-level data stored contiguously |
| **Speed** | Slower (loop-based operations) | Faster (vectorized operations in C) |
| **Functionality** | General purpose | Math/science optimized |

### Under the Hood — Why it's Faster:

**1. Contiguous Memory Storage**  
NumPy stores all elements in a single continuous block of memory. Python lists store *pointers* to scattered objects across memory — causing more cache misses.

**2. Fixed Data Type**  
A NumPy array of `int32` stores 4 bytes per element. A Python list element is a full Python object (~28 bytes). Less memory = faster access.

**3. Vectorized Operations (No Python Loops)**  
NumPy operations are executed in pre-compiled C code internally. You write clean Python syntax, but the heavy lifting runs at C speed.

```python
import numpy as np

# Python list approach — uses a Python loop
py_list = list(range(1_000_000))
result = [x * 2 for x in py_list]   # slow loop

# NumPy approach — vectorized, no explicit loop
np_array = np.arange(1_000_000)
result = np_array * 2                # runs in C, very fast
```

**4. SIMD / CPU-Level Optimizations**  
NumPy is linked to optimized libraries (BLAS, LAPACK) that leverage CPU-level instructions (SIMD) for operations like matrix multiplication.

> ⚡ **Benchmark insight:** NumPy can be **10x–100x faster** than equivalent Python list operations depending on the task and array size.

---

## 3. Installing NumPy

NumPy comes pre-installed in most data science environments (Anaconda, Google Colab, Kaggle).

**Check if already installed:**
```bash
python -c "import numpy; print(numpy.__version__)"
```

**Install via pip (if needed):**
```bash
pip install numpy
```

**Install via conda:**
```bash
conda install numpy
```

> 📌 **Note:** If you already have Pandas installed, NumPy is almost certainly installed too — Pandas lists NumPy as a direct dependency.

---

## 4. Importing NumPy

The universal convention is to import NumPy as `np`:

```python
import numpy as np
```

### Why `np`?
- It is the **standard alias** used across the entire data science ecosystem
- Every tutorial, documentation page, and library uses `np`
- Keeps code concise without sacrificing readability

```python
import numpy as np

# Creating a simple array
arr = np.array([1, 2, 3, 4, 5])
print(arr)         # [1 2 3 4 5]
print(type(arr))   # <class 'numpy.ndarray'>
```

> ⚠️ **Avoid** doing `from numpy import *` — it pollutes the namespace and makes code harder to debug.

---

## Quick Summary

```
NumPy
 ├── Core object: ndarray (N-dimensional array)
 ├── Faster than lists: contiguous memory + C-level operations + fixed dtype
 ├── Install: pip install numpy
 └── Import: import numpy as np
```

---