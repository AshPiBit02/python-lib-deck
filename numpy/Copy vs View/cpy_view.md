# NumPy — Copy vs View

---

## Table of Contents
1. [Direct Assignment](#1-direct-assignment)
2. [View (Shallow Copy)](#2-view-shallow-copy)
3. [Deep Copy](#3-deep-copy)
4. [Comparison](#4-comparison)

---

## 1. Direct Assignment

Assigning an array to a new variable does **not** create a new array — both variables point to the **same object** in memory.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = a                  # b is NOT a copy — it's the same array

b[0] = 99
print(a)               # [99  2  3  4  5]  ← a is modified too

print(a is b)          # True — same object
```

> ⚠️ Direct assignment is just a **label** pointing to the same data. Modifying one modifies both.

---

## 2. View (Shallow Copy)

A **view** shares the same underlying data — different object, same memory block. Modifying the view modifies the original.

Views are created by:
- Slicing
- `reshape()`
- `transpose()` / `.T`
- `ravel()`

```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]             # slice → view

b[0] = 99
print(a)               # [ 1 99  3  4  5]  ← original modified

print(b.base is a)     # True — b's data lives in a
```

### Check if an array is a view

```python
print(b.base)          # returns the original array if b is a view
print(b.base is None)  # False → b is a view
print(a.base is None)  # True  → a owns its data
```

### reshape and transpose also return views

```python
a = np.arange(6)
b = a.reshape(2, 3)

b[0, 0] = 99
print(a)               # [99  1  2  3  4  5]  ← original modified
```

---

## 3. Deep Copy

`copy()` creates a **completely independent** array — separate object, separate memory. Changes to the copy do **not** affect the original.

```python
a = np.array([1, 2, 3, 4, 5])
b = a.copy()           # deep copy

b[0] = 99
print(a)               # [1 2 3 4 5]  ← original untouched
print(b)               # [99  2  3  4  5]

print(b.base is None)  # True — b owns its own data
print(b is a)          # False — different objects
```

### Copy after slicing

```python
a = np.array([10, 20, 30, 40, 50])

b = a[1:4].copy()      # independent copy of slice
b[0] = 999

print(a)               # [10 20 30 40 50]  ← unaffected
```

### Copy after reshape

```python
a = np.arange(6)
b = a.reshape(2, 3).copy()

b[0, 0] = 99
print(a)               # [0 1 2 3 4 5]  ← unaffected
```

---

## 4. Comparison

| | Direct Assignment | View (Shallow Copy) | Deep Copy |
|---|---|---|---|
| **How** | `b = a` | slice, reshape, `.T`, `ravel()` | `a.copy()` |
| **Same object** | ✅ Yes | ❌ No | ❌ No |
| **Shared memory** | ✅ Yes | ✅ Yes | ❌ No |
| **Modifying b affects a** | ✅ Yes | ✅ Yes | ❌ No |
| **`b.base is None`** | ✅ True | ❌ False | ✅ True |
| **Use when** | Just aliasing | Read-only operations | Need independent data |

### Quick check — `np.shares_memory()`

```python
a = np.array([1, 2, 3, 4, 5])

b = a             # direct assignment
c = a[1:3]        # view
d = a.copy()      # deep copy

np.shares_memory(a, b)    # True
np.shares_memory(a, c)    # True
np.shares_memory(a, d)    # False
```

---

## Summary

```
Copy vs View
 ├── Direct assignment  → same object, same memory (b = a)
 ├── View               → different object, shared memory
 │    └── Created by: slicing, reshape, transpose, ravel
 └── Deep copy          → different object, independent memory (a.copy())
      └── Safe to modify without affecting the original

 Checks:
  b.base is None        → True means owns data, False means view
  np.shares_memory(a,b) → True means same underlying data
```