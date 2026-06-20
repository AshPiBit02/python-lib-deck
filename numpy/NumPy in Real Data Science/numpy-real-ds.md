# NumPy — In Real Data Science

> This chapter bridges raw NumPy to the messy reality of real datasets — the cleanup work that happens **before** a CSV becomes a clean Pandas DataFrame. Scaling formulas are recapped briefly (already covered in ch06/ch15); the focus here is on what's genuinely new: NaN handling, outlier detection, encoding, and the NumPy→Pandas handoff.

---

## Table of Contents
1. [Loading Raw Numerical Data](#1-loading-raw-numerical-data)
2. [Handling Missing Values (NaN)](#2-handling-missing-values-nan)
3. [Outlier Detection](#3-outlier-detection)
4. [Data Type Cleanup](#4-data-type-cleanup)
5. [Encoding Categorical-Looking Data](#5-encoding-categorical-looking-data)
6. [Feature Scaling — Quick Recap](#6-feature-scaling--quick-recap)
7. [NumPy → Pandas Handoff](#7-numpy--pandas-handoff)
8. [End-to-End Preprocessing Pipeline](#8-end-to-end-preprocessing-pipeline)

---

## 1. Loading Raw Numerical Data

Before Pandas, NumPy can load plain numerical files directly.

```python
import numpy as np

# From CSV — basic numeric data
data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

# genfromtxt handles missing values better than loadtxt
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1, filling_values=np.nan)
```

> 💡 `genfromtxt` is the safer default for real-world data — `loadtxt` breaks on missing/malformed entries, `genfromtxt` fills them with `nan` instead of crashing.

### Quick inspection after loading

```python
print(data.shape)
print(data.dtype)
print(np.isnan(data).sum())   # total missing values across the whole array
```

---

## 2. Handling Missing Values (NaN)

Real datasets almost always have gaps. NumPy has dedicated NaN-safe functions — regular functions (`np.mean`, `np.sum`) **propagate** NaN and silently ruin your results.

### The trap

```python
arr = np.array([10, 20, np.nan, 40, 50])

np.mean(arr)        # nan  ← entire result poisoned by one NaN
np.sum(arr)         # nan
```

### The fix — NaN-safe functions

```python
np.nanmean(arr)      # 30.0   → ignores NaN
np.nansum(arr)       # 120.0
np.nanstd(arr)       # std ignoring NaN
np.nanmin(arr)       # 10.0
np.nanmax(arr)       # 50.0
np.nanmedian(arr)    # 30.0
```

### Detecting NaN

```python
np.isnan(arr)              # [False False  True False False]
np.sum(np.isnan(arr))      # 1  → count of NaNs
np.where(np.isnan(arr))    # indices of NaN values
```

### Removing NaN rows (common before Pandas)

```python
data = np.array([
    [1, 2, 3],
    [4, np.nan, 6],
    [7, 8, 9],
    [np.nan, 11, 12]
])

# Mask rows with ANY NaN
clean = data[~np.isnan(data).any(axis=1)]
print(clean)
# [[1. 2. 3.]
#  [7. 8. 9.]]
```

### Imputing (filling) NaN instead of dropping

```python
arr = np.array([10, np.nan, 30, np.nan, 50])

# Fill with column mean
fill_value = np.nanmean(arr)
arr_filled = np.where(np.isnan(arr), fill_value, arr)
print(arr_filled)   # [10. 30. 30. 30. 50.]
```

### Per-column imputation on 2D data

```python
data = np.array([
    [1.0, np.nan, 3.0],
    [4.0, 5.0,    np.nan],
    [np.nan, 8.0, 9.0]
])

col_means = np.nanmean(data, axis=0)         # mean per column, ignoring NaN
inds = np.where(np.isnan(data))
data[inds] = np.take(col_means, inds[1])     # fill each NaN with its column's mean

print(data)
# [[1.  6.5 3. ]
#  [4.  5.  6. ]
#  [2.5 8.  9. ]]
```

> 💡 `np.take(col_means, inds[1])` looks up the right column mean for each NaN position using the column-index array from `np.where`.

---

## 3. Outlier Detection

Outliers distort means, std, and downstream models. Two standard NumPy-native approaches:

### Z-score method

A value is an outlier if it's more than `k` standard deviations from the mean (commonly `k=3`).

```python
data = np.array([12, 15, 14, 10, 200, 13, 11, 14, 12, -150])

mean = np.mean(data)
std  = np.std(data)
z_scores = (data - mean) / std

outliers = data[np.abs(z_scores) > 3]
clean    = data[np.abs(z_scores) <= 3]

print("Outliers:", outliers)
print("Clean:", clean)
```

### IQR (Interquartile Range) method — more robust to extreme values

```python
data = np.array([12, 15, 14, 10, 200, 13, 11, 14, 12, -150])

q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask = (data < lower_bound) | (data > upper_bound)
outliers = data[outlier_mask]
clean    = data[~outlier_mask]

print("Bounds:", lower_bound, upper_bound)
print("Outliers:", outliers)
```

> 💡 **Z-score vs IQR:** Z-score assumes roughly normal data and is itself sensitive to extreme outliers (since they inflate `std`). IQR is based on percentiles, so it's more robust — generally the safer default for unknown distributions.

### Capping instead of removing (Winsorization)

```python
data = np.array([12, 15, 14, 10, 200, 13, 11, 14, 12, -150])

capped = np.clip(data, lower_bound, upper_bound)
print(capped)
```

> 💡 Capping preserves row count (important when other columns in the same row hold valid data) instead of dropping the entire row.

---

## 4. Data Type Cleanup

Real CSVs often load everything as `float64` or even `object` due to mixed formatting. Cleaning dtypes early saves memory and prevents silent bugs.

```python
data = np.array([1.0, 2.0, 3.0, 4.0])

# If values are actually whole numbers, downcast
data_int = data.astype(np.int32)
```

### Detect columns that are secretly integers

```python
col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

is_whole = np.all(col == np.floor(col))
print(is_whole)   # True → safe to cast to int
```

### Replacing sentinel values (e.g. -999 used as "missing")

```python
data = np.array([23, -999, 45, 67, -999, 12])

data_clean = np.where(data == -999, np.nan, data)
print(data_clean)   # [23. nan 45. 67. nan 12.]
```

> 💡 Many real-world datasets (especially older sensor or survey data) use sentinel values like `-999`, `9999`, or `-1` instead of true NaN — always check for these before assuming the data is clean.

---

## 5. Encoding Categorical-Looking Data

Sometimes categorical labels arrive as numeric codes — useful to understand before handing off to Pandas/sklearn.

### Label encoding via `np.unique`

```python
categories = np.array(['red', 'blue', 'green', 'blue', 'red', 'green'])

unique_vals, encoded = np.unique(categories, return_inverse=True)
print(unique_vals)   # ['blue' 'green' 'red']
print(encoded)       # [2 0 1 0 2 1]  → integer codes
```

### One-hot encoding via broadcasting (recap from ch15)

```python
labels = encoded
n_classes = len(unique_vals)

one_hot = (labels[:, np.newaxis] == np.arange(n_classes)).astype(int)
print(one_hot)
# [[0 0 1]
#  [1 0 0]
#  [0 1 0]
#  [1 0 0]
#  [0 0 1]
#  [0 1 0]]
```

---

## 6. Feature Scaling — Quick Recap

Already covered in detail (ch06 for stats, ch15 for broadcasting mechanics) — summarized here as the standard preprocessing step.

```python
data = np.random.rand(100, 4) * 100

# Min-Max → range [0, 1]
data_minmax = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# Z-score (standardization) → mean=0, std=1
data_zscore = (data - data.mean(axis=0)) / data.std(axis=0)
```

> ⚠️ Always compute scaling statistics (`min`, `max`, `mean`, `std`) **only on training data**, then apply the same values to test data — never recompute on test data (data leakage).

---

## 7. NumPy → Pandas Handoff

This is the actual bridge — once data is cleaned, converting to a DataFrame is one line, but doing it *correctly* (with proper column names and dtypes) matters.

```python
import pandas as pd

clean_array = np.array([
    [25, 50000, 1],
    [32, 62000, 0],
    [45, 81000, 1]
])

columns = ['age', 'salary', 'purchased']
df = pd.DataFrame(clean_array, columns=columns)

print(df.dtypes)   # all float64/int64 depending on array dtype
```

### Preserving correct dtypes during conversion

```python
df = pd.DataFrame({
    'age': clean_array[:, 0].astype(int),
    'salary': clean_array[:, 1].astype(float),
    'purchased': clean_array[:, 2].astype(bool)
})
```

### When to do work in NumPy vs Pandas

| Task | Better in |
|---|---|
| Heavy numerical computation, large arrays | NumPy (faster, lower memory) |
| Mixed-type tables, labeled columns | Pandas |
| Joins, group-by, time series | Pandas |
| Linear algebra, simulations, vectorized math | NumPy |
| Final clean structured dataset for analysis | Pandas |

> 💡 **Common real workflow:** load raw → clean with NumPy (NaN handling, outliers, dtype fixes, scaling) → convert to Pandas for labeled exploration, joins, and reporting. Since you already know Pandas, this is the missing "before" step that's usually skipped in tutorials.

---

## 8. End-to-End Preprocessing Pipeline

A realistic mini-example tying every section together.

```python
import numpy as np
import pandas as pd

# Raw data with missing values, sentinel values, and an outlier
raw = np.array([
    [25, 50000, -999],
    [32, np.nan, 1],
    [45, 81000, 0],
    [29, 999999, 1],   # extreme outlier in salary
    [38, 67000, 0]
])

# Step 1 — Replace sentinel values with NaN
raw[:, 2] = np.where(raw[:, 2] == -999, np.nan, raw[:, 2])

# Step 2 — Impute missing values with column mean
col_means = np.nanmean(raw, axis=0)
inds = np.where(np.isnan(raw))
raw[inds] = np.take(col_means, inds[1])

# Step 3 — Outlier capping on salary (col 1) using IQR
salary = raw[:, 1]
q1, q3 = np.percentile(salary, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
raw[:, 1] = np.clip(salary, lower, upper)

# Step 4 — Scale age and salary (z-score)
raw[:, [0, 1]] = (raw[:, [0, 1]] - raw[:, [0, 1]].mean(axis=0)) / raw[:, [0, 1]].std(axis=0)

# Step 5 — Hand off to Pandas
df = pd.DataFrame(raw, columns=['age', 'salary', 'flag'])
print(df)
```

---

## Summary

```
NumPy in Real Data Science
 ├── Loading           → loadtxt vs genfromtxt (handles missing data)
 ├── Missing values
 │    ├── np.isnan()        → detect
 │    ├── np.nan* functions → nanmean, nansum, nanstd (ignore NaN)
 │    └── impute or drop    → np.where() fill, or Boolean row filter
 ├── Outliers
 │    ├── Z-score method    → |z| > 3, sensitive to extreme values
 │    ├── IQR method        → percentile-based, more robust
 │    └── np.clip()         → cap instead of drop (Winsorization)
 ├── Dtype cleanup     → detect whole-number floats, replace sentinels
 ├── Encoding          → np.unique(return_inverse) for labels, broadcasting for one-hot
 ├── Scaling (recap)   → min-max / z-score, fit on train only
 └── Pandas handoff    → pd.DataFrame(array, columns=...), preserve dtypes explicitly
```