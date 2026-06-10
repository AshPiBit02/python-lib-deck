# NumPy — Linear Algebra

> 🧠 **Why this matters for ML/DS:**  
> Nearly every ML algorithm runs on linear algebra under the hood — neural network forward passes are matrix multiplications, PCA uses eigenvalues, linear regression uses matrix inverse, and distance metrics use dot products. Mastering this chapter means you understand *what your models are actually doing*.

---

## Table of Contents
1. [Dot Product](#1-dot-product)
2. [Matrix Multiplication](#2-matrix-multiplication)
3. [Determinant](#3-determinant)
4. [Inverse](#4-inverse)
5. [Eigenvalues & Eigenvectors](#5-eigenvalues--eigenvectors)

---

## 1. Dot Product

The dot product takes two vectors and returns a **single scalar** — the sum of element-wise products.

```
a · b = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]
```

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.dot(a, b)        # 1*4 + 2*5 + 3*6 = 32
```

### On 2D — Row vector · Column vector

```python
a = np.array([[1, 2, 3]])   # shape (1, 3)
b = np.array([[4], [5], [6]])  # shape (3, 1)

np.dot(a, b)        # [[32]]  → shape (1, 1)
```

> 💡 **ML context:** The dot product is at the core of every linear model prediction:  
> `y = w · x + b` — weights dot features gives the raw score.

---

## 2. Matrix Multiplication

Multiplies two matrices — the number of **columns in A** must equal **rows in B**.

```
A (m×n) @ B (n×p) → result (m×p)
```

```python
A = np.array([[1, 2],
              [3, 4]])   # shape (2, 2)

B = np.array([[5, 6],
              [7, 8]])   # shape (2, 2)

# Two equivalent ways
np.dot(A, B)
A @ B
# [[19 22]
#  [43 50]]
```

### Non-square matrices

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

B = np.array([[7,  8],
              [9,  10],
              [11, 12]])    # shape (3, 2)

A @ B
# [[ 58  64]
#  [139 154]]               # shape (2, 2)
```

### Shape mismatch — Error

```python
A = np.array([[1, 2], [3, 4]])     # (2, 2)
B = np.array([[1, 2, 3]])          # (1, 3)

A @ B    # ❌ ValueError — A cols (2) ≠ B rows (1)
```

> ⚠️ `*` is element-wise, `@` is matrix multiplication — never confuse them.

> 💡 **ML context:** Forward pass in a neural network layer is literally `output = input @ weights + bias`.

---

## 3. Determinant

`np.linalg.det()` — returns a scalar representing how much a matrix **scales space**. Key in solving linear systems.

- `det = 0` → matrix is **singular** (not invertible, no unique solution)
- `det ≠ 0` → matrix is **invertible**

```python
A = np.array([[3, 8],
              [4, 6]])

np.linalg.det(A)    # 3*6 - 8*4 = 18 - 32 = -14.0
```

### 3×3

```python
B = np.array([[6, 1, 1],
              [4, -2, 5],
              [2,  8, 7]])

np.linalg.det(B)    # -306.0
```

### Singular matrix

```python
C = np.array([[2, 4],
              [1, 2]])

np.linalg.det(C)    # 0.0 → singular, cannot be inverted
```

> 💡 **ML context:** In linear regression, if features are perfectly correlated (multicollinearity), the matrix becomes singular — `det = 0` — and the normal equation has no unique solution.

---

## 4. Inverse

`np.linalg.inv()` — returns the inverse matrix `A⁻¹` such that `A @ A⁻¹ = I` (identity).

Only exists when `det(A) ≠ 0`.

```python
A = np.array([[3., 8.],
              [4., 6.]])

A_inv = np.linalg.inv(A)
print(A_inv)
# [[-0.214  0.571]
#  [ 0.286 -0.107]]

# Verify: A @ A⁻¹ should give identity
np.round(A @ A_inv)
# [[1. 0.]
#  [0. 1.]]
```

### Solving a Linear System using Inverse

System: `Ax = b` → solution: `x = A⁻¹ · b`

```python
A = np.array([[2., 1.],
              [5., 3.]])

b = np.array([8., 21.])

x = np.linalg.inv(A) @ b
print(x)    # [3. 2.]  → solution: x=3, y=2
```

> ⚠️ For large systems, `np.linalg.solve(A, b)` is preferred over computing the inverse explicitly — it's numerically more stable and faster.

```python
np.linalg.solve(A, b)   # [3. 2.]  → same result, better practice
```

> 💡 **ML context:** Linear regression closed-form solution (Normal Equation):  
> `w = (XᵀX)⁻¹ Xᵀy` — directly uses matrix inverse.

---

## 5. Eigenvalues & Eigenvectors

`np.linalg.eig()` — returns eigenvalues and eigenvectors of a square matrix.

For a matrix `A`, eigenvector `v` and eigenvalue `λ` satisfy:
```
A · v = λ · v
```
The matrix transforms `v` only by **scaling** it (by factor `λ`), not rotating it.

```python
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(eigenvalues)     # [5. 2.]
print(eigenvectors)
# [[ 0.894 -0.707]
#  [ 0.447  0.707]]
# each COLUMN is an eigenvector
```

> ⚠️ Columns of the returned `eigenvectors` matrix are the eigenvectors, not rows.

### Verify: A·v = λ·v

```python
λ = eigenvalues[0]        # 5.0
v = eigenvectors[:, 0]    # first eigenvector

np.allclose(A @ v, λ * v)  # True
```

### Symmetric matrices — `np.linalg.eigh()`

For symmetric matrices (like covariance matrices), use `eigh()` — more numerically stable and returns real values.

```python
cov = np.array([[4., 2.],
                [2., 3.]])

eigenvalues, eigenvectors = np.linalg.eigh(cov)
```

> 💡 **ML context:** PCA (Principal Component Analysis) works by computing eigenvectors of the covariance matrix — those eigenvectors are the **principal components** (new axes). Eigenvalues tell you how much variance each component explains.

---

## `np.linalg` — Quick Reference

```python
np.dot(a, b)              # dot product (vectors) or matrix multiply
A @ B                     # matrix multiplication (preferred syntax)
np.linalg.det(A)          # determinant
np.linalg.inv(A)          # matrix inverse
np.linalg.solve(A, b)     # solve Ax=b (better than inv for systems)
np.linalg.eig(A)          # eigenvalues & eigenvectors
np.linalg.eigh(A)         # for symmetric matrices (PCA, covariance)
```

---

## Summary

```
Linear Algebra
 ├── Dot product          → scalar from two vectors, core of weighted sum
 ├── Matrix multiply      → A @ B, cols of A must equal rows of B
 ├── Determinant          → det=0 means singular (not invertible)
 ├── Inverse              → A⁻¹ such that A @ A⁻¹ = I
 │    └── np.linalg.solve → preferred for linear systems
 └── Eigenvalues          → np.linalg.eig(), columns = eigenvectors
      └── eigh()          → use for symmetric/covariance matrices (PCA)
```