# Structured Arrays & Dtypes

---

## Table of Contents
1. [Defining Custom Data Types (dtypes)](#1-defining-custom-data-types-dtypes)
2. [Creating Structured Arrays](#2-creating-structured-arrays)
3. [Accessing and Modifying Fields](#3-accessing-and-modifying-fields)
4. [Memory Alignment (align=True)](#4-memory-alignment-aligntrue)
5. [Record Arrays (np.recarray)](#5-record-arrays-nprecarray)
6. [Nested Structured Arrays](#6-nested-structured-arrays)
7. [Advanced Field Tricks & Views](#7-advanced-field-tricks--views)
8. [Summary](#summary)

---

## 1. Defining Custom Data Types (dtypes)

Standard NumPy arrays are homogeneous, meaning every element must share the same data type. **Structured Data Types** break this rule, allowing you to define custom schemas with multiple fields of distinct types, resembling a database row or a C-struct.

### Data Type Descriptors (Type Strings)
NumPy uses a concise string syntax to define primitive components:
* `b1`: Boolean
* `i1`, `i2`, `i4`, `i8`: Signed integers (8, 16, 32, 64-bit)
* `u1`, `u2`, `u4`, `u8`: Unsigned integers
* `f2`, `f4`, `f8`: Floating-point numbers (half, single, double precision)
* `U<N>` / `S<N>`: Unicode string / Byte string of length `N`

### Byte Ordering (Endianness)
You can prefix type strings with endianness indicators to handle binary data from different CPU architectures:
* `<`: Little-Endian (least significant byte stored at the lowest address)
* `>`: Big-Endian (most significant byte stored at the lowest address)

```python
import numpy as np

# Declaring explicit data types with list of tuples format: (field_name, data_type)
simple_schema = np.dtype([('id', 'i4'), ('weight', '<f8')])

print(simple_schema.itemsize)  # 12 bytes (4 bytes for int32 + 8 bytes for float64)
```

---

## 2. Creating Structured Arrays

There are multiple ways to initialize a structured array. The most common approach uses a list of tuples containing data matching the target schema.

### Array Initialization

```python
# Defining a student log schema
student_dtype = np.dtype([
    ('roll_no', 'i4'),
    ('name', 'U20'),
    ('gpa', 'f4'),
    ('is_graduated', 'b1')
])

# Initializing with structured tuples
students = np.array([
    (101, 'Alice', 3.85, False),
    (102, 'Bob', 3.62, True),
    (103, 'Charlie', 3.91, False)
], dtype=student_dtype)

print(students.shape)  # (3,) -> It behaves like a 1D array of records
```

### Alternative Dictionary Definitions
For programmatic control over memory offsets, schemas can be declared using dictionaries:

```python
# Dictionary syntax: specify field names and formats explicitly
meta_dtype = np.dtype({
    'names': ['uid', 'version'],
    'formats': ['u4', 'u2']
})
```

---

## 3. Accessing and Modifying Fields

Unlike standard multi-dimensional indexing, structured arrays rely primarily on string-based keys to query specific record attributes.

### Dictionary-Style Field Access

```python
# Accessing a single field returns a homogeneous view of that column
names = students['name']
print(names)  # ['Alice' 'Bob' 'Charlie']

# Modifying a column directly updates the underlying array (it's a view, not a copy)
students['gpa'] += 0.05
print(students['gpa'])  # [3.9  3.67 3.96]
```

### Row and Multi-Field Slicing

```python
# Row indexing retrieves a single structured record
first_student = students[0]
print(first_student)  # (101, 'Alice', 3.9, False)

# Multi-field index queries return a view containing only requested columns
subset = students[['roll_no', 'is_graduated']]
print(subset)
# [(101, False) (102,  True) (103, False)]
```

> 💡 Modifying an array via multi-field selections like `students[['roll_no', 'gpa']]` can sometimes create a copy instead of a view depending on your NumPy version. Always update fields individually when performing guaranteed in-place mutations.

---

## 4. Memory Alignment (`align=True`)

By default, NumPy packs fields as tightly as possible in memory. However, hardware architectures often read memory more efficiently when variables line up with natural word boundaries (e.g., 4-byte boundaries for 32-bit integers). 

Passing `align=True` tells NumPy to mimic a C-compiler's padding rules.

```python
# Packed schema (unaligned)
packed_dtype = np.dtype([('flag', 'i1'), ('value', 'i4')])
print(packed_dtype.itemsize)  # 5 bytes (1 byte + 4 bytes)

# Aligned schema (adds 3 padding bytes after 'flag' to align 'value')
aligned_dtype = np.dtype([('flag', 'i1'), ('value', 'i4')], align=True)
print(aligned_dtype.itemsize)  # 8 bytes

# View the byte offsets of individual fields
print(aligned_dtype.fields)
# {'flag': (dtype('int8'), 0), 'value': (dtype('int32'), 4)} -> 'value' starts at byte 4!
```

> 💡 Use `align=True` when matching memory maps with binary data coming from compiled language assets like C, C++, or Fortran structures.

---

## 5. Record Arrays (`np.recarray`)

NumPy provides a specialized subclass of structured arrays called `np.recarray` (Record Arrays). They allow fields to be accessed as object attributes via dot notation (`arr.field`) instead of string keys (`arr['field']`).

```python
# Creating a record array from an existing structured array
rec_students = students.view(np.recarray)

# Attribute access
print(rec_students.name)  # ['Alice' 'Bob' 'Charlie']

# Assignment via attributes
rec_students.gpa[1] = 4.0
print(students[1]['gpa'])  # 4.0 -> The original structured array reflects this change
```

> 💡 While convenient, attribute access through `np.recarray` incurs a small performance overhead due to Python string lookups under the hood. For performance-critical code loops, stick to standard bracket indexing.

---

## 6. Nested Structured Arrays

Fields inside a structured array can themselves contain other structured arrays, enabling you to represent highly intricate hierarchical data layouts.

```python
# Creating sub-schemas
location_dtype = np.dtype([('x', 'f4'), ('y', 'f4')])

# Main schema referencing the sub-schema
particle_dtype = np.dtype([
    ('id', 'i4'),
    ('position', location_dtype),
    ('velocity', location_dtype)
])

# Initializing hierarchical records
particles = np.array([
    (1, (0.0, 0.0), (1.5, -0.5)),
    (2, (10.0, 5.0), (0.0, 2.1))
], dtype=particle_dtype)

# Deep field queries
print(particles['position']['x'])  # [ 0. 10.]
```

---

## 7. Advanced Field Tricks & Views

Structured arrays can be cast back and forth to standard primitive types if the memory footprints map cleanly.

### Treating Fields as Multidimensional Arrays
You can map a block of uniform variables directly into a shape matrix within a single record field:

```python
# Defining a matrix inside a field
sensor_schema = np.dtype([
    ('sensor_id', 'i4'),
    ('readings', 'f4', (3, 3))  # A 3x3 matrix embedded per record
])

data = np.zeros(2, dtype=sensor_schema)
data[0]['readings'] = np.eye(3)

print(data[0])
# (0, [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
```

### Unpacking Structured Arrays to Homogeneous Matrices
If all internal datatypes in your structured array are completely identical, you can easily view the structure as a standard flat multidimensional array:

```python
flat_dtype = np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
points_struct = np.array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], dtype=flat_dtype)

# Recasting the view into standard 2D float64 shape without allocating new memory
points_2d = points_struct.view('f8').reshape(-1, 3)
print(points_2d)
# [[1. 2. 3.]
#  [4. 5. 6.]]
```

---

## Summary

```
Structured Arrays & Dtypes
 ├── Custom Data Types (dtypes)
 │    ├── Type Descriptors  → 'i4', 'f8', 'U20' strings
 │    └── Endianness        → '<' (Little) vs '>' (Big) byte ordering
 ├── Array Creation
 │    ├── List of Tuples    → Matching shape entries sequentially
 │    └── Dictionary Schema → Explicit names, formats, and byte offsets
 ├── Operations & Management
 │    ├── Field Selection   → Bracket access using string keys
 │    └── Multi-field Views → Subset extraction of specific columns
 ├── Memory Optimization
 │    ├── align=True        → Adds C-struct padding bytes to memory blocks
 │    └── Nested Layouts    → Embed structures within parent structures
 └── Specialized Formats
      ├── np.recarray       → Attribute access lookup via dot-notation
      └── Sub-array Fields  → Shape matrices embedded natively inside fields
```