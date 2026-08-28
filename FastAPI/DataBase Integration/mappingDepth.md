# SQLAlchemy Model Mapping — In Depth

Everything involved in mapping a Python class to a real database table: column
types, constraints, primary/foreign keys, relationships, and the techniques that
tie related tables together.

---

## 1. Column Types — full reference

```python
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, Numeric
from datetime import datetime

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String(100))          # VARCHAR(100) — bounded text
    description = Column(Text)           # unbounded text — no length limit
    price = Column(Numeric(10, 2))       # exact decimal — NEVER use Float for money
    weight_kg = Column(Float)            # approximate decimal — fine for non-financial values
    in_stock = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

| SQLAlchemy type | Postgres equivalent | Use for |
|---|---|---|
| `Integer` | `INTEGER` | whole numbers, IDs |
| `String(n)` | `VARCHAR(n)` | bounded text (names, emails) |
| `Text` | `TEXT` | unbounded text (descriptions, notes) |
| `Float` | `DOUBLE PRECISION` | approximate decimals (measurements) |
| `Numeric(precision, scale)` | `NUMERIC` | **exact** decimals — always use for money |
| `Boolean` | `BOOLEAN` | true/false flags |
| `DateTime` | `TIMESTAMP` | date + time values |
| `Date` | `DATE` | date only, no time |

**Why `Numeric` over `Float` for money:** `Float` uses binary floating-point,
which can't represent values like `19.99` exactly — repeated arithmetic
accumulates rounding errors. `Numeric(10, 2)` stores an exact decimal (10 total
digits, 2 after the decimal point) — always correct for currency.

---

## 2. Constraints — enforcing rules at the database level

```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    age = Column(Integer, nullable=True)          # nullable=True is the default
    username = Column(String(50), unique=True, index=True)
```

| Constraint | Meaning |
|---|---|
| `nullable=False` | column cannot be `NULL` — `NOT NULL` |
| `unique=True` | no two rows can share this value |
| `default=...` | value used automatically if none provided on insert |
| `index=True` | creates a DB index — speeds up lookups/filters on this column |
| `server_default=...` | default computed **by Postgres itself**, not Python (e.g. `func.now()`) |

**`default` vs `server_default`:**
```python
created_at = Column(DateTime, default=datetime.utcnow)          # Python computes it at insert time
created_at = Column(DateTime, server_default=func.now())         # Postgres computes it — safer for
                                                                      # multi-app/direct-SQL scenarios
```

---

## 3. Primary Keys

```python
id = Column(Integer, primary_key=True, index=True)
```

- Uniquely identifies each row — no two rows can share a primary key value
- On Postgres, an `Integer` primary key auto-increments by default (backed by a
  `SERIAL`/sequence) — you never manually assign `id`
- Every table needs exactly one (can be composite — multiple columns together,
  rare in typical CRUD apps, common in pure join/association tables)

**Composite primary key example** (two columns together form uniqueness):
```python
class EnrollmentLog(Base):
    __tablename__ = "enrollment_logs"
    student_id = Column(Integer, ForeignKey("students.id"), primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), primary_key=True)
    enrolled_on = Column(DateTime, default=datetime.utcnow)
```

---

## 4. Foreign Keys — linking one table to another

```python
from sqlalchemy import ForeignKey

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    author_id = Column(Integer, ForeignKey("users.id"))   # <- points at users.id
```

- `ForeignKey("users.id")` — this column's values must match an existing `id` in
  the `users` table (or be `NULL`, unless also `nullable=False`)
- Enforced by Postgres itself — you cannot insert a `Post` with an `author_id`
  that doesn't exist in `users`, the DB rejects it
- The column itself (`author_id`) just stores an integer — it's the **relationship**
  (next section) that makes it feel like a Python object reference

---

## 5. Relationships — the Python-side connection

A foreign key alone only gets you an integer column. `relationship()` is what lets
you write `post.author.name` instead of manually looking up `author_id` yourself.

### One-to-Many (most common)

```python
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))

    posts = relationship("Post", back_populates="author")   # one user -> many posts

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    author_id = Column(Integer, ForeignKey("users.id"))

    author = relationship("User", back_populates="posts")   # many posts -> one user
```

Usage:
```python
user.posts          # list[Post] — all posts by this user
post.author          # User — the user who wrote this post
post.author.name     # walk the relationship like a normal attribute
```

**`back_populates` vs `backref` — two ways to declare the same thing:**
```python
# back_populates — explicit, both sides declared (recommended, clearer)
posts = relationship("Post", back_populates="author")
author = relationship("User", back_populates="posts")

# backref — implicit, only declare ONE side, the other is auto-generated
posts = relationship("Post", backref="author")   # creates post.author automatically
```
**Rule of thumb:** prefer `back_populates` — it's more explicit and both sides
are visible directly in the code, making the relationship easier to trace.

### Many-to-Many (needs an association table)

```python
from sqlalchemy import Table

student_course_association = Table(
    "student_course", Base.metadata,
    Column("student_id", Integer, ForeignKey("students.id"), primary_key=True),
    Column("course_id", Integer, ForeignKey("courses.id"), primary_key=True),
)

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    courses = relationship("Course", secondary=student_course_association, back_populates="students")

class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    students = relationship("Student", secondary=student_course_association, back_populates="courses")
```

- A many-to-many needs a **third table** (the association/join table) — a student
  can be in many courses, and a course can have many students, so a plain foreign
  key on either side isn't enough
- `secondary=student_course_association` tells SQLAlchemy to route through that
  join table automatically
- Usage: `student.courses` → `list[Course]`, `course.students` → `list[Student]`

### One-to-One

```python
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    profile = relationship("Profile", back_populates="user", uselist=False)

class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)   # unique=True enforces one-to-one
    bio = Column(Text)
    user = relationship("User", back_populates="profile")
```
- Structurally identical to one-to-many, but `unique=True` on the foreign key
  ensures only one `Profile` can point at a given `User`
- `uselist=False` tells SQLAlchemy to return a single object (`user.profile`), not
  a list — without it, `relationship()` assumes "many" by default

---

## 6. `cascade` — what happens to related rows on delete

```python
posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
```

Without `cascade`, deleting a `User` who has `Post`s either fails (FK constraint
violation) or leaves orphaned `Post`s with a dangling `author_id`, depending on DB
settings. `cascade="all, delete-orphan"` means: deleting the `User` automatically
deletes all their `Post`s too.

| Cascade option | Effect |
|---|---|
| `"all"` | propagate all operations (save, update, delete) to related objects |
| `"delete"` | deleting the parent deletes children |
| `"delete-orphan"` | a child removed from the relationship (not just parent-deleted) gets deleted too |
| `"save-update"` | adding a parent to the session also adds its children |

**Use cascades carefully** — `delete-orphan` in particular can delete data you
didn't intend to lose. Only apply it where the child genuinely can't exist
without the parent (e.g. a `Post` without an `author` makes no sense; a `Comment`
without a `Post` makes no sense).

---

## 7. Lazy Loading vs Eager Loading — when related data is actually fetched

```python
posts = relationship("Post", back_populates="author", lazy="select")   # default
```

| `lazy=` value | Behavior |
|---|---|
| `"select"` (default) | related rows fetched in a **separate query**, only when you access `.posts` |
| `"joined"` | related rows fetched via a `JOIN` in the **same** query — one round trip |
| `"subquery"` | related rows fetched via a second query using a subquery — good for collections |
| `"raise"` | accessing the relationship without explicit eager-loading raises an error — useful for catching accidental slow queries |

**The N+1 query problem** — the classic performance trap this setting affects:
```python
users = db.query(User).all()          # 1 query
for user in users:
    print(user.posts)                    # 1 query PER user — N additional queries!
```
Fix with eager loading:
```python
from sqlalchemy.orm import joinedload

users = db.query(User).options(joinedload(User.posts)).all()   # 1 query total, JOIN included
```

---

## 8. Indexes — beyond the automatic primary-key index

```python
email = Column(String(255), index=True)              # single-column index

# multi-column (composite) index
from sqlalchemy import Index
class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer)
    created_at = Column(DateTime)

    __table_args__ = (Index("ix_customer_created", "customer_id", "created_at"),)
```
Add an index on any column you frequently `filter()`/`WHERE` on. Indexes speed up
reads but slightly slow down writes (the index itself must be updated) — don't
index every column reflexively, only ones genuinely used in lookups/filters.

---