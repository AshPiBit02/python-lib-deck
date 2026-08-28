# Database Integration — SQLAlchemy + PostgreSQL

Reference notes on how a Python class becomes a real database table, and the pieces
that make the connection work.

---

## 1. The Core Idea — ORM in one sentence

An ORM (Object-Relational Mapper) lets you describe database tables as **Python
classes** and rows as **Python objects**, instead of writing raw SQL by hand.
SQLAlchemy is the ORM; it translates `User(name="Alice")` into an `INSERT` statement,
and `db.query(User).filter(...)` into a `SELECT`, behind the scenes.

---

## 2. Credentials — brief (assumes `.env` already loaded)

Credentials (`DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`) live in a
`.env` file, loaded via `pydantic-settings` into a `settings` object, and combined
into one connection string:

```
postgresql+psycopg2://<user>:<password>@<host>:<port>/<database_name>
```

- `postgresql` — the database type
- `+psycopg2` — the driver (the package that actually speaks Postgres's protocol)
- everything after — literal connection details

This string is the one thing every other piece below needs to actually reach the DB.

---

## 3. `engine` — the connection factory

```python
from sqlalchemy import create_engine

engine = create_engine(settings.database_url, pool_pre_ping=True)
```

- Does **not** connect immediately — it just knows *how to* connect, using the
  connection string
- Manages a **pool** of reusable connections internally, so your app isn't opening
  a brand new TCP connection to Postgres on every single query
- `pool_pre_ping=True` — checks a pooled connection is still alive before reusing it
  (avoids random failures if Postgres silently dropped an idle connection)

---

## 4. `SessionLocal` — the session factory

```python
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

- `SessionLocal` itself is **not** a session — it's a factory. Calling `SessionLocal()`
  produces a new `Session` object.
- A `Session` is your actual "conversation" with the DB for one unit of work — it
  tracks objects you've added/changed, and knows how to turn that into SQL.
- `autocommit=False` — nothing is written to the DB until you explicitly call
  `.commit()`. Prevents half-finished operations from silently persisting.
- `autoflush=False` — SQLAlchemy won't auto-sync pending changes before every query;
  you stay in control of when writes happen.

**Rule of thumb:** one `Session` per request. Never share a `Session` across
multiple concurrent requests.

---

## 5. `Base` — the shared parent for every table

```python
from sqlalchemy.orm import declarative_base

Base = declarative_base()
```

Every model class inherits from `Base`. This is what tells SQLAlchemy "this Python
class represents a real database table" — without it, a class is just a class,
with no connection to the DB at all.

---

## 6. `get_db()` — wiring a Session into a request

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- Opens a fresh session, hands it to whatever route needs it (`yield db`)
- **Always** closes it afterward — even if the route raises an exception — because
  of `finally`
- This is a `yield`-style dependency: setup before, guaranteed cleanup after
- Used in routes as `db: Session = Depends(get_db)`

---

## 7. Defining a Model — mapping a class to a real table

```python
from sqlalchemy import Column, Integer, String, Boolean
from db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
```

This is a direct translation of:
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);
```

| Python (SQLAlchemy) | SQL equivalent |
|---|---|
| `class User(Base):` | the class this maps to `users` |
| `__tablename__ = "users"` | actual table name in Postgres |
| `Column(Integer, primary_key=True)` | `SERIAL PRIMARY KEY` (auto-incrementing ID) |
| `index=True` | creates a DB index on that column, for faster lookups |
| `String(100)` | `VARCHAR(100)` |
| `nullable=False` | `NOT NULL` |
| `unique=True` | `UNIQUE` constraint |
| `default=True` | `DEFAULT TRUE` |

**Important distinction:** these `Column` types describe **storage**, not
validation. `EmailStr`-style format checking belongs in a separate Pydantic
schema (used at the API layer), never inside a SQLAlchemy `Column()` — the two
type systems serve different jobs and aren't interchangeable.

---

## 8. Creating the actual table in Postgres

Defining the class alone does **not** create anything in the database — it's just
a Python description until you explicitly run:

```python
from db.database import Base, engine
from models.user import User   # import required so Base knows this model exists

Base.metadata.create_all(bind=engine)
```

This inspects every model that inherits from `Base` and issues `CREATE TABLE` for
any that don't exist yet. Fine for early development; real projects use Alembic
migrations instead once the schema needs to evolve safely without dropping data.

---

## 9. The Full Chain — what happens, in order

1. `.env` is read → `settings` holds validated credentials
2. `settings.database_url` is built → `engine` is created (knows how to connect,
   hasn't yet)
3. A request comes in on a route using `Depends(get_db)` → `SessionLocal()` opens
   a real connection → a `Session` is handed to the route
4. Route/CRUD code uses that `Session` to query/insert/update/delete
5. Request finishes → `get_db()`'s `finally: db.close()` runs → connection released
   back to the pool

---