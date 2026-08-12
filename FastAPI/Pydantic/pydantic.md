# Pydantic — Reference Notes

FastAPI's Backbone. Where `dataclass`/`TypedDict` only *describe* shape, Pydantic **enforces** it at runtime — this is the one difference that matters most, and the reason FastAPI is built on it instead of the others.

You've already used `BaseModel`, basic field types, and default values in the Library API — this doc goes deeper and fills the gaps (nested models, validators, config, serialization details).

---

## 1. `BaseModel` — the foundation

```python
from pydantic import BaseModel

class Book(BaseModel):
    title: str
    author: str
    copies_available: int
```

What you get automatically, beyond what `dataclass` gives you:
- **Runtime validation** — wrong type raises `ValidationError` at construction, not silently accepted
- **Type coercion** — `"5"` passed for an `int` field is auto-converted to `5` (configurable, see §7)
- **Automatic JSON schema generation** — this is what powers FastAPI's `/docs` (Swagger UI)
- Everything `dataclass` gives (`__init__`, `__repr__`, `__eq__`)

```python
Book(title="Clean Code", author="Robert Martin", copies_available=2)   # ✅ works
Book(title="Clean Code", author="Robert Martin", copies_available="two")  # ❌ raises ValidationError
```

---

## 2. Field Types

Pydantic supports plain Python types directly, plus a set of special validated types.

### Standard types
```python
class Example(BaseModel):
    name: str
    age: int
    price: float
    in_stock: bool
    tags: list[str]
    metadata: dict[str, str]
```

### `Optional` fields
```python
from typing import Optional

class Example(BaseModel):
    nickname: Optional[str] = None   # same as: nickname: str | None = None
```
`Optional[X]` is shorthand for `X | None` — **it does not make the field optional to omit by itself**; you still need `= None` (or another default) for that. `Optional` only widens the *type*, the `=` sets the *default*.

### Special validated types (beyond plain Python types)
```python
from pydantic import EmailStr, HttpUrl

class User(BaseModel):
    email: EmailStr        # validates actual email format
    website: HttpUrl        # validates it's a well-formed URL
```
`EmailStr` requires the `email-validator` package (`pip install pydantic[email]` or `pip install email-validator`).

Other useful built-in special types: `PositiveInt`, `NegativeInt`, `conint()`, `constr()` (older-style constrained types — mostly superseded by `Field()`, see §5), `UUID4`, `SecretStr` (hides value in repr/logs — good for passwords).

```python
from pydantic import SecretStr

class Credentials(BaseModel):
    password: SecretStr

c = Credentials(password="hunter2")
print(c)   # password=SecretStr('**********') — value hidden in repr/logs
print(c.password.get_secret_value())  # 'hunter2' — explicit access only
```

---

## 3. Default Values and Required Fields

```python
class Book(BaseModel):
    title: str                          # required — no default
    copies_available: int = 1           # optional — has a default
    tags: list[str] = []                # ⚠️ safe in Pydantic (unlike dataclass) — see note
```

**Note on mutable defaults:** unlike `@dataclass`, Pydantic **does** handle `tags: list[str] = []` safely — it deep-copies the default per instance internally. You don't need `Field(default_factory=list)` for correctness in Pydantic, though it's still used for defaults that need to be *computed*.

### Explicitly required with `...`
```python
from pydantic import Field

class Book(BaseModel):
    title: str = Field(...)   # explicit "required", same as no default at all
```
`Field(...)` (Ellipsis) marks a field required even when you want to attach other metadata via `Field()` — you'll use this constantly once you add constraints (§5).

### Optional vs required — the four real combinations
| Declaration | Required to pass? | Can be `None`? |
|---|---|---|
| `x: str` | ✅ yes | ❌ no |
| `x: str = "default"` | ❌ no | ❌ no |
| `x: str \| None` | ✅ yes (but can pass `None`) | ✅ yes |
| `x: str \| None = None` | ❌ no | ✅ yes |

This table trips people up constantly — `str | None` alone does **not** make a field optional to omit.

---

## 4. Nested Models

Models can contain other models — Pydantic validates recursively.

```python
class Address(BaseModel):
    city: str
    zip_code: str

class User(BaseModel):
    name: str
    address: Address          # nested model

user = User(
    name="Alice",
    address={"city": "Kathmandu", "zip_code": "44600"}   # dict auto-converted to Address
)
print(user.address.city)   # "Kathmandu" — nested attribute access works normally
```

### Lists of nested models
```python
class Order(BaseModel):
    items: list[str]

class Customer(BaseModel):
    name: str
    orders: list[Order] = []
```

### Why this matters for FastAPI
Incoming JSON is often nested — Pydantic validates the whole tree in one shot, and a validation failure anywhere in the nested structure produces a precise error path (e.g. `address -> zip_code: field required`) instead of a vague top-level failure.

---

## 5. `Field()` — Constraints and Metadata

`Field()` attaches validation rules and documentation metadata to a field, beyond just its type.

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    price: float = Field(..., gt=0)                    # greater than 0
    quantity: int = Field(default=0, ge=0)              # >= 0
    discount: float = Field(default=0.0, ge=0, le=1)    # between 0 and 1
```

### Common constraint keywords
| Keyword | Applies to | Meaning |
|---|---|---|
| `gt`, `ge` | numbers | greater than / greater-or-equal |
| `lt`, `le` | numbers | less than / less-or-equal |
| `min_length`, `max_length` | str, list | length bounds |
| `pattern` | str | must match this regex |
| `default` | any | default value |
| `default_factory` | any | callable that produces the default (for computed defaults, e.g. `datetime.now`) |

### Metadata (documentation, not validation)
```python
price: float = Field(..., gt=0, description="Price in USD", examples=[19.99])
```
This metadata shows up directly in FastAPI's auto-generated `/docs` — a real, immediate payoff of using `Field()` beyond validation.

---

## 6. Validators — `field_validator` and `model_validator`

For validation logic that can't be expressed with `Field()` constraints alone.

### `field_validator` — validates a single field
```python
from pydantic import BaseModel, field_validator

class RegisterRequest(BaseModel):
    username: str

    @field_validator("username")
    @classmethod
    def no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("Username cannot contain spaces")
        return v   # must return the (possibly transformed) value
```
- Must be a `@classmethod`
- Runs **after** Pydantic's own type validation for that field
- Must `return` the value (you can also transform it, e.g. `return v.lower()`)
- Raising `ValueError` (not `HTTPException`) is correct here — Pydantic catches it and folds it into the overall `ValidationError` → FastAPI turns that into a `422` automatically

### `model_validator` — validates across multiple fields
```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode="after")
    def check_dates(self) -> "DateRange":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be after start_date")
        return self
```
Use `model_validator` whenever a rule depends on **more than one field together** (a single `field_validator` only sees its own field, in isolation).

- `mode="after"` — runs once all individual fields are already validated (most common; `self` is a fully-built instance)
- `mode="before"` — runs on the raw input dict, before field validation (rare; used for preprocessing raw data)

---

## 7. `model_config` — Configuring Model Behavior

```python
from pydantic import BaseModel, ConfigDict

class Book(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,   # auto .strip() all str fields
        str_to_lower=True,            # auto-lowercase all str fields
        extra="forbid",               # reject unknown fields instead of silently ignoring
    )
    title: str
```

| Option | Effect |
|---|---|
| `str_strip_whitespace` | trims leading/trailing whitespace on all `str` fields |
| `extra` | `"ignore"` (default) / `"forbid"` (reject unknown keys) / `"allow"` (keep them) |
| `frozen` | `True` makes the whole model immutable after creation, like `dataclass(frozen=True)` |
| `validate_assignment` | if `True`, re-validates fields when you mutate them after creation (`book.title = "x"` gets checked too, not just at construction) |

`extra="forbid"` is worth defaulting to in real APIs — silently dropping unexpected client fields can hide bugs.

---

## 8. Serialization — `model_dump()` and `model_dump_json()`

```python
book = Book(title="Clean Code", author="Robert Martin", copies_available=2)

book.model_dump()        # -> dict: {"title": "Clean Code", "author": "Robert Martin", "copies_available": 2}
book.model_dump_json()   # -> str:  '{"title":"Clean Code","author":"Robert Martin","copies_available":2}'
```

You already used `model_dump()` in the Library API to merge a validated `Book` back into your dict-based storage. Useful modifiers:

```python
book.model_dump(exclude_unset=True)     # only fields the client actually provided — you used this for PATCH
book.model_dump(exclude={"password"})   # drop specific fields (e.g. never leak a hash back to client)
book.model_dump(include={"title"})      # keep only specific fields
```

`exclude_unset=True` is exactly what makes partial-update (`PATCH`) endpoints work correctly — it distinguishes "client didn't send this field" from "client explicitly sent `null`."

---

## 9. Input Model vs Output Model — a pattern you'll use constantly

Don't reuse one model for both what the client sends and what you return — this is how password hashes or internal fields leak into responses.

```python
class UserIn(BaseModel):
    username: str
    password: str          # client sends this

class UserOut(BaseModel):
    username: str            # server returns this — no password field at all
```
This connects directly to FastAPI's `response_model=` (Phase 2.6) — the output model acts as a filter, stripping anything not declared on it even if your internal object has extra fields.

---

## 10. Pydantic vs `dataclass` vs `TypedDict` — the payoff, now concrete

| | `dataclass` | `TypedDict` | Pydantic `BaseModel` |
|---|---|---|---|
| Runtime validation | ❌ | ❌ | ✅ |
| Type coercion (`"5"` → `5`) | ❌ | ❌ | ✅ |
| Nested validation | manual | manual | ✅ automatic, recursive |
| Custom validation rules | manual `__post_init__` | ❌ not possible | ✅ `field_validator`/`model_validator` |
| JSON schema / docs generation | ❌ | ❌ | ✅ (powers FastAPI `/docs`) |
| `.model_dump()`/`.model_dump_json()` | manual (`asdict`) | trivial (already a dict) | ✅ built-in, with filtering options |

This is why Phase 3 exists as its own phase — Pydantic isn't "yet another way to define data," it's specifically the layer that turns untrusted incoming JSON into something you can trust inside your route logic.

---
