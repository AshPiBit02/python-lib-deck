# HTTP Foundations — Reference Notes

Covers: HTTP Status Codes, Request Anatomy, JSON as the Data Format, REST Principles.
(HTTP Methods skipped — already implemented in the Library API project.)

---

## 1. HTTP Status Codes

Status codes tell the client **what happened**, without it having to parse the body. Grouped by first digit:

| Range | Meaning |
|---|---|
| 2xx | Success |
| 3xx | Redirection |
| 4xx | Client made a mistake |
| 5xx | Server made a mistake |

### The ones you'll actually use

| Code | Name | Use case |
|---|---|---|
| `200` | OK | Successful GET, PUT, PATCH, DELETE — general success |
| `201` | Created | Successful POST that created a new resource |
| `204` | No Content | Success but nothing to return (e.g. DELETE with no body) — *not in your list but very common, worth knowing* |
| `400` | Bad Request | Client sent malformed/invalid data that isn't a validation-schema issue (e.g. business rule violation like "no copies available") |
| `401` | Unauthorized | Client isn't authenticated at all (no/invalid token) |
| `403` | Forbidden | Client **is** authenticated but lacks permission (e.g. non-admin hitting an admin route) |
| `404` | Not Found | Resource with that ID/path doesn't exist |
| `422` | Unprocessable Entity | Request body/params failed **schema validation** (FastAPI + Pydantic raises this automatically) |
| `500` | Internal Server Error | Unhandled exception on the server — you should almost never raise this manually; it means something broke |

### 401 vs 403 — the distinction people mix up
- `401` = "I don't know who you are" (missing/invalid credentials)
- `403` = "I know who you are, but you can't do this" (valid credentials, insufficient permission)

### 400 vs 422 — the distinction relevant to FastAPI
- `422` = Pydantic/FastAPI automatically raises this when the **shape/type** of incoming data is wrong (missing required field, wrong type, failed `Field()` constraint)
- `400` = **you** manually raise this for a **business logic** failure where the data was well-formed but the action itself isn't allowed (e.g. borrowing a book with 0 copies left — valid request, invalid *outcome*)

```python
from fastapi import HTTPException

# business rule failure -> 400, not 422
if copies_available == 0:
    raise HTTPException(status_code=400, detail="No copies available")
```

### Setting status codes in FastAPI
```python
@app.post("/books", status_code=201)   # default success code for the route
def create_book(book: Book):
    ...
```
`HTTPException(status_code=...)` overrides this for error paths within the same route.

---

## 2. Request Anatomy

Every HTTP request has four parts a FastAPI route can pull data from. Knowing which part data belongs in is what decides how you write the function signature.

| Part | What it carries | FastAPI equivalent | Example |
|---|---|---|---|
| **Path params** | Identifies a specific resource | plain function parameter matching `{}` in the route | `/books/{book_id}` → `book_id: int` |
| **Query params** | Filters/modifies a collection, optional by nature | function parameter with a default value | `/books?author=Martin` → `author: str \| None = None` |
| **Headers** | Metadata about the request (auth tokens, content type, client info) | `Header()` | `X-API-Key: abc123` |
| **Body** | The actual payload — data being created/updated | Pydantic model as a parameter | `POST /books` with JSON `{"title": "..."}` |

```python
from fastapi import Header

@app.get("/books/{book_id}")
def get_book(
    book_id: int,                     # path param
    include_reviews: bool = False,    # query param
    x_api_key: str = Header(...),     # header
):
    ...
```

### Rule of thumb for choosing where data goes
- **Identifies *which* resource** → path param
- **Optional, filters/sorts/paginates a list** → query param
- **Metadata not about the resource itself** (auth, tracing, content negotiation) → header
- **The actual data being sent to create/update something** → body

---

## 3. JSON as the Data Format

JSON (`JavaScript Object Notation`) is the near-universal format for REST API payloads — text-based, human-readable, maps cleanly to nested dicts/lists.

### Why JSON (vs XML, which Java devs often used historically)
- Smaller payload size than XML
- Native mapping to Python dicts/lists and JS objects — no parsing ceremony
- FastAPI + Pydantic handle JSON ↔ Python object conversion automatically in both directions

### Content-Type header
```
Content-Type: application/json
```
Tells the server how to interpret the request body. FastAPI expects this by default when a route parameter is a Pydantic model — if a client sends a body without this header (or with the wrong one), parsing fails.

### Conversion is automatic in FastAPI
```python
class Book(BaseModel):
    title: str
    author: str

@app.post("/books")
def create_book(book: Book):   # incoming JSON -> Book instance, automatically
    return book                 # Book instance -> outgoing JSON, automatically
```
You already used this in the Library API — `book.model_dump()` explicitly converts a `Book` back to a dict when you needed to merge it with other data; FastAPI does the dict→JSON step for you on return regardless.

### JSON data types → Python types
| JSON | Python |
|---|---|
| `string` | `str` |
| `number` | `int` / `float` |
| `true`/`false` | `bool` |
| `null` | `None` |
| `array` | `list` |
| `object` | `dict` (or a Pydantic model) |

Note: JSON has **no `date`/`datetime` type** — dates are sent as ISO-8601 strings (`"2026-08-12T10:00:00"`) and Pydantic parses them into `datetime` objects for you if the field is typed as `datetime`.

---

## 4. REST Principles

REST (Representational State Transfer) is a set of conventions, not a protocol — you already followed most of these in the Library API without naming them.

### Stateless
Each request must contain **everything** the server needs to process it — the server holds no memory of previous requests from a client between calls.

- No server-side "session" that remembers what a client did last
- Any state (auth, identity) must travel *with* the request — typically as a header (e.g. `Authorization: Bearer <token>`), not stored server-side per-client
- **Why it matters:** statelessness is what lets you run multiple identical server instances behind a load balancer — any instance can handle any request, since none of them are holding onto client-specific memory

```
❌ Stateful (bad for REST): server remembers "this client is logged in" from a previous request
✅ Stateless: client sends its token on every request; server validates fresh each time
```

### Resource-based URLs
URLs should name **things (nouns)**, not **actions (verbs)** — the HTTP method itself already expresses the action.

| ❌ Not resource-based | ✅ Resource-based |
|---|---|
| `/getBooks` | `GET /books` |
| `/createNewBook` | `POST /books` |
| `/deleteBook?id=3` | `DELETE /books/3` |
| `/books/find/Martin` *(borderline — action-like)* | `GET /books?author=Martin` |

Your Library API mostly follows this — `/books/{id}/borrow` and `/books/{id}/return` are a common, accepted exception: they represent **actions on a resource's state** where modeling them as a "sub-resource" (e.g. a `POST /loans` resource) would be more purist-REST but often more complex than it's worth for smaller APIs.

### Other core REST conventions (bonus, not in your list but commonly grouped with this topic)
- **Uniform interface** — same HTTP methods mean the same thing across every resource in the API (a `GET` never modifies data, a `DELETE` always removes)
- **Client–server separation** — frontend and backend evolve independently as long as the API contract holds
- **Cacheable** — responses should indicate whether they can be cached (relevant later for performance/Redis phase)

---
