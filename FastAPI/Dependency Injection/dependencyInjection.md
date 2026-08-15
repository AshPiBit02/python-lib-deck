# Dependency Injection (DI) — Reference Notes

FastAPI's Most Powerful Feature. This is the mechanism that lets you share logic — DB connections, auth checks, pagination params, config — across many routes without copy-pasting it into every function.

---

## 1. What Dependency Injection Actually Is

**The core idea:** instead of a route function creating what it needs internally, it **declares what it needs as a parameter**, and something external ("the framework") supplies it at call time.

```python
# WITHOUT DI — route creates its own DB connection
@app.get("/users")
def get_users():
    db = connect_to_database()   # route is responsible for creating this
    ...

# WITH DI — route just declares it needs a db, doesn't know/care how it's made
from fastapi import Depends

@app.get("/users")
def get_users(db = Depends(get_db)):   # FastAPI calls get_db() and hands you the result
    ...
```

The route function no longer knows *how* `db` is constructed — only that it receives one. This is the whole point: **decoupling "what a route needs" from "how that thing is built."**

### Why this matters practically
- **Reusability** — the same `get_db` dependency plugs into every route that needs a DB session, written once
- **Testability** — in tests, you can swap `get_db` for a fake one without touching route code at all (§7)
- **Separation of concerns** — auth logic, DB logic, and business logic each live in their own function instead of tangled together inside every route

---

## 2. `Depends()` — the mechanism

```python
from fastapi import Depends

def get_query_params(skip: int = 0, limit: int = 10) -> dict:
    return {"skip": skip, "limit": limit}

@app.get("/items")
def list_items(params: dict = Depends(get_query_params)):
    return {"skip": params["skip"], "limit": params["limit"]}
```

What happens under the hood:
1. FastAPI sees `Depends(get_query_params)` on the route's parameter
2. Before calling `list_items`, FastAPI calls `get_query_params()` itself
3. `get_query_params` can itself take arguments that FastAPI resolves the *same way* it resolves route parameters (query params, path params, headers, other dependencies)
4. The **return value** of `get_query_params()` is passed in as `params`

**Key insight:** a dependency function is just a regular function. `Depends()` doesn't require anything special about how it's written — any callable can be a dependency, which is why it composes so cleanly.

---

## 3. Function-Based Dependencies

The most common form — a plain function, typically doing setup/validation/lookup and returning a value the route needs.

```python
def verify_api_key(x_api_key: str = Header(...)) -> str:
    if x_api_key != "secret123":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

@app.get("/protected")
def protected_route(api_key: str = Depends(verify_api_key)):
    return {"message": "You're in", "key_used": api_key}
```

Note: a dependency can `raise HTTPException` — if it does, the route function **never runs at all**. This is exactly how auth-gating works: the check happens before your business logic, and a failure short-circuits the whole request.

---

## 4. Sharing Dependencies Across Routes

The entire value proposition — write once, use everywhere:

```python
def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    ...
    return user

@app.get("/me")
def read_profile(user: User = Depends(get_current_user)):
    return user

@app.get("/orders")
def read_orders(user: User = Depends(get_current_user)):
    return get_orders_for(user)

@app.post("/orders")
def create_order(order: OrderIn, user: User = Depends(get_current_user)):
    ...
```

Three completely different routes, same auth logic, defined **once**. If the auth rule changes, you edit `get_current_user` and every route using it updates automatically.

---

## 5. Nested Dependencies

Dependencies can depend on other dependencies — FastAPI resolves the whole chain automatically, in order.

```python
def get_db() -> Session:
    return SessionLocal()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    user = db.query(User).filter(User.token == token).first()
    return user

def get_current_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    return user

@app.delete("/admin/users/{id}")
def delete_user(id: int, admin: User = Depends(get_current_admin)):
    ...
```

Chain: `delete_user` → needs `get_current_admin` → needs `get_current_user` → needs `get_db` + `oauth2_scheme`. FastAPI resolves this whole tree automatically, calling each dependency exactly once per request (see caching, §9) and wiring the results together.

This is how you build **layered permission checks** without duplicating the "who is this user" logic in every permission level.

---

## 6. `yield` Dependencies — Setup + Teardown

For anything that needs cleanup after the request finishes (DB sessions, file handles, network connections) — a `return` dependency has no way to run code *after* the route is done. `yield` does.

```python
def get_db():
    db = SessionLocal()
    try:
        yield db          # <- this is what gets injected into the route
        # nothing here runs on success — control returns after route finishes
    finally:
        db.close()          # <- ALWAYS runs, even if the route raised an exception
```

Execution order for a single request:
1. Code **before** `yield` runs (setup — open the connection)
2. The yielded value is injected into the route
3. The route function runs completely
4. Code **after** `yield` runs (teardown — close the connection), regardless of whether the route succeeded or raised an exception (if wrapped in `try/finally`, which it always should be)

```python
@app.get("/users/{id}")
def get_user(id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == id).first()
    # after this returns, get_db's `finally: db.close()` runs automatically
```

**Rule of thumb:** if a dependency acquires a resource that must be released, use `yield` + `try/finally`. If it just computes/returns a value with nothing to clean up, plain `return` is enough.

---

## 7. Class-Based Dependencies

A class with `__init__` (for configuration) and `__call__` (to act as the callable FastAPI invokes) — same functor pattern from decorators/Callable earlier in your learning.

```python
class Paginator:
    def __init__(self, default_limit: int = 10):
        self.default_limit = default_limit

    def __call__(self, skip: int = 0, limit: int | None = None) -> dict:
        return {"skip": skip, "limit": limit or self.default_limit}

pagination = Paginator(default_limit=20)

@app.get("/items")
def list_items(page: dict = Depends(pagination)):
    return page
```

**When to prefer this over a function:** when the dependency needs **configuration at definition time** that isn't itself a per-request input — here, `default_limit=20` is baked in once, while `skip`/`limit` still vary per request. A plain function can't easily carry that kind of "fixed config + per-call behavior" combo without extra parameters cluttering the route signature.

### Alternative — `Depends(ClassName)` shorthand
FastAPI also lets you pass the class itself (not an instance) directly:
```python
class CommonParams:
    def __init__(self, skip: int = 0, limit: int = 10):
        self.skip = skip
        self.limit = limit

@app.get("/items")
def list_items(commons: CommonParams = Depends(CommonParams)):
    return {"skip": commons.skip, "limit": commons.limit}
```
Here FastAPI treats `__init__`'s parameters exactly like a function dependency's parameters (resolving `skip`/`limit` from the query string) and constructs the instance itself. Common shorthand: `Depends()` with no argument at all, using the type annotation to infer the class — `commons: CommonParams = Depends()`.

---

## 8. Global Dependencies

Apply a dependency to **every route** in the app (or every route in a router), without adding `Depends(...)` to each function signature individually.

### Whole-app level
```python
app = FastAPI(dependencies=[Depends(verify_api_key)])
```

### Router level
```python
from fastapi import APIRouter

router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.get("/items")
def list_items():
    ...   # verify_api_key still runs, even though it's not in the signature
```

Use this for cross-cutting concerns that apply to *everything* under a router/app — e.g. "every admin route requires an admin token" — rather than repeating `Depends(get_current_admin)` on 15 separate functions.

**Important nuance:** when a dependency is declared this way, its **return value is not injected** into the route (since it's not a named parameter) — it only runs for its side effects (raising an exception on failure, logging, etc.). If you need the *value* it returns inside the route, you still declare it per-route as a normal parameter.

---

## 9. Dependency Caching (important, easy to miss)

Within a **single request**, if the same dependency is used multiple times (e.g. both `get_current_user` and a nested dependency both depend on `get_db`), FastAPI calls it **only once** and reuses the result — by default.

```python
def get_db(): ...

def dep_a(db = Depends(get_db)): ...
def dep_b(db = Depends(get_db)): ...

@app.get("/x")
def route(a=Depends(dep_a), b=Depends(dep_b)):
    # get_db() is called ONCE for this request, not twice
    ...
```

To force a fresh call even within the same request (rare), use `Depends(get_db, use_cache=False)`.

---

## 10. Overriding Dependencies (for testing)

This is the payoff for having used DI in the first place — you can swap real dependencies for fakes **without touching route code**, critical for Phase 10 (Testing).

```python
def override_get_db():
    db = TestSessionLocal()   # points at a test database instead
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# now every route using Depends(get_db) transparently uses the test version
client = TestClient(app)
response = client.get("/users")
```

`app.dependency_overrides` is a dict mapping the original dependency function to its replacement. This is *the* reason DI is worth the extra layer of indirection — untestable code (hardcoded DB connections inside routes) becomes trivially testable.

---

## 11. Multiple Approaches — summary comparison

| Style | Best for |
|---|---|
| Plain function, `return` | Stateless checks/lookups with nothing to clean up (auth validation, query param bundling) |
| Plain function, `yield` | Anything acquiring a resource needing guaranteed cleanup (DB sessions, file handles) |
| Class-based (`__call__`) | Dependency needs configuration baked in at definition time, reused with different configs across routes |
| Global (`app`/`router`-level) | Cross-cutting concern applying to *every* route in a scope, where you don't need the return value in the route itself |

---

## 📌 Note for your background
- `Depends()` ≈ Java Spring's `@Autowired`/`@Inject` — same concept, function-based instead of annotation-on-field
- Nested dependencies ≈ Spring's dependency graph resolution — a bean depending on other beans, resolved automatically
- `yield` dependencies ≈ Java's try-with-resources / C++ RAII (constructor acquires, destructor releases) — guaranteed cleanup tied to a scope
- Class-based dependencies (`__call__`) ≈ a configured service object injected once, reused across calls — similar to a Spring `@Bean` with constructor-injected config
- Dependency overriding ≈ Spring's `@MockBean` / test profile beans — swapping real implementations for test doubles without changing consuming code

---

