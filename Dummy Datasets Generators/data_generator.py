import numpy as np
import pandas as pd

# ── 1. Define your schema here ────────────────────────────────────────────────
# Format: ("column_name", "dtype", predefined_values_or_range)
#
#   dtype options:
#     "int"      → (min, max)          e.g. (1, 1000)
#     "float"    → (min, max)          e.g. (0.0, 100.0)
#     "bool"     → None
#     "str"      → list of choices     e.g. ["A", "B", "C"]
#     "date"     → ("start", "end")    e.g. ("2020-01-01", "2024-12-31")

from product_catalog import product_catalog

items=list(product_catalog.keys())
from address_sets import addresses
payment=["Cash On Delivery","credit/debit cards","Apple Pay","Google Pay","Paypal","NFC","Buy now pay later"]
columns = [
    # Employees Fields
    # ("id",          "int",   (1, 10_000_000)),
    # ("name",        "str",   ["Alice", "Bob", "Carol", "Dave", "Eve","Jon","Sara","Rob","Che","Arya",""]),
    # ("score",       "float", (0.0, 100.0)),
    # ("is_active",   "bool",  None),
    # ("department",  "str",   ["HR", "Engineering", "Sales", "Marketing"]),
    # ("joined_date", "date",  ("2015-01-01", "2024-12-31")),


    # Sales Fields
    ("order_id",     "int", (1000, 11_000)),
    ("item", "str",items),
    ("quantity", "int", (14,555)),
    ("rate","float",(1.25,282.20)),
    ("order_data", "date", ("2023-01-23","2026-03-19")),
    ("delivery_address", "str",addresses ),
    ("payment_method","str",payment)

]

# ── 2. Number of rows to generate ─────────────────────────────────────────────
NUM_ROWS = 1_000

# ── 3. Generator ──────────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

data = {}
for col_name, dtype, values in columns:

    if dtype == "int":
        data[col_name] = rng.integers(values[0], values[1] + 1, size=NUM_ROWS)

    elif dtype == "float":
        data[col_name] = np.round(rng.uniform(values[0], values[1], size=NUM_ROWS), 2)

    elif dtype == "bool":
        data[col_name] = rng.integers(0, 2, size=NUM_ROWS, dtype=np.int8).astype(bool)

    elif dtype == "str":
        picks = np.array(values, dtype=object)
        data[col_name] = picks[rng.integers(0, len(picks), size=NUM_ROWS)]

    elif dtype == "date":
        start = np.datetime64(values[0], "D")
        end   = np.datetime64(values[1], "D")
        days  = int((end - start) / np.timedelta64(1, "D"))
        data[col_name] = pd.to_datetime(start + rng.integers(0, days + 1, size=NUM_ROWS).astype("timedelta64[D]"))

# ── Auto-derive category from product_catalog ─────────────────────────────────
data["category"] = np.array(
    [product_catalog.get(item, "Miscellaneous") for item in data["item"]],
    dtype=object
)

df = pd.DataFrame(data)

# reorder: move category right after item
cols = list(df.columns)
cols.insert(cols.index("item") + 1, cols.pop(cols.index("category")))
df = df[cols]

# ── 4. Output ──────────────────────────────────────────────────────────────────
print(df.head(10).to_string(index=False))
print(f"\nShape : {df.shape}")
print(f"dtypes:\n{df.dtypes}")
file=input("Enter file name: ")
file_name="dataFiles/"+file
df.to_csv(file_name,index=False)
print(f"\nSaved → {file}")