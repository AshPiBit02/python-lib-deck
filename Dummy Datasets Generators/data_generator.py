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
items=[
                     "Wireless Earbuds", "Mechanical Keyboard", "USB-C Hub", "Noise-Cancelling Headphones",
                     "Portable Charger", "Smart Watch", "Webcam HD", "LED Desk Lamp", "Gaming Mouse",
                     "Bluetooth Speaker", "Laptop Stand", "External SSD", "Smart Plug", "Screen Protector",
                     "Cable Organizer", "Running Shoes", "Leather Boots", "Canvas Sneakers", "Sandals",
                     "Hiking Boots", "Slip-On Loafers", "Rain Boots", "Dress Shoes", "Flip Flops",
                     "Ankle Boots", "Denim Jacket", "Hooded Sweatshirt", "Slim Fit Jeans", "Cotton T-Shirt",
                     "Wool Sweater", "Jogger Pants", "Polo Shirt", "Windbreaker", "Cargo Shorts",
                     "Formal Blazer", "Air Fryer", "Coffee Maker", "Blender", "Non-Stick Pan",
                     "Dish Rack", "Knife Set", "Rice Cooker", "Water Filter", "Toaster Oven",
                     "Salad Spinner", "Cutting Board", "Electric Kettle", "Food Processor", "Mixing Bowl Set",
                     "Spice Rack", "Yoga Mat", "Resistance Bands", "Dumbbells", "Jump Rope",
                     "Foam Roller", "Gym Gloves", "Protein Shaker", "Fitness Tracker", "Pull-Up Bar",
                     "Knee Sleeves", "Face Moisturizer", "Sunscreen SPF50", "Electric Toothbrush", "Hair Dryer",
                     "Shaving Kit", "Lip Balm", "Facial Cleanser", "Nail Polish Set", "Beard Oil",
                     "Eye Cream", "Sticky Notes", "Ballpoint Pens", "Spiral Notebook", "Highlighter Set",
                     "Desk Planner", "Whiteboard Markers", "File Folders", "Stapler", "Correction Tape",
                     "Index Cards", "Backpack", "Tote Bag", "Fanny Pack", "Laptop Bag",
                     "Crossbody Bag", "Wallet", "Sunglasses", "Baseball Cap", "Umbrella",
                     "Luggage Tag", "Jigsaw Puzzle", "Playing Cards", "Chess Set", "Building Blocks",
                     "Rubik's Cube", "Board Game", "Water Gun", "Frisbee", "Skipping Rope",
                     "Action Figure", "Green Tea", "Instant Coffee", "Protein Bar", "Dark Chocolate",
                     "Oat Milk", "Hot Sauce", "Honey Jar", "Trail Mix", "Granola",
                     "Sparkling Water", "Vitamin C Tablets", "Fish Oil Capsules", "Melatonin", "Multivitamin",
                     "Zinc Supplements", "Probiotic Capsules", "Hand Sanitizer", "First Aid Kit", "Heating Pad",
                     "Digital Thermometer",
                    ]
addresses = [
    "12 Baker Street - London - UK", "45 Elm Avenue - New York - USA", "78 Oak Lane - Toronto - Canada",
    "23 Maple Drive - Sydney - Australia", "56 Pine Road - Auckland - New Zealand", "89 Cedar Court - Dublin - Ireland",
    "34 Birch Boulevard - Berlin - Germany", "67 Walnut Street - Paris - France", "90 Ash Way - Tokyo - Japan",
    "11 Chestnut Close - Mumbai - India", "44 Willow Walk - Beijing - China", "77 Poplar Place - Seoul - South Korea",
    "22 Sycamore Square - Mexico City - Mexico", "55 Magnolia Mews - São Paulo - Brazil", "88 Hazel Hill - Cape Town - South Africa",
    "33 Spruce Street - Amsterdam - Netherlands", "66 Fir Avenue - Stockholm - Sweden", "99 Beech Road - Oslo - Norway",
    "14 Laurel Lane - Copenhagen - Denmark", "47 Holly Drive - Helsinki - Finland", "80 Ivy Court - Vienna - Austria",
    "25 Rowan Boulevard - Zurich - Switzerland", "58 Alder Close - Brussels - Belgium", "91 Juniper Way - Lisbon - Portugal",
    "36 Cypress Square - Madrid - Spain", "69 Redwood Mews - Rome - Italy", "10 Palm Place - Athens - Greece",
    "43 Mango Hill - Istanbul - Turkey", "76 Bamboo Street - Bangkok - Thailand", "21 Lotus Avenue - Singapore",
    "54 Orchid Road - Kuala Lumpur - Malaysia", "87 Jasmine Drive - Jakarta - Indonesia", "32 Blossom Court - Manila - Philippines",
    "65 Petal Lane - Dhaka - Bangladesh", "98 Garden Boulevard - Karachi - Pakistan", "13 Sunrise Close - Cairo - Egypt",
    "46 Sunset Way - Nairobi - Kenya", "79 Horizon Square - Lagos - Nigeria", "24 Breeze Mews - Accra - Ghana",
    "57 Valley Place - Casablanca - Morocco", "85 Riverside Drive - Chicago - USA", "17 Lakeview Avenue - Vancouver - Canada",
    "50 Hillside Road - Melbourne - Australia", "83 Seaside Court - Wellington - New Zealand", "28 Clifftop Lane - Edinburgh - Scotland",
    "61 Moorland Boulevard - Manchester - UK", "94 Harborview Street - Hamburg - Germany", "39 Vineyard Close - Lyon - France",
    "72 Meadow Way - Osaka - Japan", "15 Mountain Square - Bangalore - India", "48 Riverside Mews - Shanghai - China",
    "81 Lakeside Place - Busan - South Korea", "26 Beachfront Drive - Cancun - Mexico", "59 Rainforest Avenue - Rio de Janeiro - Brazil",
    "92 Savanna Road - Johannesburg - South Africa", "37 Canal Court - Rotterdam - Netherlands", "70 Fjord Lane - Gothenburg - Sweden",
    "13 Tundra Boulevard - Bergen - Norway", "46 Harbor Close - Aarhus - Denmark", "79 Arctic Way - Tampere - Finland",
    "24 Alpine Square - Salzburg - Austria", "57 Glacier Mews - Geneva - Switzerland", "90 Forest Place - Ghent - Belgium",
    "35 Coastal Drive - Porto - Portugal", "68 Desert Road - Seville - Spain", "11 Volcano Avenue - Naples - Italy",
    "44 Olive Court - Thessaloniki - Greece", "77 Spice Lane - Ankara - Turkey", "22 Temple Boulevard - Chiang Mai - Thailand",
    "55 Marina Close - Penang - Malaysia", "88 Coral Way - Bali - Indonesia", "33 Pearl Square - Cebu - Philippines",
    "66 Delta Mews - Chittagong - Bangladesh", "99 Silk Place - Lahore - Pakistan", "14 Pyramid Drive - Alexandria - Egypt",
    "47 Safari Road - Mombasa - Kenya", "80 Savannah Avenue - Abuja - Nigeria", "25 Kente Court - Kumasi - Ghana",
    "58 Atlas Lane - Marrakech - Morocco", "91 Lakeshore Boulevard - Houston - USA", "36 Ridgeway Close - Calgary - Canada",
    "69 Bayside Way - Brisbane - Australia", "12 Terrace Square - Christchurch - New Zealand", "45 Cobblestone Mews - Cork - Ireland",
    "78 Cathedral Place - Birmingham - UK", "23 Promenade Drive - Munich - Germany", "56 Montmartre Road - Marseille - France",
    "89 Sakura Avenue - Kyoto - Japan", "34 Bollywood Court - Chennai - India", "67 Hutong Lane - Chengdu - China",
    "10 Hanok Way - Incheon - South Korea", "43 Aztec Boulevard - Guadalajara - Mexico", "76 Carnival Close - Brasilia - Brazil",
    "21 Zulu Square - Durban - South Africa", "54 Windmill Mews - Utrecht - Netherlands", "87 Viking Place - Malmo - Sweden",
    "32 Aurora Drive - Stavanger - Norway", "65 Lighthouse Road - Odense - Denmark", "98 Sauna Court - Turku - Finland",
]
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
    ("rate","float",(47.50,134002.20)),
    ("order_data", "date", ("2023-01-23","2026-03-19")),
    ("delivery_address", "str",addresses ),
    ("payment_method","str",payment)

]

# ── 2. Number of rows to generate ─────────────────────────────────────────────
NUM_ROWS = 10_000

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

df = pd.DataFrame(data)

# ── 4. Output ──────────────────────────────────────────────────────────────────
print(df.head(10).to_string(index=False))
print(f"\nShape : {df.shape}")
print(f"dtypes:\n{df.dtypes}")

df.to_csv("pandas/dataFiles/dummy_sales.csv",index=False)
print("\nSaved → dummy_sales.csv")