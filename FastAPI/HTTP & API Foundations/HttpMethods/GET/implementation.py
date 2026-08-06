from fastapi import FastAPI
app=FastAPI()

products = [
    {
        "id": 1,
        "name": "Laptop",
        "category": "Electronics",
        "price": 95000,
        "brand": "Dell",
        "stock": 12
    },
    {
        "id": 2,
        "name": "Mouse",
        "category": "Electronics",
        "price": 1200,
        "brand": "Logitech",
        "stock": 40
    },
    {
        "id": 3,
        "name": "Keyboard",
        "category": "Electronics",
        "price": 2500,
        "brand": "HP",
        "stock": 18
    },
    {
        "id": 4,
        "name": "Notebook",
        "category": "Stationery",
        "price": 120,
        "brand": "Classmate",
        "stock": 150
    },
    {
        "id": 5,
        "name": "Pen",
        "category": "Stationery",
        "price": 25,
        "brand": "Cello",
        "stock": 300
    }
]

@app.get("/products")
def get_products():
    return products

@app.get("/products/{id}")
def productById(id:int):
    for product in products:
        if product["id"]==id:
            return product
    return {"message":f"product not found with id {id}"}

@app.get("/products/search/{key}")
def search_product(key:str):
    results=[product for product in products if key.lower() in product["name"].lower()]
    if results:
        return results
    else:
        return {"message":f"product having key '{key}' not found!"}

@app.get("/products/filter/by_category/{category}")
def category_filter(category:str):
    results=[product for product in products if category.lower()==product["category"].lower()]
    if results:
        return results
    else:
        return {"message":f"no product found in category '{category}'"}

@app.get("/products/filter/by_brand/{brand}")
def brand_filter(brand:str):
    results=[product for product in products if product["brand"].lower()==brand.lower()]
    if results:
        return results
    else:
        return {"message":f"no product found for brand '{brand}'"}

@app.get("/products/page/{page}/limit/{limit}")
def pagination(page:int,limit:int):
    start=(page-1)*limit
    end=start+limit
    return products[start:end]

@app.get("/products/sort/{order}")
def sort_product(order:str):
    if order.lower()=="desc":
        sorted_product=sorted(products,key=lambda x:x["price"],reverse=True)
        return sorted_product
    elif order.lower()=="asc":
        sorted_product=sorted(products,key=lambda x:x["price"])
        return sorted_product
    else:
        return {"message":f"invalid sorting order '{order}'"}

@app.get("/products/filter/category/{category}/brand/{brand}")
def filter1(category:str,brand:str):
    category_exists=any(product["category"].lower()==category.lower() for product in products)
    if not category_exists:
        return {"message":f"Category '{category}' not found!"}
    results=[product for product in products if product["category"].lower()==category.lower() and product["brand"].lower()==brand.lower()]
    if results:
        return results
    else:
        return {"message":f"No products found in category '{category}' with brand '{brand}'"}