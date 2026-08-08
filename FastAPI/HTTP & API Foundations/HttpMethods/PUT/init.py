# PUT -> Update or replace an existing resource at a specific location.

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

app=FastAPI()

class Product(BaseModel):
    name:str
    category:str
    price:float
    quantity:int

products= [
    {
        "id": 101,
        "name": "Laptop",
        "category": "Electronics",
        "price": 95000,
        "quantity": 10
    },
    {
        "id": 102,
        "name": "Mouse",
        "category": "Electronics",
        "price": 1500,
        "quantity": 25
    }
]

@app.put("/products/{product_id}")
def update_product(product_id:int,product:Product):
    for index,existing_product in enumerate(products):
        if existing_product["id"]==product_id:
            update_product={
                "id":product_id,
                **product.model_dump()
            }
            products[index]=update_product
            return update_product
    raise HTTPException(status_code=404,detail=f"Product with {product_id} not found!")

@app.get("/products")
def inventory():
    return products