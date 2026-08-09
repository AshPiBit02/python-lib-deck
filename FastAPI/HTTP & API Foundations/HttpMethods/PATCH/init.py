from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

app=FastAPI()

products = [
    {"id":191,"name": "Laptop", "category": "Electronics", "price": 95000, "quantity": 10},
    {"id":192,"name": "Mouse", "category": "Electronics", "price": 1500, "quantity": 25},
    {"id":193,"name": "Chair", "category": "Furniture", "price": 7000, "quantity": 50},
    {"id":194,"name": "Table", "category": "Furniture", "price": 12000, "quantity": 15},
    {"id":195,"name": "Headphones", "category": "Electronics", "price": 5000, "quantity": 40},
]
class Product(BaseModel):
    name:str|None=None
    category:str|None=None
    price:float|None=None
    quantity:int|None=None


@app.patch("/products/{product_id}")
def update_product(product_id:int,product:Product):
    for existing_product in products:
        if existing_product["id"]==product_id:
            if product.name is not None:
                existing_product["name"]=product.name
                return {"message":f"Updated product name ({existing_product['name']} -> {product.name})"}
            if product.categroy is not None:
                existing_product["category"]=product.category
                return {"message":f"Updated product category ({existing_product['category']} -> {product.category})"}
            if product.price is not None:
                existing_product["price"]=product.price
                return {"message":f"Updated product price ({existing_product['price']} -> {product.price})"}
            if product.quantity is not None:
                existing_product["quantity"]=product.quantity
                return {"message":f"Updated product quantity ({existing_product['quantity']} -> {product.quantity})"}
    raise HTTPException(status_code=404,detail=f"Product with id {product_id} not found!")
            
@app.get("/products")
def product_list():
    return products