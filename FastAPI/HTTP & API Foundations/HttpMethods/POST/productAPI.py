from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

class Product(BaseModel):
    name:str
    category:str
    price:float
    quantity:int

products=[]

@app.post("/products")
def add_product(product:Product):
    new_product={
    "id":len(products)+101,
    **product.model_dump()
    }
    products.append(new_product)
    return {"message":f"{new_product["name"]} ({new_product["id"]}) added to inventory successfully!"}

@app.get("/products")
def product_list():
    return products

@app.get("/products/{id}")
def product_by_id(id:int):
    for product in products:
        if product["id"]==id:
            return product
    return {"message":f"no product found with product id '{id}'"}