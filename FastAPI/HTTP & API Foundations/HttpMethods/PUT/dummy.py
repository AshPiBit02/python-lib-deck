from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

app=FastAPI()

products = [
    {"id":191,"name": "Laptop", "category": "Electronics", "price": 95000, "quantity": 10},
    {"id":192,"name": "Mouse", "category": "Electronics", "price": 1500, "quantity": 25},
    {"id":193,"name": "Chair", "category": "Furniture", "price": 7000, "quantity": 50},
    {"id":194,"name": "Table", "category": "Furniture", "price": 12000, "quantity": 15},
    {"id":195,"name": "Headphones", "category": "Electronics", "price": 5000, "quantity": 40},
]

class Product(BaseModel):
    name:str=Field(min_length=1)
    category:str=Field(min_length=1)
    price:float=Field(gt=0)
    quantity:int=Field(ge=0)

class ProductResponse(BaseModel):
    id:int
    name:str
    category:str
    price:float
    quantity:int
    total_value:float

@app.put("/products/{id}",status_code=201,response_model=ProductResponse)
def update_product(id:int,product:Product):
    for idx,item in enumerate(products):
        if item["id"]==id:
            new_product={
                "id":id,
                **product.model_dump(),
                "total_value":product.price*product.quantity
            }
            products[idx]=new_product
            return new_product
    raise HTTPException(status_code=404,detail=f"Product with id {id} not found!")

@app.get("/products")
def inventory():
    return products
