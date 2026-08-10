from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field

app = FastAPI()

products=[{
    "id":101,
    "name":"Laptop",
    "category":"Electronics",
    "price":95000,
    "quantity":10
},
{
    "id":102,
    "name":"Mouse",
    "category":"Electronics",
    "price":1500,
    "quantity":25
}]

class ProductUpdate(BaseModel):
    name:str|None=Field(default=None,min_length=1)
    category:str|None=Field(default=None,min_length=1)
    price:float|None=Field(default=None,gt=0)
    quantity:int|None=Field(default=None,ge=0)

class ProductResponse(BaseModel):
    id:int
    name:str
    category:str
    price:float
    quantity:int

@app.patch("/products/{product_id}",response_model=ProductResponse)
def update_product(product_id:int,product:ProductUpdate):
    for existing_product in products:
        if existing_product["id"]==product_id:
            update_data=product.model_dump(exclude_unset=True)
            existing_product.update(update_data)
            return existing_product
    raise HTTPException(status_code=404,detail=f"Product with id {product_id} not found!")

@app.get("/products")
def inventory():
    return products