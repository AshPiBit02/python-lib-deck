# Response Model -> a model that restrics specific field to get returned at client side for security or simiplicity purpose.
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
app=FastAPI()

products = []


class Product(BaseModel):
    name:str=Field(min_length=1)
    category:str=Field(min_length=1)
    price:int=Field(gt=0)
    quantity:int=Field(ge=0)
    internal_code:str=Field(min_length=1)
    secret_field:str=Field(min_length=1)


class ProductResponse(BaseModel):
    id:int
    name:str
    category:str
    price:int
    quantity:int

@app.post("/products",status_code=201,response_model=ProductResponse)
def add_product(product:Product):
    new_product={
        "id":len(products)+101,
        **product.model_dump()
    }
    products.append(new_product)
    return new_product
# will add those two fields(internal_code & secret_field) in the product but not return (display to client)


