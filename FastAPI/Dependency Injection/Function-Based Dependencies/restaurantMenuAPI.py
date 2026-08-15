from fastapi import FastAPI,Depends,HTTPException
from pydantic import BaseModel,Field
app=FastAPI()
menu_items:list[dict]=[
     {"id": 1, "name": "Spring Rolls", "price": 6.5, "category": "starter"},
    {"id": 2, "name": "Grilled Salmon", "price": 22.0, "category": "main"},
    {"id": 3, "name": "Chocolate Cake", "price": 8.0, "category": "dessert"},
    {"id": 4, "name": "Iced Tea", "price": 3.5, "category": "drink"},
    {"id": 5, "name": "Caesar Salad", "price": 11.0, "category": "starter"}
]

ALLOWED_CATEGORIES={"starter","main","dessert","drink"}

def get_pagination(skip:int=0,limit:int=10)->dict:
    return {"skip":skip,"limit":limit}

def get_price_filter(min_price:float|None=None,max_price:float|None=None,category:str|None=None)->dict:
    filters:dict={}
    if min_price is not None:
        filters["min_price"]=min_price
    if max_price is not None:
        filters["max_price"]=max_price
    if category:
        filters["category"]=validate_category(category)
    return filters

def validate_category(category:str)->str:
    category=category.lower()
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=400,detail=f"{category} doesn't exists!")
    return category

class Product(BaseModel):
    name:str=Field(...,min_length=4,max_length=20)
    price:float=Field(...,gt=0)
    category:str=Field(...,min_length=4,max_length=20)