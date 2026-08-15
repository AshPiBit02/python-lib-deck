from fastapi import FastAPI,Depends,HTTPException
from pydantic import BaseModel,Field
from dummydata import menu_items
app=FastAPI()

ALLOWED_CATEGORIES={"starter","main","dessert","drink"}

def get_pagination(skip:int=0,limit:int=10)->dict:
    return {"skip":skip,"limit":limit}

def get_filters(min_price:float|None=None,max_price:float|None=None,category:str|None=None)->dict:
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

class Item(BaseModel):
    name:str=Field(...,min_length=4,max_length=20)
    price:float=Field(...,gt=0)

@app.get("/menu")
def get_menu(pagination:dict=Depends(get_pagination)):
    skip=pagination["skip"]
    limit=pagination["limit"]
    return menu_items[skip:skip+limit]

@app.get("/menu/filter")
def filter_menu(filters:dict=Depends(get_filters))->dict:
    result=menu_items
    if "min_price" in filters:
        result=[item for item in result if item["price"]>=filters["min_price"]]
    if "max_price" in filters:
        result=[item for item in result if item["price"]<=filters["max_price"]]
    if "category" in filters:
        result=[item for item in result if item["category"]==filters["category"]]
    return {
        "filters":filters,
        "result":result
    }

@app.post("/menu/{category}/items")
def add_item(item:Item,category:str=Depends(validate_category)):
    new_id=max(i["id"] for i in menu_items)+1
    new_item={"id":new_id,"name":item.name,"price":item.price,"category":category}
    menu_items.append(new_item)
    return new_item

@app.get("/menu/search")
def search_menu(items:dict=Depends(filter_menu),pagination:dict=Depends(get_pagination)):
    skip=pagination["skip"]
    limit=pagination["limit"]
    return {
        "pagination":pagination,
        "filters":items["filters"],
        "result":items["result"][skip:skip+limit]
    }

