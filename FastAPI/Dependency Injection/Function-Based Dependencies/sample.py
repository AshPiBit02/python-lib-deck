from fastapi import FastAPI,Depends
app=FastAPI()

products = [
    {"id": 1, "name": "Wireless Mouse", "price": 25.0, "category": "electronics"},
    {"id": 2, "name": "Mechanical Keyboard", "price": 80.0, "category": "electronics"},
    {"id": 3, "name": "Coffee Mug", "price": 10.0, "category": "kitchen"},
    {"id": 4, "name": "Desk Lamp", "price": 35.0, "category": "furniture"},
    {"id": 5, "name": "Notebook", "price": 5.0, "category": "stationery"},
]

def get_pagination(skip:int=0,limit:int=10)->dict:
    return {"skip":skip,"limit":limit}

def get_search_filters(category:str|None=None,min_price:float|None=None)->dict:
    filters:dict={}
    if category:
        filters["category"]=category
    if min_price is not None:
        filters["min_price"]=min_price
    return filters

@app.get("/products")
def list_products(pagination:dict=Depends(get_pagination)):
    skip=pagination["skip"]
    limit=pagination["limit"]
    return {
        "pagination_used":pagination,
        "results":products[skip:skip+limit],
    }

@app.get("/products/search")
def search_products(
    pagination:dict=Depends(get_pagination),
    filters:dict=Depends(get_search_filters)
):
    result=products

    if "category" in filters:
        result=[p for p in result if p["category"]==filters["category"]]
    if "min_price" in filters:
        result=[p for p in result if p["price"]>=filters["min_price"]]
    skip=pagination["skip"]
    limit=pagination["limit"]

    return {
        "filters_applied":filters,
        "pagination_used":pagination,
        "results":result[skip:skip+limit]
    }