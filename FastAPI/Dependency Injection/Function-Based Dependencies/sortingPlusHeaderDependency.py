from fastapi import FastAPI,Depends,HTTPException,Header
from dummydata import flights

app=FastAPI()
VALID_FIELDS=["price","duration"]

def get_sort_order(order:str="price")->str:
    order=order.lower()
    if order not in VALID_FIELDS:
        raise HTTPException(status_code=400,detail=f"Unknown order '{order}' cannot sort!")
    return order

def get_client_platform(platform:str=Header(default="web"))->str:
    return platform

@app.get("/flights/search")
def search_fights(client_platform:str=Depends(get_client_platform),sort_order:str=Depends(get_sort_order)):
    result=sorted(flights,key=lambda f:f[sort_order])
    return {
        "Client Platform":client_platform,
        "Result":result
    }
