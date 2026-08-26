from fastapi import FastAPI,Depends
from typing import Annotated
app=FastAPI()

class DiscountApplier:
    def __init__(self,percentage:float):
        self.percentage=percentage

    def __call__(self,price:float)->dict:
        discounted_price=price-self.percentage*0.01*price
        return {"Original Price":price,"Discounted Price":round(discounted_price,2)}

member_discount_dependency=Annotated[dict,Depends(DiscountApplier(10))]
vip_discount_dependency=Annotated[dict,Depends(DiscountApplier(25))]

@app.get("/price/member")
def price_for_members(price:member_discount_dependency):
    return price

@app.get("/price/vip")
def price_for_vip(price:vip_discount_dependency):
    return price

    