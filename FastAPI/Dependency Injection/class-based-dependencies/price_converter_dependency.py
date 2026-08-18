from fastapi import FastAPI,Depends
from typing import Annotated

app=FastAPI()

class PriceConverter:
    def __init__(self,tax_rate:float=0.13):
        self.tax_rate=tax_rate

    def __call__(self,amount:float)->dict:
        tax=amount*self.tax_rate
        return {"subtotal":amount,"tax":round(tax,2),"total":round(amount+tax,2)}

standard_converter_dependency=Annotated[PriceConverter,Depends(PriceConverter())]
luxury_converter_dependency=Annotated[PriceConverter,Depends(PriceConverter(0.25))]

@app.get("/checkout/standard")
def checkout_standard(breakdown:standard_converter_dependency):
    return breakdown

@app.get("/checkout/luxury")
def checkout_luxury(breakdown:luxury_converter_dependency):
    return breakdown