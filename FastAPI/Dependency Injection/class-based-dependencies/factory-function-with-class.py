from fastapi import FastAPI,Depends
from typing import Annotated

app=FastAPI()

class TaxCalculator:
    def __init__(self,rate:float):
        self.rate=rate

    def __call__(self,price:float)->dict:
        taxed_price=price+self.rate*price*0.01
        return {"Tax Rate":self.rate,"Original Price":price,"Taxed Price":round(taxed_price,2)}

def get_tax_calculator(price:float,tax:float=13)->dict:
    calculator=TaxCalculator(tax)
    return calculator(price)

tax_dependency=Annotated[dict,Depends(get_tax_calculator)]

@app.get("/tax")
def calculate_tax(taxed:tax_dependency):
    return taxed

