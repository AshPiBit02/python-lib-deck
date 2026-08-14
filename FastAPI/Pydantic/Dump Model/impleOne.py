from pydantic import BaseModel

class Order(BaseModel):
    order_id:int
    customer_name:str
    total:float
    payment_method:str
    notes:str

order=Order(order_id=102,customer_name="ashpibit",total=1024.52,payment_method="Credit Card",notes="13% VAT included")
print(order)
print(order.model_dump())
print(order.model_dump(include={"order_id","customer_name","total"}))
print(order.model_dump(exclude={"notes"}))