from pydantic import BaseModel,Field
from fastapi import FastAPI
app=FastAPI()

class Order(BaseModel):
    order_id:int=Field(gt=0)
    client_name:str=Field(min_length=1)
    amount_due:float=Field(gt=0)
    due_days:int=Field(ge=0)

orders=[]
@app.post("/orders")
def place_order(order:Order):
    orders.append(order.model_dump())
    return {"message":f"Order {order.order_id} placed by {order.client_name} successfully!"}

@app.get("/orders")
def order_list():
    return orders