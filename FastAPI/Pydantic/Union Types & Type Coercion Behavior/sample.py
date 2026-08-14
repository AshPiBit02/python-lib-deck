from pydantic import BaseModel,Field
from typing import Literal,Union

# class Payment(BaseModel):
#     amount:int|str

# p1=Payment(amount=100)
# p2=Payment(amount="100")

# print(type(p2.amount),p2.amount)

class CardPayment(BaseModel):
    method:Literal["card"]
    card_number:str

class CashPayment(BaseModel):
    method:Literal["cash"]
    amount_tendered:float

class Order(BaseModel):
    payment:Union[CardPayment,CashPayment]=Field(...,discriminator="method")
    # payment:CardPayment|CashPayment=Field(...,discriminator="method")

print(Order(payment={"method":"card","card_number":"4242XXX"}))
print(Order(payment={"method":"cash","amount_tendered":20.25}))

 