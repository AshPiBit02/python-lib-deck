from pydantic import BaseModel,Field,EmailStr,ConfigDict,ValidationError


class Ticket(BaseModel):
    event_code:str=Field(...,min_length=3,max_length=8,pattern=r"^[A-Z]+[0-9]{2,4}$")
    price:float=Field(...,gt=0,multiple_of=0.50
    quantity:int=Field(default=1,ge=1,le=10)
    model_config=ConfigDict(populate_by_name=True)
    purchaser_email:EmailStr=Field(...,alias="purchaseEmail")
try:
    tick1=Ticket(event_code="TEL2254",price=17.5,quantity=6,purchaseEmail="aashishchadhari249@gmail.com")
    print(tick1)
except ValidationError as e:
    print(e.errors())