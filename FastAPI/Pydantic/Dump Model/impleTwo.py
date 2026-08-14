from pydantic import BaseModel
class UpdateOrder(BaseModel):
    payment_method:str|None=None
    notes:str|None=None
    total:float|None=None

Updated_order1=UpdateOrder(notes="I don't know")
Updated_order2=UpdateOrder(total=936.56)
Updated_order3=UpdateOrder()

print(Updated_order1.model_dump(exclude_unset=True))
print(Updated_order2.model_dump(exclude_unset=True))
print(Updated_order3.model_dump(exclude_unset=True))

print(Updated_order1.model_dump(exclude_none=True))
print(Updated_order2.model_dump(exclude_none=True))
print(Updated_order3.model_dump(exclude_none=True))

print(type(Updated_order1.model_dump()))
print(type(Updated_order1.model_dump_json()))

