from pydantic import BaseModel,Field,ConfigDict
class User(BaseModel):
    model_config=ConfigDict(populate_by_name=True) # allows using either name internally
    full_name:str=Field(...,alias="fullName")

u=User(fullName="Ashpi Bit")
print(u.full_name)

from typing import Annotated

PositiveInt=Annotated[int,Field(gt=0)]

class Product(BaseModel):
    price:PositiveInt
    quantity:PositiveInt
