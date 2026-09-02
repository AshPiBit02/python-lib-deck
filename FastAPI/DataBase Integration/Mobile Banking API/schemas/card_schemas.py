from pydantic import BaseModel,ConfigDict,Field
from datetime import date
from typing import Optional

class CardBase(BaseModel):
    card_number:str=Field(...,min_length=16,max_length=16)
    expiry_date:str=Field(...,min_length=7,max_length=7)

class CardCreate(CardBase):
    account_id:int=Field(...)

class CardUpdate(BaseModel):
    is_active:Optional[bool]=True

class CardResponse(BaseModel):
    id:int
    card_number:str
    expiry_date:str
    is_active:bool
    model_config=ConfigDict(from_attributes=True)

