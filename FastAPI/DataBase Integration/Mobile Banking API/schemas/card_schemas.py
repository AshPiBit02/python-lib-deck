from pydantic import BaseModel,ConfigDict,Field
from datetime import date
from typing import Optional

class CardBase(BaseModel):
    card_number:str=Field(...,min_length=16,max_length=16)
    expiry_date:date

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

    @classmethod
    def from_orm(cls, obj):
        expiry_str=obj.expiry_date.strftime("%m/%y")
        return cls(
            id=obj.id,
            card_number=obj.card_number,
            expiry_date=expiry_str,
            is_active=obj.is_active
                   )

