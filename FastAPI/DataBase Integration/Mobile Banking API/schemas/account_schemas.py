from pydantic import BaseModel,Field,ConfigDict
from datetime import datetime
from typing import Optional
from models import AccountType

class AccountBase(BaseModel):
    account_number:str=Field(...,min_length=20,max_length=20)
    account_type:AccountType

class AccountCreate(AccountBase):
    customer_id:int

class AccountUpdate(AccountBase):
    account_number:Optional[str]=Field(default=None,min_length=20,max_length=20)
    account_type:Optional[AccountType]=None
    customer_id:Optional[int]=None

class AccountResponse(AccountBase):
    id:int
    customer_id:int
    created_at:datetime
    model_config=ConfigDict(from_attributes=True)



