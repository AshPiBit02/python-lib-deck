from pydantic import BaseModel,Field,EmailStr,ConfigDict
from datetime import datetime
from typing import Optional,List
from schemas.account_schemas import AccountResponse

class CustomerBase(BaseModel):
    full_name:str=Field(...,min_length=5)
    email:EmailStr
    phone:str

class CustomerCreate(CustomerBase):
    pass

class CustomerUpdate(BaseModel):
    full_name:Optional[str]=Field(default=None,min_length=5)
    email:Optional[EmailStr]=None
    phone:Optional[str]=None

class CustomerResponse(CustomerBase):
    id:int
    created_at:datetime
    model_config=ConfigDict(from_attributes=True)

class CustomerWithAccounts(CustomerResponse):
    account_list:List[AccountResponse]
