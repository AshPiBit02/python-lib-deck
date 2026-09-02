from pydantic import BaseModel,Field,ConfigDict
from datetime import datetime
from typing import Optional,List
from models import AccountType,OwnerRole
from schemas import TransactionResponse
from decimal import Decimal
class AccountBase(BaseModel):
    account_number:str=Field(...,min_length=20,max_length=20)
    account_type:AccountType

class AccountCreate(AccountBase):
    customer_id:int=Field(...)

class AccountUpdate(BaseModel):
    account_number:Optional[str]=Field(default=None,min_length=20,max_length=20)
    account_type:Optional[AccountType]=None
    customer_id:Optional[int]=None

class AccountResponse(AccountBase):
    id:int
    customer_id:int
    created_at:datetime
    model_config=ConfigDict(from_attributes=True)

class AccountWithBalance(AccountResponse):
    balance:Decimal

class AccountWithTransaction(AccountResponse):
    transactions:List[TransactionResponse]

class JointOwnerAdd(BaseModel):
    customer_id:int
    role:OwnerRole

class JointOwnerResponse(JointOwnerAdd):
    account_id:int



