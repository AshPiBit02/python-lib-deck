from pydantic import BaseModel,Field,ConfigDict
from decimal import Decimal
from models import TransactionType
from datetime import datetime
from typing import Optional

class TransactionBase(BaseModel):
    amount:Decimal=Field(gt=0)
    type:TransactionType

class DepositRequest(TransactionBase):
    account_id:int=Field(...)

class WithdrawRequest(TransactionBase):
    account_id:int=Field(...)

class TransferRequest(BaseModel):
    from_account_id:int=Field(...)
    to_account_id:int=Field(...)
    amount:Decimal=Field(...,gt=0)

class ReversalRequest(TransactionBase):
    transaction_id:int=Field(...)

class TransactionResponse(BaseModel):
    id:int
    account_id:int
    amount:Decimal
    type:TransactionType
    reversed_transaction_id:Optional[int]=None
    created_at:datetime
    model_config=ConfigDict(from_attributes=True)

class TransactionHistoryQuery(BaseModel):
    start_date:datetime
    end_date:datetime
    type:Optional[TransactionType]=None
    page:int=1
    page_size:int=20


