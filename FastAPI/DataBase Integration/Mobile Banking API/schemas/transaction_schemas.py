from pydantic import BaseModel,Field
from decimal import Decimal
from models import TransactionType
from datetime import datetime

class TransactionBase(BaseModel):
    amount:Decimal=Field(gt=0)
    type:TransactionType

class DepositRequest(TransactionBase):
    account_id:int=Field(...)
    amount:Decimal=Field(gt=0)

class WithdrawRequest(TransactionBase):
    account_id:int=Field(...)
    amount:Decimal=Field(gt=0)

class TransferRequest(BaseModel):
    from_account_id:int=Field(...)
    to_account_id:int=Field(...)
    amount:Decimal=Field(gt=0)

class ReversalRequest(TransactionBase):
    transaction_id:int=Field(...)

class TransactionResponse(BaseModel):
    id:int
    account_id:int
    amount:Decimal
    type:TransactionType
    reversed_transaction_id:int
    created_at:datetime

class TransactionHistoryQuery(BaseModel):
    start_date:datetime
    end_date:datetime
    type:TransactionType
    page:int
    page_size:int


