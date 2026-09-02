from pydantic import BaseModel,Field
from decimal import Decimal
from models import TransactionType
from datetime import datetime

class TransactionBase(BaseModel):
    amount:Decimal
    type:TransactionType

class DepositRequest(TransactionBase):
    account_id:int
    amount:Decimal

class WithdrawRequest(TransactionBase):
    account_id:int
    amount:Decimal

class TransferRequest(BaseModel):
    from_account_id:int
    to_account_id:int
    amount:Decimal

class ReversalRequest(TransactionBase):
    transaction_id:int

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
    

