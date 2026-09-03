from decimal import Decimal
from sqlalchemy.orm import Session
from fastapi import HTTPException
from models import Transaction,TransactionType,Account
from schemas import DepositRequest,WithdrawRequest

def get_account_balance(db:Session,account_id:int)->Decimal:
    account=db.query(Account).filter(Account.id==account_id).first()
    if account is None:
        raise HTTPException(status_code=404,detail=f"Account {account_id} not found")
    total=sum((t.amount for t in account.transactions),Decimal("0.00"))
    return total

def deposit(db:Session,request:DepositRequest)->Transaction:
    account=db.query(Account).filter(Account.id==request.account_id).first()
    if account is None:
        raise HTTPException(status_code=404,detail=f"Account {request.account_id} not found")
    new_txn=Transaction(
        account_id=request.account_id,
        amount=request.amount,
        type=TransactionType.deposit,
    )
    db.add(new_txn)
    db.commit()
    db.refresh(new_txn)
    return new_txn

def withdraw(db:Session,request:WithdrawRequest)->Transaction:
    account=db.query(Account).filter(Account.id==request.account_id).first()
    if account is None:
        raise HTTPException(status_code=404,detail=f"Account '{request.account_id}' not found")

    current_balance=get_account_balance(db,request.account_id)
    if current_balance<request.amount:
        raise HTTPException(status_code=400,detail="Insufficient funds")
    new_txn=Transaction(
        account_id=request.account_id,
        amount=-request.amount,
        type=TransactionType.withdrawal,
    )
    db.add(new_txn)
    db.commit()
    db.refresh(new_txn)
    return new_txn

